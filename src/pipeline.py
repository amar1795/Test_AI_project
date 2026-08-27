from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import JsonCache
from src.config import load_config
from src.disambiguate_llm import OpenAICompatibleClient, disambiguate
from src.extract_gliner import extract_mentions as extract_gliner, merge_mentions
from src.extract_keyword import extract_mentions as extract_keyword
from src.extract_spacy import extract_mentions as extract_spacy
from src.ingest import ingest_directory
from src.resolve import EntityRegistry, ResolutionResult
from src.schema import validate_document
from src.segment import regex_segment, spacy_segment
from src.utils import stable_hash, write_json_atomic


def _stage_dump(stage: str, document: dict[str, Any], enabled: bool) -> None:
    if enabled:
        print(f"\n--- {stage}: {document['doc_id']} ---")
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def _base_document(article: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    recorded_config = {
        key: config.get(key)
        for key in (
            "segmenter", "extractor", "spacy_model", "gliner_model", "gliner_threshold",
            "fuzzy_accept_threshold", "fuzzy_ambiguous_threshold", "llm_enabled",
            "llm_model", "prompt_template_version",
        )
    }
    return {
        **article,
        "sentences": [],
        "mentions": [],
        "entities": [],
        "pipeline_version": config["pipeline_version"],
        "config": recorded_config,
    }


def _segment(document: dict[str, Any], config: dict[str, Any]) -> None:
    if config["segmenter"] == "regex":
        document["sentences"] = regex_segment(document["text"])
    elif config["segmenter"] == "spacy":
        document["sentences"] = spacy_segment(document["text"], config["spacy_model"])
    else:
        raise ValueError(f"unknown segmenter: {config['segmenter']}")


def _extract(document: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    extractor = config["extractor"]
    common = (document["text"], document["sentences"])
    if extractor == "keyword":
        return extract_keyword(*common, config["keyword_entities"])
    if extractor == "spacy":
        return extract_spacy(*common, config["spacy_model"], config["spacy_label_map"])
    if extractor == "gliner":
        return extract_gliner(
            *common,
            config["gliner_model"], config["gliner_labels"], config["gliner_label_map"], config["gliner_threshold"],
        )
    if extractor == "merged":
        spacy_mentions = extract_spacy(*common, config["spacy_model"], config["spacy_label_map"])
        gliner_mentions = extract_gliner(
            *common,
            config["gliner_model"], config["gliner_labels"], config["gliner_label_map"], config["gliner_threshold"],
        )
        core = {"PERSON", "ORG", "GPE"}
        gliner_domain = [mention for mention in gliner_mentions if mention["label"] not in core]
        return [*spacy_mentions, *gliner_domain]
    raise ValueError(f"unknown extractor: {extractor}")


def _llm_client(config: dict[str, Any]):
    if not config.get("llm_enabled"):
        return None
    endpoint = config.get("llm_base_url") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = config.get("llm_model") or os.getenv("LLM_MODEL")
    if not all((endpoint, api_key, model)):
        raise RuntimeError("LLM mode requires LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL")
    return OpenAICompatibleClient(str(endpoint), str(api_key), str(model))


def _entity_output(result: ResolutionResult, mention_id: str) -> dict[str, Any]:
    return {
        "entity_id": result.entity_id,
        "canonical": result.canonical,
        "label": result.label,
        "mention_ids": [mention_id],
        "resolution_method": result.method,
        "confidence": result.confidence,
    }


def _add_resolution(entity_map: dict[str, dict[str, Any]], result: ResolutionResult, mention_id: str) -> None:
    assert result.entity_id is not None
    if result.entity_id not in entity_map:
        entity_map[result.entity_id] = _entity_output(result, mention_id)
        return
    entity = entity_map[result.entity_id]
    entity["mention_ids"].append(mention_id)
    methods = set(entity["resolution_method"].removeprefix("mixed:").split("+"))
    methods.add(result.method)
    entity["resolution_method"] = next(iter(methods)) if len(methods) == 1 else "mixed:" + "+".join(sorted(methods))
    entity["confidence"] = min(entity["confidence"], result.confidence)


def run_pipeline(
    input_dir: str | Path = "data/raw",
    normalized_dir: str | Path = "data/normalized",
    output_dir: str | Path = "data/output",
    registry_path: str | Path = "registry/entities.json",
    cache_path: str | Path = ".cache/pipeline-cache.json",
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    trace_stages: bool = False,
    llm_client=None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    config.update(overrides or {})
    articles, ingest_summary = ingest_directory(input_dir, normalized_dir)
    articles.sort(key=lambda article: (article["published_at"], article["doc_id"]))
    registry = EntityRegistry(registry_path, config)
    cache = JsonCache(cache_path)
    client = llm_client if llm_client is not None else _llm_client(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    mention_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    review_queue: list[dict[str, Any]] = []
    llm_calls = 0
    output_ids: list[str] = []

    for article in articles:
        document = _base_document(article, config)
        validate_document(document, "ingest")
        _stage_dump("ingest", document, trace_stages)

        _segment(document, config)
        validate_document(document, "segment")
        _stage_dump("segment", document, trace_stages)

        extraction_version = stable_hash(
            (
                json.dumps(config.get("keyword_entities", {}), sort_keys=True),
                json.dumps(config.get("spacy_label_map", {}), sort_keys=True),
                json.dumps(config.get("gliner_labels", []), sort_keys=True),
                str(config.get("gliner_threshold")),
                str(config.get("segmenter")),
            )
        )
        cache_key = cache.make_key(
            "extract", str(config["extractor"]), extraction_version,
            str(config.get("prompt_template_version", "1")), document["text"],
        )
        raw_mentions = cache.get(cache_key)
        if raw_mentions is None:
            raw_mentions = _extract(document, config)
            cache.set(cache_key, raw_mentions)
        document["mentions"] = merge_mentions([dict(mention) for mention in raw_mentions])
        validate_document(document, "extract")
        _stage_dump("extract", document, trace_stages)

        entity_map: dict[str, dict[str, Any]] = {}
        for mention in document["mentions"]:
            mention_counts[mention["label"]] += 1
            result = registry.resolve(mention, document)
            if result.status == "ambiguous" and client and result.candidate_entity_ids:
                candidates = [registry.entities[entity_id] for entity_id in result.candidate_entity_ids[:3]]
                llm_calls += 1
                llm_result = disambiguate(client, mention, document, candidates)
                if llm_result and llm_result["entity_id"] is not None:
                    result = registry.accept_disambiguation(
                        llm_result["entity_id"], mention, document, float(llm_result["confidence"])
                    )
            if result.status == "resolved":
                method_counts[result.method] += 1
                _add_resolution(entity_map, result, mention["id"])
            else:
                method_counts["ambiguous"] += 1
                sentence = next(
                    (item["text"] for item in document["sentences"] if item["id"] == mention["sent_id"]), ""
                )
                review_queue.append(
                    {
                        "doc_id": document["doc_id"],
                        "mention": mention,
                        "sentence": sentence,
                        "reason": result.method,
                        "candidate_entity_ids": list(result.candidate_entity_ids),
                    }
                )

        document["entities"] = [entity_map[key] for key in sorted(entity_map)]
        validate_document(document, "resolve")
        _stage_dump("resolve", document, trace_stages)
        write_json_atomic(destination / f"{document['doc_id']}.json", document)
        output_ids.append(document["doc_id"])

    registry.save()
    cache.save()
    write_json_atomic(destination / "review_queue.json", review_queue)
    manifest = {
        "pipeline_version": config["pipeline_version"],
        "config": _base_document(
            {"doc_id": "", "url": "", "title": "", "published_at": "", "text": ""}, config
        )["config"],
        "document_ids": output_ids,
    }
    write_json_atomic(destination / "manifest.json", manifest)
    summary = {
        "documents_processed": len(articles),
        "ingest": ingest_summary,
        "mentions_per_label": dict(sorted(mention_counts.items())),
        "resolutions_by_method": dict(sorted(method_counts.items())),
        "ambiguous_count": len(review_queue),
        "cache_hit_rate": round(cache.hit_rate, 4),
        "llm_calls": llm_calls,
        "wall_clock_seconds": round(time.perf_counter() - started, 4),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity extraction and cross-document resolution")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--normalized", default="data/normalized")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--registry", default="registry/entities.json")
    parser.add_argument("--cache", default=".cache/pipeline-cache.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--segmenter", choices=("regex", "spacy"))
    parser.add_argument("--extractor", choices=("keyword", "spacy", "gliner", "merged"))
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--trace-stages", action="store_true")
    args = parser.parse_args()
    overrides = {key: value for key, value in {"segmenter": args.segmenter, "extractor": args.extractor}.items() if value}
    if args.enable_llm:
        overrides["llm_enabled"] = True
    summary = run_pipeline(
        args.input, args.normalized, args.output, args.registry, args.cache,
        args.config, overrides, args.trace_stages,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

