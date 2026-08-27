from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import read_json, utc_timestamp, write_json_atomic


Mention = dict[str, Any]


def _coerce_mention(value: Mention) -> Mention:
    return {
        "doc_id": str(value["doc_id"]),
        "start": int(value["start"]),
        "end": int(value["end"]),
        "text": str(value.get("text", "")),
        "label": str(value["label"]),
    }


def load_gold(path: str | Path) -> list[Mention]:
    source = Path(path)
    records: list[Mention] = []
    files = sorted(source.glob("*")) if source.is_dir() else [source]
    for file in files:
        if file.suffix.lower() == ".csv":
            with file.open("r", encoding="utf-8-sig", newline="") as handle:
                records.extend(_coerce_mention(row) for row in csv.DictReader(handle))
        elif file.suffix.lower() == ".json" and file.name != "manifest.json":
            value = read_json(file)
            if isinstance(value, list):
                records.extend(_coerce_mention(item) for item in value)
            elif isinstance(value, dict) and "mentions" in value:
                records.extend(_coerce_mention({"doc_id": value["doc_id"], **item}) for item in value["mentions"])
    return records


def load_predictions(path: str | Path) -> list[Mention]:
    source = Path(path)
    records: list[Mention] = []
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    for file in files:
        value = read_json(file)
        if not isinstance(value, dict) or "doc_id" not in value or "mentions" not in value:
            continue
        records.extend(_coerce_mention({"doc_id": value["doc_id"], **item}) for item in value["mentions"])
    return records


def strict_match(predicted: Mention, gold: Mention) -> bool:
    return (
        predicted["doc_id"] == gold["doc_id"]
        and predicted["label"] == gold["label"]
        and predicted["start"] == gold["start"]
        and predicted["end"] == gold["end"]
    )


def relaxed_match(predicted: Mention, gold: Mention) -> bool:
    return (
        predicted["doc_id"] == gold["doc_id"]
        and predicted["label"] == gold["label"]
        and predicted["start"] < gold["end"]
        and gold["start"] < predicted["end"]
    )


def _maximum_matches(predicted: list[Mention], gold: list[Mention], matches: Callable[[Mention, Mention], bool]) -> int:
    adjacency = [[index for index, target in enumerate(gold) if matches(item, target)] for item in predicted]
    assigned: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in assigned or augment(assigned[gold_index], visited):
                assigned[gold_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predicted)))


def _metric(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def score_mentions(predicted: list[Mention], gold: list[Mention]) -> dict[str, Any]:
    labels = sorted({item["label"] for item in [*predicted, *gold]})
    output: dict[str, Any] = {"strict": {}, "relaxed": {}}
    for mode, matcher in (("strict", strict_match), ("relaxed", relaxed_match)):
        total_tp = 0
        for label in labels:
            pred_label = [item for item in predicted if item["label"] == label]
            gold_label = [item for item in gold if item["label"] == label]
            tp = _maximum_matches(pred_label, gold_label, matcher)
            total_tp += tp
            output[mode][label] = _metric(tp, len(pred_label), len(gold_label))
        output[mode]["OVERALL"] = _metric(total_tp, len(predicted), len(gold))
    return output


def _print_table(scores: dict[str, Any]) -> None:
    for mode in ("strict", "relaxed"):
        print(f"\n{mode.upper()}")
        print(f"{'Label':24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'Pred':>6} {'Gold':>6}")
        for label, row in scores[mode].items():
            print(
                f"{label:24} {row['precision']:10.3f} {row['recall']:10.3f} {row['f1']:10.3f} "
                f"{row['true_positive']:6d} {row['predicted']:6d} {row['gold']:6d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predicted entity mentions")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--results", default="eval/results")
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    predicted, gold = load_predictions(args.pred), load_gold(args.gold)
    scores = score_mentions(predicted, gold)
    result = {"timestamp": utc_timestamp(), "config": args.config, "scores": scores}
    _print_table(scores)
    if not args.no_write:
        path = Path(args.results) / f"{result['timestamp']}_{args.config}.json"
        write_json_atomic(path, result)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

