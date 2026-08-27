from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import read_json, sha1_text, write_json_atomic


REQUIRED_FIELDS = ("url", "title", "published_at", "text")


def _records_from_file(path: Path) -> Iterable[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict) and isinstance(value.get("articles"), list):
        yield from value["articles"]
    elif isinstance(value, dict):
        yield value
    else:
        raise ValueError(f"{path}: expected a JSON object or array")


def normalize_article(article: dict[str, Any], source: str = "article") -> dict[str, str]:
    missing = [field for field in REQUIRED_FIELDS if field not in article]
    if missing:
        raise ValueError(f"{source}: missing fields {', '.join(missing)}")
    normalized = {field: article[field] for field in REQUIRED_FIELDS}
    if not all(isinstance(normalized[field], str) for field in REQUIRED_FIELDS):
        raise ValueError(f"{source}: url, title, published_at, and text must be strings")
    if not normalized["url"].strip():
        raise ValueError(f"{source}: url cannot be empty")
    return {"doc_id": sha1_text(normalized["url"]), **normalized}


def ingest_directory(input_dir: str | Path, output_dir: str | Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    source_dir, destination = Path(input_dir), Path(output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"raw input directory does not exist: {source_dir}")
    seen: dict[str, dict[str, str]] = {}
    read_count = 0
    for path in sorted(source_dir.glob("*.json")):
        for index, record in enumerate(_records_from_file(path)):
            read_count += 1
            article = normalize_article(record, f"{path}[{index}]")
            seen.setdefault(article["doc_id"], article)

    articles = [seen[key] for key in sorted(seen)]
    destination.mkdir(parents=True, exist_ok=True)
    for article in articles:
        write_json_atomic(destination / f"{article['doc_id']}.json", article)
    summary = {"read": read_count, "duplicates": read_count - len(articles), "written": len(articles)}
    return articles, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and deduplicate raw article JSON")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/normalized")
    args = parser.parse_args()
    _, summary = ingest_directory(args.input, args.output)
    print(f"Read {summary['read']} articles; duplicates {summary['duplicates']}; wrote {summary['written']}.")


if __name__ == "__main__":
    main()

