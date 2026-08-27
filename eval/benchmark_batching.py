from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-document and batched spaCy inference")
    parser.add_argument("--input", default="data/normalized")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        import spacy
    except ImportError as error:
        raise SystemExit("Install spaCy before running this benchmark") from error
    texts = [json.loads(path.read_text(encoding="utf-8"))["text"] for path in sorted(Path(args.input).glob("*.json"))]
    nlp = spacy.load(args.model)
    started = time.perf_counter()
    for text in texts:
        nlp(text)
    individual = time.perf_counter() - started
    started = time.perf_counter()
    list(nlp.pipe(texts, batch_size=args.batch_size))
    batched = time.perf_counter() - started
    speedup = individual / batched if batched else 0.0
    print(json.dumps({"documents": len(texts), "individual_seconds": individual, "batched_seconds": batched, "speedup": speedup}, indent=2))


if __name__ == "__main__":
    main()

