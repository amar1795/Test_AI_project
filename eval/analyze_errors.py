from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.score import load_gold, load_predictions, strict_match
from src.utils import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Export strict NER errors for manual categorization")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default="eval/results/error_analysis.json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    predicted, gold = load_predictions(args.pred), load_gold(args.gold)
    false_positives = [item for item in predicted if not any(strict_match(item, target) for target in gold)]
    false_negatives = [item for item in gold if not any(strict_match(target, item) for target in predicted)]
    write_json_atomic(
        args.output,
        {
            "false_positives": false_positives[:args.limit],
            "false_negatives": false_negatives[:args.limit],
            "category_notes": {},
        },
    )
    print(f"Wrote {min(len(false_positives), args.limit)} false positives and {min(len(false_negatives), args.limit)} false negatives.")


if __name__ == "__main__":
    main()

