from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.score import load_gold, score_mentions


def load_with_confidence(path: str | Path) -> list[dict]:
    mentions = []
    for file in sorted(Path(path).glob("*.json")):
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "doc_id" not in value:
            continue
        for item in value.get("mentions", []):
            mentions.append({"doc_id": value["doc_id"], **item})
    return mentions


def write_svg(path: Path, rows: list[dict]) -> None:
    width, height, margin = 640, 400, 50
    x = lambda threshold: margin + (threshold - 0.3) / 0.6 * (width - 2 * margin)
    y = lambda value: height - margin - value * (height - 2 * margin)
    precision = " ".join(f"{x(row['threshold']):.1f},{y(row['precision']):.1f}" for row in rows)
    recall = " ".join(f"{x(row['threshold']):.1f},{y(row['recall']):.1f}" for row in rows)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
<text x="{width/2}" y="{height-10}" text-anchor="middle">Confidence threshold</text><text x="15" y="{height/2}" transform="rotate(-90 15 {height/2})" text-anchor="middle">Score</text>
<polyline points="{precision}" fill="none" stroke="#1565c0" stroke-width="3"/><polyline points="{recall}" fill="none" stroke="#c62828" stroke-width="3"/>
<text x="{width-150}" y="25" fill="#1565c0">Precision</text><text x="{width-75}" y="25" fill="#c62828">Recall</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mention confidence and plot precision/recall")
    parser.add_argument("--pred", required=True, help="Predictions produced with the lowest threshold (0.3)")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-prefix", default="eval/results/threshold_sweep")
    args = parser.parse_args()
    predictions, gold = load_with_confidence(args.pred), load_gold(args.gold)
    rows = []
    for integer in range(3, 10):
        threshold = integer / 10
        filtered = [item for item in predictions if float(item.get("confidence", 0.0)) >= threshold]
        metric = score_mentions(filtered, gold)["strict"]["OVERALL"]
        rows.append({"threshold": threshold, "precision": metric["precision"], "recall": metric["recall"], "f1": metric["f1"]})
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("threshold", "precision", "recall", "f1"))
        writer.writeheader()
        writer.writerows(rows)
    write_svg(prefix.with_suffix(".svg"), rows)
    print(f"Wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

