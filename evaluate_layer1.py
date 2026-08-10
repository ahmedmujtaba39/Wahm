"""Evaluate Layer 1 against AraHalluEval's human binary labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

from judge_data import load_judge_rows
from score_layer1 import score_answer


def run(data="judge_train.csv", output="results/layer1_arahallueval.json",
        clean_threshold=0.9):
    rows = load_judge_rows(data)
    decisions = []
    degeneration = Counter()
    for row in rows:
        coverage, decision, reasons = score_answer(
            row["answer"], row["gold_answer"], clean_threshold)
        predicted = {"clean": 0, "hallucinated": 1}.get(decision)
        decisions.append((row["label"], predicted, coverage, reasons))
        degeneration.update(reasons)

    resolved = [item for item in decisions if item[1] is not None]
    correct = sum(truth == predicted for truth, predicted, _, _ in resolved)
    confusion = Counter((truth, predicted) for truth, predicted, _, _ in resolved)
    result = {
        "data": data,
        "n_valid_human_labels": len(rows),
        "clean_threshold": clean_threshold,
        "n_resolved": len(resolved),
        "resolution_rate": round(len(resolved) / len(rows), 6),
        "accuracy_on_resolved": round(correct / len(resolved), 6),
        "n_deferred": len(rows) - len(resolved),
        "confusion": {
            "true_clean_pred_clean": confusion[(0, 0)],
            "true_clean_pred_hallucinated": confusion[(0, 1)],
            "true_hallucinated_pred_clean": confusion[(1, 0)],
            "true_hallucinated_pred_hallucinated": confusion[(1, 1)],
        },
        "degeneration_flags": dict(degeneration),
        "note": ("This evaluates against the source labels, including their known "
                 "inconsistency on degenerate outputs."),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"resolved={len(resolved)}/{len(rows)} "
          f"({result['resolution_rate']:.1%}) "
          f"accuracy={result['accuracy_on_resolved']:.1%}")
    print(f"wrote {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="judge_train.csv")
    parser.add_argument("--output", default="results/layer1_arahallueval.json")
    parser.add_argument("--clean-threshold", type=float, default=0.9)
    args = parser.parse_args()
    run(args.data, args.output, args.clean_threshold)
