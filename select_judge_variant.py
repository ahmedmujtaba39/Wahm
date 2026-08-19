"""Select a Judge input variant using validation results only."""

import argparse
import json
from pathlib import Path


def select(metadata_by_variant):
    if set(metadata_by_variant) != {
            "answer_only", "gold_answer", "question_gold_answer"}:
        raise ValueError("all three input variants are required")
    for variant, metadata in metadata_by_variant.items():
        if metadata.get("test_metrics") is not None:
            raise ValueError(f"{variant} exposed test metrics before selection")
        if metadata.get("input_variant") != variant:
            raise ValueError(f"metadata variant mismatch for {variant}")
    ranking = sorted(
        metadata_by_variant,
        key=lambda variant: (
            metadata_by_variant[variant]["validation_metrics"]["roc_auc"],
            metadata_by_variant[variant]["validation_metrics"]["f1"],
            variant,
        ),
        reverse=True,
    )
    return ranking[0], ranking


def run(root, output):
    root = Path(root)
    metadata = {
        variant: json.loads((root / variant / "judge_metadata.json").read_text(
            encoding="utf-8"))
        for variant in ("answer_only", "gold_answer", "question_gold_answer")
    }
    winner, ranking = select(metadata)
    result = {
        "selection_partition": "validation",
        "primary_metric": "roc_auc",
        "tie_break_metric": "f1_at_validation_selected_threshold",
        "winner": winner,
        "ranking": ranking,
        "validation_results": {
            variant: {
                "decision_threshold": metadata[variant]["decision_threshold"],
                **metadata[variant]["validation_metrics"],
            }
            for variant in ranking
        },
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.output)
