"""Compute Layer 3 inter-annotator and automatic-vs-human agreement."""

import argparse
import csv
import json
from pathlib import Path


def parse_label(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "hallucinated", "hallucination", "h"}:
        return 1
    if normalized in {"0", "clean", "correct", "c"}:
        return 0
    return None


def cohen_kappa(left, right):
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_positive = sum(left) / len(left)
    right_positive = sum(right) / len(right)
    expected = (left_positive * right_positive
                + (1 - left_positive) * (1 - right_positive))
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def _load(path, label_column):
    with open(path, encoding="utf-8-sig", newline="") as source:
        return {row["sample_id"]: (parse_label(row.get(label_column)), row)
                for row in csv.DictReader(source)}


def run(key_path, validator_a_path, validator_b_path, output_path):
    key = _load(key_path, "combined_decision")
    validator_a = _load(validator_a_path, "human_label")
    validator_b = _load(validator_b_path, "human_label")
    shared = sorted(set(key) & set(validator_a) & set(validator_b))
    completed = [sample_id for sample_id in shared
                 if validator_a[sample_id][0] is not None
                 and validator_b[sample_id][0] is not None]
    if not completed:
        raise SystemExit("ERROR: no samples have labels from both validators")

    automatic = [key[sample_id][0] for sample_id in completed]
    labels_a = [validator_a[sample_id][0] for sample_id in completed]
    labels_b = [validator_b[sample_id][0] for sample_id in completed]
    agreed = [index for index in range(len(completed))
              if labels_a[index] == labels_b[index]]
    consensus = [labels_a[index] for index in agreed]
    automatic_consensus = [automatic[index] for index in agreed]
    result = {
        "n_keyed": len(key),
        "n_double_annotated": len(completed),
        "double_annotation_coverage": round(len(completed) / len(key), 6),
        "inter_annotator_kappa": round(cohen_kappa(labels_a, labels_b), 6),
        "raw_inter_annotator_agreement": round(
            len(agreed) / len(completed), 6),
        "n_consensus_without_adjudication": len(consensus),
        "automatic_vs_validator_a_kappa": round(
            cohen_kappa(automatic, labels_a), 6),
        "automatic_vs_validator_b_kappa": round(
            cohen_kappa(automatic, labels_b), 6),
        "automatic_vs_consensus_kappa": round(
            cohen_kappa(automatic_consensus, consensus), 6) if consensus else None,
        "automatic_vs_consensus_accuracy": round(sum(
            left == right for left, right in zip(automatic_consensus, consensus))
            / len(consensus), 6) if consensus else None,
        "n_requires_adjudication": len(completed) - len(consensus),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--validator-a", required=True)
    parser.add_argument("--validator-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.key, args.validator_a, args.validator_b, args.output)
