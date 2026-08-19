"""Evaluate Judge v2 against two human validators and adjudicated consensus."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABELS = ("clean", "factual_hallucination", "degeneration")


def parse_label(value):
    normalized = str(value).strip().lower()
    aliases = {
        "clean": "clean", "correct": "clean", "0": "clean",
        "factual_hallucination": "factual_hallucination",
        "hallucinated": "factual_hallucination", "hallucination": "factual_hallucination",
        "1": "factual_hallucination",
        "degeneration": "degeneration", "invalid": "degeneration", "2": "degeneration",
    }
    return aliases.get(normalized)


def _load(path, column):
    with open(path, encoding="utf-8-sig", newline="") as source:
        return {row["sample_id"]: (parse_label(row.get(column)), row)
                for row in csv.DictReader(source)}


def _multiclass_kappa(left, right):
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum((left_counts[label] / len(left))
                   * (right_counts[label] / len(right)) for label in LABELS)
    return ((observed - expected) / (1 - expected)
            if expected != 1 else (1.0 if observed == 1 else 0.0))


def _metrics(actual, predicted):
    confusion = {label: Counter() for label in LABELS}
    for truth, guess in zip(actual, predicted):
        confusion[truth][guess] += 1
    per_class = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[truth][label]
                             for truth in LABELS if truth != label)
        false_negative = sum(confusion[label][guess]
                             for guess in LABELS if guess != label)
        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": round(sum(a == p for a, p in zip(actual, predicted))
                          / len(actual), 6),
        "macro_f1": round(sum(item["f1"] for item in per_class.values())
                          / len(LABELS), 6),
        "per_class": per_class,
        "confusion": {truth: dict(confusion[truth]) for truth in LABELS},
    }


def run(key_path, validator_a_path, validator_b_path, output_path,
        adjudicated_path=None):
    key = _load(key_path, "combined_decision")
    validator_a = _load(validator_a_path, "human_label")
    validator_b = _load(validator_b_path, "human_label")
    shared = sorted(set(key) & set(validator_a) & set(validator_b))
    completed = [sample_id for sample_id in shared
                 if validator_a[sample_id][0] and validator_b[sample_id][0]]
    if not completed:
        raise SystemExit("ERROR: no rows have labels from both validators")

    labels_a = [validator_a[sample_id][0] for sample_id in completed]
    labels_b = [validator_b[sample_id][0] for sample_id in completed]
    agreement = sum(left == right for left, right in zip(labels_a, labels_b))
    result = {
        "n_keyed": len(key),
        "n_double_annotated": len(completed),
        "double_annotation_coverage": round(len(completed) / len(key), 6),
        "raw_inter_annotator_agreement": round(agreement / len(completed), 6),
        "inter_annotator_kappa": round(
            _multiclass_kappa(labels_a, labels_b), 6),
        "n_requires_adjudication": len(completed) - agreement,
    }

    if adjudicated_path:
        adjudicated = _load(adjudicated_path, "human_label")
        final_ids = [sample_id for sample_id in completed
                     if adjudicated.get(sample_id, (None,))[0]]
        actual = [adjudicated[sample_id][0] for sample_id in final_ids]
    else:
        final_ids = [sample_id for sample_id in completed
                     if validator_a[sample_id][0] == validator_b[sample_id][0]]
        actual = [validator_a[sample_id][0] for sample_id in final_ids]
    predicted = [key[sample_id][0] for sample_id in final_ids]
    result["n_human_consensus"] = len(final_ids)
    result["pipeline_vs_human"] = _metrics(actual, predicted) if final_ids else None

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
    parser.add_argument("--adjudicated", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.key, args.validator_a, args.validator_b, args.output,
        args.adjudicated)
