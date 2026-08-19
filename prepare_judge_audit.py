"""Create a blinded, stratified Judge v2 manual-audit packet."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


HUMAN_LABELS = "clean | factual_hallucination | degeneration"
ERROR_TYPES = (
    "wrong_entity | wrong_number_date | contradiction | unsupported_elaboration | "
    "generic_imprecise | instruction_mismatch | degeneration | other"
)


def audit_route(row):
    decision = row["combined_decision"]
    if decision == "degeneration":
        return "layer1_degeneration"
    if decision == "clean":
        return "layer2_clean"
    if decision in {"factual_hallucination", "hallucinated"}:
        return "layer2_hallucination"
    raise ValueError(f"unexpected combined decision: {decision!r}")


def threshold_band(row):
    probability = row.get("layer2_hallucination_probability", "").strip()
    threshold = row.get("layer2_decision_threshold", "").strip()
    if not probability:
        return "not_applicable"
    distance = abs(float(probability) - float(threshold or 0.5))
    if distance <= 0.05:
        return "near_0.05"
    if distance <= 0.20:
        return "near_0.20"
    return "far"


def stratified_sample(rows, size=250, seed=42):
    """Balance varieties/routes and deliberately cover threshold-near rows."""
    strata = defaultdict(list)
    for row in rows:
        key = (row["variety"], audit_route(row), threshold_band(row))
        strata[key].append(row)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)

    target = min(size, len(rows))
    allocation = {key: 0 for key in strata}
    ordered_keys = sorted(
        strata, key=lambda key: (key[2] != "near_0.05", key))
    if target >= len(ordered_keys):
        allocation = {key: 1 for key in strata}
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [key for key in ordered_keys
                    if allocation[key] < len(strata[key])]
        if not eligible:
            break
        key = min(eligible, key=lambda candidate: (
            allocation[candidate] / min(len(strata[candidate]), target),
            candidate[2] != "near_0.05",
            candidate))
        allocation[key] += 1
        remaining -= 1

    selected = []
    for key in ordered_keys:
        selected.extend(strata[key][:allocation[key]])
    rng.shuffle(selected)
    return selected


def run(input_paths, output_dir="judge_v2_audit", size=250, seed=42):
    rows = []
    for input_path in input_paths:
        with open(input_path, encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                record = dict(row)
                record["source_file"] = str(input_path)
                rows.append(record)
    selected = stratified_sample(rows, size, seed)
    if not selected:
        raise SystemExit("ERROR: no scored rows were available for audit")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    keyed = []
    for index, row in enumerate(selected, 1):
        record = dict(row)
        record["sample_id"] = f"audit_{index:03d}"
        record["audit_route"] = audit_route(row)
        record["threshold_band"] = threshold_band(row)
        keyed.append(record)

    key_fields = ["sample_id"] + [field for field in keyed[0]
                                   if field != "sample_id"]
    with (root / "judge_v2_audit_key.csv").open(
            "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(keyed)

    blind_fields = ["sample_id", "qid", "variety", "question", "gold_answer",
                    "answer", "human_label", "error_type", "comments"]
    blind_rows = []
    for row in keyed:
        record = {field: row.get(field, "") for field in blind_fields}
        record["human_label"] = ""
        record["error_type"] = ""
        blind_rows.append(record)
    for validator_index, validator in enumerate(("a", "b")):
        ordered = list(blind_rows)
        random.Random(seed + validator_index + 1).shuffle(ordered)
        with (root / f"judge_v2_audit_validator_{validator}.csv").open(
                "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=blind_fields)
            writer.writeheader()
            writer.writerows(ordered)

    (root / "README.md").write_text(
        "# Judge v2 manual audit\n\n"
        f"This packet contains {len(keyed)} blinded rows sampled with seed {seed}.\n\n"
        f"Allowed `human_label` values: `{HUMAN_LABELS}`.\n\n"
        f"Suggested `error_type` values: `{ERROR_TYPES}`.\n\n"
        "Validators must not inspect `judge_v2_audit_key.csv` until both blinded "
        "files are complete. Resolve disagreements through adjudication before "
        "reporting final pipeline precision, recall, or F1.\n",
        encoding="utf-8")
    print(f"wrote {len(keyed)} blinded audit rows to {root}")
    return keyed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="judge_v2_audit")
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.inputs, args.output_dir, args.size, args.seed)
