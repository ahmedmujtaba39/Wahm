"""Create blinded, reproducible Layer 3 annotation sheets and a private key."""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def _route(row):
    return "layer2" if row.get("layer2_hallucination_probability", "") else "layer1"


def stratified_sample(rows, size, seed=42):
    """Sample across model, automatic label, and judge route."""
    strata = defaultdict(list)
    for row in rows:
        key = (row["model"], row["combined_decision"], _route(row))
        strata[key].append(row)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)

    target = min(size, len(rows))
    allocation = {key: 0 for key in strata}
    # Cover every available stratum once when the requested sample permits it.
    if target >= len(strata):
        allocation = {key: 1 for key in strata}
    remaining = target - sum(allocation.values())
    while remaining:
        eligible = [key for key, group in strata.items()
                    if allocation[key] < len(group)]
        if not eligible:
            break
        key = max(eligible, key=lambda candidate:
                  len(strata[candidate]) / len(rows) - allocation[candidate] / target)
        allocation[key] += 1
        remaining -= 1

    selected = []
    for key in sorted(strata):
        selected.extend(strata[key][:allocation[key]])
    rng.shuffle(selected)
    return selected


def run(input_path="scores_combined.csv", output_dir="layer3",
        per_dialect=150, seed=42, conditions=("direct",)):
    with open(input_path, encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row["variety"] != "msa" and row["condition"] in conditions]
    dialects = sorted({row["variety"] for row in rows})
    if not dialects:
        raise SystemExit("ERROR: no dialect rows matched the requested conditions")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for dialect_index, dialect in enumerate(dialects):
        candidates = [row for row in rows if row["variety"] == dialect]
        selected = stratified_sample(
            candidates, per_dialect, seed + dialect_index)
        keyed = []
        for index, row in enumerate(selected, 1):
            record = dict(row)
            record["sample_id"] = f"{dialect}_{index:03d}"
            record["judge_route"] = _route(row)
            keyed.append(record)

        key_fields = ["sample_id"] + list(selected[0]) + ["judge_route"]
        with (root / f"layer3_{dialect}_key.csv").open(
                "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=key_fields)
            writer.writeheader()
            writer.writerows(keyed)

        blind_fields = ["sample_id", "qid", "question", "gold_answer", "answer",
                        "human_label", "comments"]
        blind_rows = [{field: row.get(field, "") for field in blind_fields}
                      for row in keyed]
        for validator_index, validator in enumerate(("a", "b")):
            ordered = list(blind_rows)
            random.Random(seed + dialect_index * 10 + validator_index).shuffle(ordered)
            with (root / f"layer3_{dialect}_validator_{validator}.csv").open(
                    "w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=blind_fields)
                writer.writeheader()
                writer.writerows(ordered)
        print(f"{dialect}: wrote {len(keyed)} blinded samples + private key")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scores_combined.csv")
    parser.add_argument("--output-dir", default="layer3")
    parser.add_argument("--per-dialect", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", default=["direct"])
    args = parser.parse_args()
    run(args.input, args.output_dir, args.per_dialect,
        args.seed, tuple(args.conditions))
