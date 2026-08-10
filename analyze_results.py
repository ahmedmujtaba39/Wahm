"""Compute paired hallucination drift, error overlap, and degeneration rates."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _label(row):
    decision = row.get("combined_decision") or row.get("layer1_decision")
    if decision not in {"clean", "hallucinated"}:
        return None
    return int(decision == "hallucinated")


def analyze(rows):
    indexed = {}
    for row in rows:
        label = _label(row)
        if label is not None:
            key = (row["model"], row["variety"], row["condition"], row["qid"])
            indexed[key] = (label, row)

    arms = defaultdict(set)
    for model, variety, condition, qid in indexed:
        if variety != "msa":
            arms[(model, variety, condition)].add(qid)

    results = []
    for (model, variety, condition), dialect_qids in sorted(arms.items()):
        msa_qids = {qid for candidate_model, candidate_variety,
                    candidate_condition, qid in indexed
                    if candidate_model == model and candidate_variety == "msa"
                    and candidate_condition == "direct"}
        paired = sorted(dialect_qids & msa_qids)
        if not paired:
            continue
        dialect_hallucinated = {
            qid for qid in paired
            if indexed[(model, variety, condition, qid)][0] == 1}
        msa_hallucinated = {
            qid for qid in paired
            if indexed[(model, "msa", "direct", qid)][0] == 1}
        union = dialect_hallucinated | msa_hallucinated
        degeneration_count = sum(bool(indexed[
            (model, variety, condition, qid)][1].get("degeneration_reasons", ""))
            for qid in paired)
        dialect_rate = len(dialect_hallucinated) / len(paired)
        msa_rate = len(msa_hallucinated) / len(paired)
        results.append({
            "model": model,
            "family": indexed[(model, variety, condition, paired[0])][1].get(
                "family", ""),
            "dialect": variety,
            "condition": condition,
            "n_paired": len(paired),
            "hallucination_rate_msa": round(msa_rate, 6),
            "hallucination_rate_dialect": round(dialect_rate, 6),
            "hds": round(dialect_rate - msa_rate, 6),
            "hallucination_iou": round(
                len(dialect_hallucinated & msa_hallucinated) / len(union), 6)
                if union else None,
            "degeneration_rate_dialect": round(
                degeneration_count / len(paired), 6),
        })
    return results


def run(input_path="scores_combined.csv",
        json_path="results/analysis.json", csv_path="results/analysis.csv"):
    with open(input_path, encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    results = analyze(rows)
    if not results:
        raise SystemExit("ERROR: no paired dialect/MSA results were available")

    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {json_output} and {csv_output} ({len(results)} arms)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scores_combined.csv")
    parser.add_argument("--json", default="results/analysis.json")
    parser.add_argument("--csv", default="results/analysis.csv")
    args = parser.parse_args()
    run(args.input, args.json, args.csv)
