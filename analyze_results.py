"""Compute paired Judge v2 hallucination drift and uncertainty statistics."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def _decision(row):
    return row.get("combined_decision") or row.get("layer1_decision")


def _factual_label(row):
    decision = _decision(row)
    if decision == "clean":
        return 0
    if decision in {"factual_hallucination", "hallucinated"}:
        return 1
    return None


def _is_degeneration(row):
    return _decision(row) == "degeneration" or bool(
        row.get("degeneration_reasons", ""))


def _percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - index)
            + ordered[upper] * (index - lower))


def _bootstrap_intervals(pairs, iterations, seed):
    if not pairs or iterations <= 0:
        return (None, None), (None, None), (None, None)
    rng = random.Random(seed)
    msa_rates, dialect_rates, differences = [], [], []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        msa_rate = sum(left for left, _ in sample) / len(sample)
        dialect_rate = sum(right for _, right in sample) / len(sample)
        msa_rates.append(msa_rate)
        dialect_rates.append(dialect_rate)
        differences.append(dialect_rate - msa_rate)
    bounds = lambda values: (_percentile(values, 0.025),
                             _percentile(values, 0.975))
    return bounds(msa_rates), bounds(dialect_rates), bounds(differences)


def _mcnemar_exact_p(clean_to_hallucination, hallucination_to_clean):
    discordant = clean_to_hallucination + hallucination_to_clean
    if discordant == 0:
        return 1.0
    smaller = min(clean_to_hallucination, hallucination_to_clean)
    tail = sum(math.comb(discordant, value)
               for value in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def analyze(rows, bootstrap_iterations=2000, seed=42):
    indexed = {}
    for row in rows:
        model = row.get("analysis_run") or row["model"]
        key = (model, row["variety"], row["condition"], row["qid"])
        indexed[key] = row

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
        paired_qids = sorted(dialect_qids & msa_qids)
        if not paired_qids:
            continue

        factual_pairs = []
        degeneration_msa = degeneration_dialect = 0
        headline_failure_msa = headline_failure_dialect = 0
        for qid in paired_qids:
            msa_row = indexed[(model, "msa", "direct", qid)]
            dialect_row = indexed[(model, variety, condition, qid)]
            degeneration_msa += _is_degeneration(msa_row)
            degeneration_dialect += _is_degeneration(dialect_row)
            msa_label = _factual_label(msa_row)
            dialect_label = _factual_label(dialect_row)
            headline_failure_msa += int(
                _is_degeneration(msa_row) or msa_label == 1)
            headline_failure_dialect += int(
                _is_degeneration(dialect_row) or dialect_label == 1)
            if msa_label is not None and dialect_label is not None:
                factual_pairs.append((msa_label, dialect_label))

        n_valid = len(factual_pairs)
        clean_clean = sum(left == 0 and right == 0
                          for left, right in factual_pairs)
        clean_to_hallucination = sum(left == 0 and right == 1
                                     for left, right in factual_pairs)
        hallucination_to_clean = sum(left == 1 and right == 0
                                     for left, right in factual_pairs)
        hallucination_hallucination = sum(left == 1 and right == 1
                                          for left, right in factual_pairs)
        msa_rate = (sum(left for left, _ in factual_pairs) / n_valid
                    if n_valid else None)
        dialect_rate = (sum(right for _, right in factual_pairs) / n_valid
                        if n_valid else None)
        hds = dialect_rate - msa_rate if n_valid else None
        union = clean_to_hallucination + hallucination_to_clean \
            + hallucination_hallucination
        arm_seed = int(hashlib.sha256(
            f"{seed}|{model}|{variety}|{condition}".encode()).hexdigest()[:16], 16)
        msa_ci, dialect_ci, hds_ci = _bootstrap_intervals(
            factual_pairs, bootstrap_iterations, arm_seed)

        def rounded(value):
            return round(value, 6) if value is not None else None

        results.append({
            "model": model,
            "family": indexed[(model, variety, condition, paired_qids[0])].get(
                "family", ""),
            "dialect": variety,
            "condition": condition,
            "n_paired_total": len(paired_qids),
            "n_paired_factual": n_valid,
            "n_degenerated_msa": degeneration_msa,
            "n_degenerated_dialect": degeneration_dialect,
            "degeneration_rate_msa": round(
                degeneration_msa / len(paired_qids), 6),
            "degeneration_rate_dialect": round(
                degeneration_dialect / len(paired_qids), 6),
            "headline_failure_rate_msa": round(
                headline_failure_msa / len(paired_qids), 6),
            "headline_failure_rate_dialect": round(
                headline_failure_dialect / len(paired_qids), 6),
            "hallucination_rate_msa": rounded(msa_rate),
            "hallucination_rate_msa_ci_low": rounded(msa_ci[0]),
            "hallucination_rate_msa_ci_high": rounded(msa_ci[1]),
            "hallucination_rate_dialect": rounded(dialect_rate),
            "hallucination_rate_dialect_ci_low": rounded(dialect_ci[0]),
            "hallucination_rate_dialect_ci_high": rounded(dialect_ci[1]),
            "hds": rounded(hds),
            "hds_ci_low": rounded(hds_ci[0]),
            "hds_ci_high": rounded(hds_ci[1]),
            "clean_clean": clean_clean,
            "clean_to_hallucination": clean_to_hallucination,
            "hallucination_to_clean": hallucination_to_clean,
            "hallucination_hallucination": hallucination_hallucination,
            "mcnemar_exact_p": round(_mcnemar_exact_p(
                clean_to_hallucination, hallucination_to_clean), 6),
            "hallucination_iou": round(
                hallucination_hallucination / union, 6) if union else None,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": seed,
        })
    return results


def run(input_paths="scores_combined.csv",
        json_path="results/analysis.json", csv_path="results/analysis.csv",
        bootstrap_iterations=2000, seed=42):
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]
    input_paths = list(input_paths)
    rows = []
    for input_path in input_paths:
        with open(input_path, encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                record = dict(row)
                if len(input_paths) > 1:
                    record["analysis_run"] = Path(input_path).parent.name
                rows.append(record)
    results = analyze(rows, bootstrap_iterations, seed)
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
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", dest="input_paths", action="append")
    inputs.add_argument("--inputs", dest="input_paths", nargs="+")
    parser.add_argument("--json", default="results/analysis.json")
    parser.add_argument("--csv", default="results/analysis.csv")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input_paths or ["scores_combined.csv"], args.json, args.csv,
        args.bootstrap_iterations, args.seed)
