"""
WAHM stage 3: ingest returned validation sheets, compute the translation-quality
metrics, apply the meaning-preservation hard filter, and build the final
benchmark file.

Reads   candidates_<dialect>.csv    (machine output + back-translations)
        validation_<dialect>.csv    (returned from your native-speaker validator)
Writes  final_<dialect>.csv         (the benchmark arm: human-approved questions)
        metrics_<dialect>.json      (the numbers for the paper's table)

Metrics reported per dialect
    acceptance_rate        fraction approved unedited          (headline quality)
    mean_edit_distance     how badly wrong the rejects were    (severity)
    backtrans_similarity   meaning preserved?                  (validity)
    dialect_distance       did it actually leave MSA?          (authenticity)
    mean_fluency           validator 1-5 naturalness rating    (AAVENUE-style)
    filtered_out           rows failing the hard filter

Hard filter
    A row whose back-translation similarity is below --min-backtrans AND which
    the validator did not rewrite is EXCLUDED from the benchmark. Meaning
    preservation is a correctness precondition, not a metric you report and
    move past: a question that changed meaning produces a model answer that
    looks hallucinated when it is not.

Usage
    python finalize.py gulf
    python finalize.py gulf --min-backtrans 0.45
"""

import argparse, csv, json, statistics, sys

from wahm_text import normalize_arabic, token_f1


def levenshtein(a, b):
    """Character-level edit distance, normalized by the longer string."""
    a, b = normalize_arabic(a), normalize_arabic(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def load(dialect):
    with open(f"candidates_{dialect}.csv", encoding="utf-8", newline="") as source:
        cand = {r["qid"]: r for r in csv.DictReader(source)}
    try:
        with open(f"validation_{dialect}.csv", encoding="utf-8", newline="") as source:
            val = list(csv.DictReader(source))
    except FileNotFoundError:
        sys.exit(f"ERROR: validation_{dialect}.csv not found. "
                 "Has the validator returned their sheet yet?")
    return cand, val


def decision(row):
    """Read the validator's APPROVE/REWRITE column tolerantly."""
    d = row.get("APPROVE_or_REWRITE", "").strip().upper()
    rewrite = row.get("your_rewrite", "").strip()
    if d.startswith("A"):
        return "approved", ""
    if d.startswith("R") or rewrite:
        return "rewritten", rewrite
    if d.startswith("D"):
        return "dropped", ""
    return "unreviewed", rewrite


def run(dialect, min_backtrans=0.4):
    cand, val = load(dialect)

    rows, unreviewed = [], 0
    for v in val:
        qid = v["qid"]
        c = cand.get(qid)
        if not c:
            continue
        state, rewrite = decision(v)
        if state == "unreviewed":
            unreviewed += 1
            continue

        machine = c["dialect_candidate"]
        final = rewrite if (state == "rewritten" and rewrite) else machine

        bt = token_f1(c["question_msa"], c["backtranslation_msa"]) \
            if c.get("backtranslation_msa") else None
        dd = token_f1(c["question_msa"], final)
        ed = levenshtein(machine, final) if state == "rewritten" else 0.0

        try:
            flu = float(v.get("fluency_1_5", "") or 0) or None
        except ValueError:
            flu = None

        # hard filter: meaning drifted AND the human did not fix it
        excluded = (bt is not None and bt < min_backtrans
                    and state != "rewritten") or state == "dropped"

        rows.append(dict(qid=qid, msa=c["question_msa"], gold=c["gold_answer"],
                         machine=machine, final=final, state=state,
                         backtrans=bt, dialect_distance=dd, edit=ed,
                         fluency=flu, excluded=excluded,
                         sub_variety=c.get("sub_variety", "")))

    if not rows:
        sys.exit("ERROR: no reviewed rows found. Check the "
                 "APPROVE_or_REWRITE column is filled in.")

    kept = [r for r in rows if not r["excluded"]]
    rewritten = [r for r in rows if r["state"] == "rewritten"]
    bts = [r["backtrans"] for r in rows if r["backtrans"] is not None]
    flus = [r["fluency"] for r in rows if r["fluency"]]

    metrics = {
        "dialect": dialect,
        "sub_variety": rows[0]["sub_variety"],
        "n_reviewed": len(rows),
        "n_unreviewed": unreviewed,
        "acceptance_rate": round(sum(r["state"] == "approved" for r in rows) / len(rows), 3),
        "rewrite_rate": round(len(rewritten) / len(rows), 3),
        "mean_edit_distance_on_rewrites": round(
            statistics.mean(r["edit"] for r in rewritten), 3) if rewritten else None,
        "mean_backtrans_similarity": round(statistics.mean(bts), 3) if bts else None,
        "mean_dialect_distance": round(
            statistics.mean(r["dialect_distance"] for r in kept), 3) if kept else None,
        "mean_fluency_1_5": round(statistics.mean(flus), 2) if flus else None,
        "n_filtered_out": len(rows) - len(kept),
        "n_final_benchmark": len(kept),
        "min_backtrans_threshold": min_backtrans,
    }

    out = f"final_{dialect}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question_msa", "question_dialect", "gold_answer",
                    "dialect", "sub_variety", "provenance"])
        for r in kept:
            w.writerow([r["qid"], r["msa"], r["final"], r["gold"],
                        dialect, r["sub_variety"],
                        "machine" if r["state"] == "approved" else "human_rewrite"])

    with open(f"metrics_{dialect}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"=== {dialect} ({metrics['sub_variety'] or 'sub-variety unspecified'}) ===")
    for k, v in metrics.items():
        if k not in ("dialect", "sub_variety"):
            print(f"  {k:34s} {v}")
    if unreviewed:
        print(f"\n  NOTE: {unreviewed} rows still unreviewed — chase the validator")
    print(f"\nwrote {out} ({len(kept)} benchmark items)")
    print(f"wrote metrics_{dialect}.json")

    # authenticity warning: high dialect_distance means it barely left MSA
    dd = metrics["mean_dialect_distance"]
    if dd and dd > 0.85:
        print(f"\n  WARNING: mean dialect distance {dd} is very high — this arm "
              "may be close to MSA, which would flatten HDS for the wrong reason.")

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dialect")
    p.add_argument("--min-backtrans", type=float, default=0.4,
                   help="hard filter threshold; set it from your pilot data")
    a = p.parse_args()
    run(a.dialect, a.min_backtrans)
