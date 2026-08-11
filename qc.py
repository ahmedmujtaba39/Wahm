"""
Translation QC: score each dialect candidate by how well its back-translation
recovers the original MSA question, then flag the worst ones for priority
human review.

This is an automatic meaning-preservation check. It does NOT replace the native
speaker — it tells the validator which rows to look at hardest.

Writes: validation_<dialect>.csv  (the sheet the validator actually fills in)

Usage:
    python qc.py gulf
"""

import argparse, csv, re
from difflib import SequenceMatcher

from wahm_text import normalize_arabic, token_f1


def char_sim(a, b):
    return SequenceMatcher(None, normalize_arabic(a), normalize_arabic(b)).ratio()


def dialect_distance(msa, dia):
    """How much the translation actually changed. Near 1.0 = barely translated."""
    return token_f1(msa, dia)


def run(dialect, flag_below=0.5, unchanged_above=0.95):
    with open(f"candidates_{dialect}.csv", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit(f"ERROR: candidates_{dialect}.csv contains no rows")

    scored = []
    for r in rows:
        bt = token_f1(r["question_msa"], r["backtranslation_msa"]) if r["backtranslation_msa"] else None
        dd = dialect_distance(r["question_msa"], r["dialect_candidate"])
        flags = []
        if bt is not None and bt < flag_below:
            flags.append("MEANING_DRIFT")
        if dd >= unchanged_above:
            flags.append("BARELY_TRANSLATED")
        if len(r["dialect_candidate"].split()) > 2 * len(r["question_msa"].split()):
            flags.append("TOO_LONG")
        if re.search(r"[A-Za-z]{3,}", r["dialect_candidate"]):
            flags.append("LATIN_TEXT")
        scored.append((r, bt, dd, flags))

    out = f"validation_{dialect}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "msa_question", "dialect_candidate",
                    "APPROVE_or_REWRITE", "your_rewrite", "fluency_1_5",
                    "auto_flags", "backtrans_sim", "dialect_distance"])
        for r, bt, dd, flags in scored:
            w.writerow([r["qid"], r["question_msa"], r["dialect_candidate"],
                        "", "", "", "|".join(flags),
                        f"{bt:.3f}" if bt is not None else "",
                        f"{dd:.3f}"])

    n = len(scored)
    bts = [b for _, b, _, _ in scored if b is not None]
    print(f"{dialect}: {n} candidates")
    if bts:
        print(f"  mean back-translation similarity : {sum(bts)/len(bts):.3f}")
    print(f"  mean dialect distance            : {sum(d for _, _, d, _ in scored)/n:.3f}")
    from collections import Counter
    c = Counter(fl for _, _, _, flags in scored for fl in flags)
    print(f"  flagged rows: {sum(1 for _,_,_,fl in scored if fl)}/{n}")
    for k, v in c.most_common():
        print(f"    {k}: {v}")
    print(f"\nwrote {out} — send this to the validator")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dialect")
    p.add_argument("--flag-below", type=float, default=0.5)
    a = p.parse_args()
    run(a.dialect, a.flag_below)
