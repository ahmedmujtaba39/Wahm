"""
WAHM Stage 3: LLM-judged translation quality (AAVENUE-style).

Uses GPT-5.6 (or any chat model) to rate each translation on four dimensions:
  Quality, Fluency, Coherence, Understandability (each 0-100)

This mirrors AAVENUE's evaluation methodology exactly, enabling direct
methodological comparison.

Reads   candidates_<dialect>.csv
Writes  judge_translation_<dialect>.csv   (per-row LLM scores)
        judge_translation_<dialect>_summary.json  (aggregates)

Usage:
    export OPENAI_API_KEY=...
    python judge_translation.py gulf
    python judge_translation.py gulf --model gpt-4o   # compare judges
"""

import argparse, csv, json, os, re, sys, time
from openai import OpenAI
import numpy as np

MODEL = os.getenv("WAHM_JUDGE_MODEL", "gpt-5.6")

JUDGE_PROMPT = """You are evaluating the quality of a dialect Arabic translation.

Original MSA question: {msa}
Dialect translation ({dialect}): {dia}

Rate the translation on four dimensions. For each, give a score from 0 to 100.

Quality: How good is the overall translation? (accuracy, style, appropriateness)
Fluency: How grammatically correct and well-written is it?
Coherence: Does it make logical sense and maintain consistency?
Understandability: How easily can a native {dialect} speaker comprehend this?

Output EXACTLY four lines in this format, nothing else:
Quality: [number]
Fluency: [number]
Coherence: [number]
Understandability: [number]"""


def parse_scores(text):
    """Parse the four scores from the LLM judge response."""
    scores = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        for dim in ["Quality", "Fluency", "Coherence", "Understandability"]:
            if line.lower().startswith(dim.lower()):
                match = re.search(r"(\d+(?:\.\d+)?)", line)
                if match:
                    scores[dim.lower()] = float(match.group(1))
    return scores if len(scores) == 4 else None


DIALECT_NAMES = {
    "gulf": "Gulf (Khaleeji) Arabic",
    "egyptian": "Egyptian (Masri) Arabic",
    "levantine": "Levantine (Shami) Arabic",
    "sudanese": "Sudanese Arabic",
}


def run(dialect, model=None):
    model = model or MODEL
    client = OpenAI()

    path = f"candidates_{dialect}.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        sys.exit(f"ERROR: {path} is empty")

    name = DIALECT_NAMES[dialect]
    print(f"=== LLM-judging {dialect}: {len(rows)} translations ===")
    print(f"  judge model: {model}")

    all_scores = []
    failed = 0

    out = f"judge_translation_{dialect}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question_msa", "dialect_candidate",
                    "quality", "fluency", "coherence", "understandability",
                    "judge_model", "raw_response"])

        for i, r in enumerate(rows, 1):
            prompt = JUDGE_PROMPT.format(
                msa=r["question_msa"], dia=r["dialect_candidate"], dialect=name)

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=100)
                    raw = resp.choices[0].message.content.strip()
                    scores = parse_scores(raw)
                    break
                except Exception as e:
                    if attempt == 2:
                        raw = f"ERROR: {e}"
                        scores = None
                    time.sleep(2 ** attempt)

            if scores:
                w.writerow([r["qid"], r["question_msa"], r["dialect_candidate"],
                            scores["quality"], scores["fluency"],
                            scores["coherence"], scores["understandability"],
                            model, raw])
                all_scores.append(scores)
            else:
                w.writerow([r["qid"], r["question_msa"], r["dialect_candidate"],
                            "", "", "", "", model, raw])
                failed += 1

            f.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  (failed: {failed})")

    # Aggregate
    def mean_dim(dim):
        vals = [s[dim] for s in all_scores if dim in s]
        return round(np.mean(vals), 2) if vals else None

    summary = {
        "dialect": dialect,
        "judge_model": model,
        "n_judged": len(all_scores),
        "n_failed": failed,
        "quality_mean": mean_dim("quality"),
        "fluency_mean": mean_dim("fluency"),
        "coherence_mean": mean_dim("coherence"),
        "understandability_mean": mean_dim("understandability"),
    }

    summary_path = f"judge_translation_{dialect}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"AAVENUE-STYLE SCORES ({dialect}, judged by {model}):")
    for k in ["quality_mean", "fluency_mean", "coherence_mean", "understandability_mean"]:
        print(f"  {k:35s} {summary[k]}")
    print(f"  failed: {failed}")
    print(f"\nwrote {out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dialect", choices=list(DIALECT_NAMES))
    p.add_argument("--model", default=None, help="override judge model")
    a = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY")

    run(a.dialect, a.model)
