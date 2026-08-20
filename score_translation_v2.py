"""
WAHM Stage 2: automated translation quality scores.

Computes four automated metrics per translation:
  1. BERTScore (semantic similarity, embedding-level)
  2. BARTScore (generation log-likelihood fidelity)
  3. Back-translation cosine similarity (sentence embeddings, MSA round trip)
  4. Dialect lexical overlap (token-F1 vs MSA original, lower = more dialectal)

Reads   candidates_<dialect>.csv
Writes  scores_translation_<dialect>_v2.csv  (per-row scores)
        scores_translation_<dialect>_v2_summary.json  (aggregate stats)

Usage:
    pip install bert-score  # needs torch
    python score_translation_v2.py gulf
"""

import argparse, csv, json, re, sys, unicodedata
import numpy as np
from translation_metrics_v2 import backtrans_cosine_similarities

# ---- Arabic normalization (same as Layer 1) ----
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")

def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = _DIAC.sub("", text)
    for a, b in [("أ","ا"),("إ","ا"),("آ","ا"),("ى","ي"),("ة","ه")]:
        text = text.replace(a, b)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def token_f1(a, b):
    ta, tb = set(normalize_arabic(a).split()), set(normalize_arabic(b).split())
    if not ta or not tb: return 0.0
    inter = len(ta & tb)
    if inter == 0: return 0.0
    p, r = inter/len(tb), inter/len(ta)
    return 2*p*r/(p+r)


def compute_bertscore(sources, translations, lang="ar"):
    """BERTScore using a multilingual model."""
    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(translations, sources, lang=lang, verbose=True,
                             model_type="bert-base-multilingual-cased")
        return F.tolist()
    except ImportError:
        print("WARNING: bert-score not installed, skipping BERTScore")
        return [None] * len(sources)


def compute_bartscore(sources, translations):
    """BARTScore: log-likelihood of translation given source."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        model_name = "facebook/mbart-large-50"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

        scores = []
        for src, tgt in zip(sources, translations):
            inputs = tok(src, return_tensors="pt", truncation=True, max_length=512)
            labels = tok(tgt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inputs, labels=labels["input_ids"])
            scores.append(-out.loss.item())  # negative log-likelihood
        return scores
    except Exception as e:
        print(f"WARNING: BARTScore failed ({e}), skipping")
        return [None] * len(sources)


def run(dialect):
    path = f"candidates_{dialect}.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        sys.exit(f"ERROR: {path} is empty")

    print(f"=== scoring {dialect}: {len(rows)} translations ===")

    sources = [r["question_msa"] for r in rows]
    translations = [r["dialect_candidate"] for r in rows]
    backtranslations = [r.get("backtranslation_msa", "") for r in rows]

    # 1. Semantic similarity of original MSA and its MSA back-translation.
    bt_sims = backtrans_cosine_similarities(sources, backtranslations)

    # 2. Dialect lexical overlap (kept separate from semantic back-translation QC)
    dd_sims = [token_f1(src, tgt) for src, tgt in zip(sources, translations)]

    # 3. BERTScore
    print("\ncomputing BERTScore...")
    bert_scores = compute_bertscore(sources, translations)

    # 4. BARTScore
    print("\ncomputing BARTScore...")
    bart_scores = compute_bartscore(sources, translations)

    # Write per-row scores
    out = f"scores_translation_{dialect}_v2.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question_msa", "dialect_candidate",
                    "backtrans_cosine_similarity", "dialect_lexical_overlap",
                    "bertscore_f1", "bartscore"])
        for i, r in enumerate(rows):
            w.writerow([r["qid"], r["question_msa"], r["dialect_candidate"],
                        f"{bt_sims[i]:.4f}" if bt_sims[i] is not None else "",
                        f"{dd_sims[i]:.4f}",
                        f"{bert_scores[i]:.4f}" if bert_scores[i] is not None else "",
                        f"{bart_scores[i]:.4f}" if bart_scores[i] is not None else ""])

    # Aggregate summary
    def safe_mean(lst):
        clean = [x for x in lst if x is not None]
        return round(np.mean(clean), 4) if clean else None

    summary = {
        "dialect": dialect,
        "n_translations": len(rows),
        "backtrans_cosine_similarity_mean": safe_mean(bt_sims),
        "dialect_lexical_overlap_mean": safe_mean(dd_sims),
        "bertscore_f1_mean": safe_mean(bert_scores),
        "bartscore_mean": safe_mean(bart_scores),
        "barely_translated_count": sum(1 for d in dd_sims if d >= 0.95),
        "meaning_drift_count": sum(1 for b in bt_sims if b is not None and b < 0.4),
    }

    summary_path = f"scores_translation_{dialect}_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    for k, v in summary.items():
        print(f"  {k:35s} {v}")
    print(f"\nwrote {out}")
    print(f"wrote {summary_path}")
    print(f"next: python judge_translation.py {dialect}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dialect", choices=["gulf","egyptian","levantine","sudanese"])
    a = p.parse_args()
    run(a.dialect)
