"""Leakage-safe TF-IDF baseline for the Layer 2 hallucination judge."""

import argparse
import json
import statistics
from pathlib import Path

from judge_data import grouped_split, load_judge_rows


def evaluate_fold(rows, fold=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.pipeline import FeatureUnion

    train, test = grouped_split(rows, fold=fold)
    texts = [f"{row['gold_answer']} [SEP] {row['answer']}" for row in rows]
    labels = [row["label"] for row in rows]

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, max_features=60_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, max_features=60_000)),
    ])
    train_matrix = features.fit_transform([texts[i] for i in train])
    test_matrix = features.transform([texts[i] for i in test])
    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced")
    classifier.fit(train_matrix, [labels[i] for i in train])
    predictions = classifier.predict(test_matrix)
    probabilities = classifier.predict_proba(test_matrix)[:, 1]
    truth = [labels[i] for i in test]

    accuracy = accuracy_score(truth, predictions)
    auc = roc_auc_score(truth, probabilities)
    print(f"fold={fold} train={len(train)} test={len(test)} "
          f"accuracy={accuracy:.3f} roc_auc={auc:.3f}")
    return {
        "fold": fold,
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": round(accuracy, 6),
        "roc_auc": round(auc, 6),
        "classification_report": classification_report(
            truth, predictions, output_dict=True),
    }


def run(path="judge_train.csv", fold=None,
        output="results/tfidf_grouped_cv.json"):
    rows = load_judge_rows(path)
    folds = range(5) if fold is None else [fold]
    results = [evaluate_fold(rows, current) for current in folds]
    summary = {
        "data": path,
        "n_rows": len(rows),
        "split": "StratifiedGroupKFold grouped by sample_index",
        "folds": results,
        "mean_accuracy": round(statistics.mean(
            result["accuracy"] for result in results), 6),
        "mean_roc_auc": round(statistics.mean(
            result["roc_auc"] for result in results), 6),
    }
    if len(results) > 1:
        summary["std_accuracy"] = round(statistics.stdev(
            result["accuracy"] for result in results), 6)
        summary["std_roc_auc"] = round(statistics.stdev(
            result["roc_auc"] for result in results), 6)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean_accuracy={summary['mean_accuracy']:.3f} "
          f"mean_roc_auc={summary['mean_roc_auc']:.3f}")
    print(f"wrote {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="judge_train.csv")
    parser.add_argument("--fold", type=int, default=None,
                        help="evaluate one fold; default evaluates all five")
    parser.add_argument("--output", default="results/tfidf_grouped_cv.json")
    args = parser.parse_args()
    run(args.data, args.fold, args.output)
