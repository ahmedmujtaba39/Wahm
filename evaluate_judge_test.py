"""Evaluate the frozen ablation winner once on the held-out test partition."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from judge_data import grouped_train_validation_test_split, load_judge_rows
from train_judge import (_metrics_at_threshold, _per_tag_recall,
                         _probabilities, _tokenize_batch)


def run(data_path, model_dir, batch_size=32):
    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    metadata_path = model_dir / "judge_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("test_metrics") is not None:
        raise SystemExit("ERROR: test metrics already exist; refusing a second evaluation")
    raw_data = Path(data_path).read_bytes()
    if hashlib.sha256(raw_data).hexdigest() != metadata["data_sha256"]:
        raise SystemExit("ERROR: judge dataset hash does not match training metadata")
    rows = load_judge_rows(data_path)
    _, _, test_indices = grouped_train_validation_test_split(
        rows, test_fold=metadata["test_fold"],
        validation_fold=metadata["validation_fold"])
    expected_ids = sorted({rows[index]["sample_index"] for index in test_indices},
                          key=int)
    if expected_ids != metadata["test_question_ids"]:
        raise SystemExit("ERROR: reconstructed test split does not match metadata")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(test_indices), batch_size):
            indices = test_indices[start:start + batch_size]
            batch = {
                "question": [rows[i].get("question", "") for i in indices],
                "gold": [rows[i]["gold_answer"] for i in indices],
                "answer": [rows[i]["answer"] for i in indices],
            }
            encoded = _tokenize_batch(tokenizer, batch, metadata["input_variant"])
            encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities.extend(_probabilities(model(**encoded).logits.cpu().numpy(),
                                                np).tolist())

    labels = np.asarray([rows[index]["label"] for index in test_indices])
    probabilities = np.asarray(probabilities)
    threshold = float(metadata["decision_threshold"])
    functions = {
        "accuracy": accuracy_score, "f1": f1_score,
        "precision": precision_score, "recall": recall_score,
        "roc_auc": roc_auc_score, "pr_auc": average_precision_score,
        "confusion": confusion_matrix,
    }
    metadata["test_metrics"] = _metrics_at_threshold(
        labels, probabilities, threshold, functions)
    metadata["test_per_tag_recall"] = _per_tag_recall(
        rows, test_indices, probabilities, threshold)
    metadata["test_evaluated_after_variant_selection"] = True

    fields = ["row_index", "sample_index", "question", "gold_answer", "answer",
              "gold_label", "hallucination_probability", "predicted_label"]
    with (model_dir / "test_predictions.csv").open(
            "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, label, probability in zip(test_indices, labels, probabilities):
            writer.writerow({
                "row_index": index, "sample_index": rows[index]["sample_index"],
                "question": rows[index].get("question", ""),
                "gold_answer": rows[index]["gold_answer"],
                "answer": rows[index]["answer"], "gold_label": int(label),
                "hallucination_probability": f"{probability:.8f}",
                "predicted_label": int(probability >= threshold),
            })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(metadata["test_metrics"], indent=2))
    return metadata["test_metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="judge_train.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.data, args.model, args.batch_size)
