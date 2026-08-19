"""Apply the factual judge to valid rows routed by WAHM Judge v2 Layer 1."""

import argparse
import csv
import json
from pathlib import Path


def final_decision(layer1_decision, probability, threshold):
    """Return three-class decision, factual label, and headline failure label."""
    if layer1_decision == "degeneration":
        return "degeneration", "", 1
    if layer1_decision != "defer" or probability is None:
        raise ValueError(f"unexpected Layer 1 route: {layer1_decision!r}")
    hallucinated = probability >= threshold
    return ("factual_hallucination" if hallucinated else "clean",
            int(hallucinated), int(hallucinated))


def run(input_path="scores_layer1.csv", model_path="arabert_judge_gold_answer",
        output_path="scores_combined.csv", threshold=None, batch_size=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(input_path, encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit(f"ERROR: {input_path} contains no scores")

    deferred = [index for index, row in enumerate(rows)
                if row["layer1_decision"] == "defer"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if threshold is None:
        metadata_path = Path(model_path) / "judge_metadata.json"
        if not metadata_path.exists():
            raise SystemExit("ERROR: no --threshold supplied and judge_metadata.json "
                             "is missing from the model directory")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        threshold = float(metadata["decision_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    probabilities = {}
    with torch.inference_mode():
        for start in range(0, len(deferred), batch_size):
            indices = deferred[start:start + batch_size]
            encoded = tokenizer(
                [rows[i]["gold_answer"] for i in indices],
                [rows[i]["answer"] for i in indices],
                padding=True, truncation=True, max_length=512,
                return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            probabilities.update(zip(indices, probs))

    fields = list(rows[0]) + ["layer2_hallucination_probability",
                              "layer2_decision_threshold",
                              "combined_decision", "combined_label",
                              "headline_failure_label"]
    with open(output_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            probability = probabilities.get(index)
            decision, label, headline_label = final_decision(
                row["layer1_decision"], probability, threshold)
            probability_text = "" if probability is None else f"{probability:.6f}"
            row.update(layer2_hallucination_probability=probability_text,
                       layer2_decision_threshold=(
                           "" if probability is None else f"{threshold:.6f}"),
                       combined_decision=decision,
                       combined_label=label,
                       headline_failure_label=headline_label)
            writer.writerow(row)
    print(f"wrote {output_path}: Layer 2 scored {len(deferred)}/{len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scores_layer1.csv")
    parser.add_argument("--model", default="arabert_judge_gold_answer")
    parser.add_argument("--output", default="scores_combined.csv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the validation-calibrated model threshold")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.input, args.model, args.output, args.threshold, args.batch_size)
