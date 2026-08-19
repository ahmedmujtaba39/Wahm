"""Export a compact, reproducible per-epoch Judge training report."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = (
    "epoch", "train_loss", "learning_rate", "grad_norm", "validation_loss",
    "validation_accuracy", "validation_precision", "validation_recall",
    "validation_f1", "validation_roc_auc",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(model_dir, output_dir):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    state_path = model_dir / "trainer_state.json"
    metadata_path = model_dir / "judge_metadata.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    by_epoch = {}
    for event in state.get("log_history", []):
        if "epoch" not in event:
            continue
        raw_epoch = float(event["epoch"])
        nearest_epoch = round(raw_epoch)
        epoch = (nearest_epoch if abs(raw_epoch - nearest_epoch) < 0.02
                 else round(raw_epoch, 6))
        row = by_epoch.setdefault(epoch, {field: "" for field in FIELDS})
        row["epoch"] = epoch
        mappings = {
            "loss": "train_loss",
            "learning_rate": "learning_rate",
            "grad_norm": "grad_norm",
            "eval_loss": "validation_loss",
            "eval_accuracy": "validation_accuracy",
            "eval_precision": "validation_precision",
            "eval_recall": "validation_recall",
            "eval_f1": "validation_f1",
            "eval_roc_auc": "validation_roc_auc",
        }
        for source, target in mappings.items():
            if source in event:
                row[target] = event[source]

    epoch_rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if not epoch_rows:
        raise ValueError("trainer_state.json has no epoch-level events")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "epoch_metrics.csv").open(
            "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(epoch_rows)

    configuration = {
        "base_model": metadata["base_model"],
        "base_model_revision": metadata.get("base_model_revision"),
        "data": metadata["data"],
        "data_sha256": metadata["data_sha256"],
        "input": metadata["input"],
        "seed": metadata["seed"],
        "split_method": metadata["split_method"],
        "test_fold": metadata["test_fold"],
        "validation_fold": metadata["validation_fold"],
        "n_train": metadata["n_train"],
        "n_validation": metadata["n_validation"],
        "n_test": metadata["n_test"],
        "decision_threshold": metadata["decision_threshold"],
        "threshold_selection": metadata["threshold_selection"],
        "training_arguments": metadata["training_arguments"],
        "max_steps": state.get("max_steps"),
        "num_train_epochs": state.get("num_train_epochs"),
        "versions": metadata["versions"],
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    weights_path = model_dir / "model.safetensors"
    summary = {
        "decision_threshold": metadata["decision_threshold"],
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metadata["test_metrics"],
        "epochs_logged": len(epoch_rows),
        "judge_metadata_sha256": _sha256(metadata_path),
        "trainer_state_sha256": _sha256(state_path),
        "model_weights_sha256": _sha256(weights_path),
        "model_weights_bytes": weights_path.stat().st_size,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return epoch_rows, configuration, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    rows, _, report = summarize(arguments.model, arguments.output)
    print(json.dumps({"epochs": len(rows), **report}, indent=2))
