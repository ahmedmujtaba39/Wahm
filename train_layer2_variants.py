"""Train both Layer 2 AraBERT variants from the supplied run specification."""

import argparse
import json
from pathlib import Path

from judge_data import load_judge_rows


def stratified_row_split(rows, random_state=42):
    """Return the specified row-random, stratified 80/20 split."""
    from sklearn.model_selection import train_test_split

    indices = list(range(len(rows)))
    labels = [row["label"] for row in rows]
    return train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=labels)


def run(data_path, output_dir, model_name, model_revision=None,
        model_source_id=None, epochs=3, batch_size=8, eval_batch_size=32):
    import datasets
    import numpy as np
    import sklearn
    import torch
    import transformers
    from datasets import Dataset
    from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                                 roc_auc_score)
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    rows = load_judge_rows(data_path)
    train_indices, test_indices = stratified_row_split(rows, random_state=42)
    train_labels = np.array([rows[index]["label"] for index in train_indices])
    test_labels = np.array([rows[index]["label"] for index in test_indices])
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=train_labels)

    model_path = Path(model_name)
    load_kwargs = {} if model_path.is_dir() else {"revision": model_revision}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    split_manifest = {
        "method": "train_test_split",
        "test_size": 0.2,
        "random_state": 42,
        "stratified": True,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "train_clean": int((train_labels == 0).sum()),
        "train_hallucinated": int((train_labels == 1).sum()),
        "test_clean": int((test_labels == 0).sum()),
        "test_hallucinated": int((test_labels == 1).sum()),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    (root / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            weights = torch.tensor(
                class_weights, dtype=outputs.logits.dtype,
                device=outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    def make_dataset(indices, use_gold):
        dataset = Dataset.from_dict({
            "text": [rows[index]["answer"] for index in indices],
            "gold": [rows[index]["gold_answer"] for index in indices],
            "label": [rows[index]["label"] for index in indices],
        })

        def tokenize(batch):
            if use_gold:
                return tokenizer(
                    batch["gold"], batch["text"], truncation=True,
                    padding="max_length", max_length=512)
            return tokenizer(
                batch["text"], truncation=True,
                padding="max_length", max_length=512)

        return dataset.map(tokenize, batched=True)

    def compute_metrics(prediction):
        predictions = np.argmax(prediction.predictions, axis=1)
        return {
            "accuracy": accuracy_score(prediction.label_ids, predictions),
            "f1": f1_score(prediction.label_ids, predictions),
        }

    results = {}
    for variant, use_gold in (("answer_only", False), ("gold_answer", True)):
        print(f"\n{'=' * 60}\nVARIANT: {variant}\n{'=' * 60}", flush=True)
        variant_dir = root / f"arabert_judge_{variant}"
        train_dataset = make_dataset(train_indices, use_gold)
        test_dataset = make_dataset(test_indices, use_gold)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, **load_kwargs)
        arguments = TrainingArguments(
            output_dir=str(variant_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            logging_first_step=True,
            disable_tqdm=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            seed=42,
            fp16=torch.cuda.is_available(),
        )
        trainer = WeightedTrainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        prediction = trainer.predict(test_dataset)
        probabilities = torch.softmax(
            torch.tensor(prediction.predictions), dim=1)[:, 1].numpy()
        predicted = np.argmax(prediction.predictions, axis=1)
        report = classification_report(
            test_labels, predicted, target_names=["clean", "hallucinated"],
            digits=3, output_dict=True)
        report_text = classification_report(
            test_labels, predicted, target_names=["clean", "hallucinated"],
            digits=3)
        auc = float(roc_auc_score(test_labels, probabilities))
        print(f"\n--- {variant} held-out results ---", flush=True)
        print(report_text, flush=True)
        print(f"ROC-AUC: {auc:.3f}", flush=True)
        trainer.save_model(str(variant_dir))
        tokenizer.save_pretrained(str(variant_dir))
        metadata = {
            "variant": variant,
            "use_gold": use_gold,
            "data": data_path,
            "base_model": model_source_id or model_name,
            "base_model_load_path": model_name,
            "base_model_revision": model_revision,
            "epochs": epochs,
            "train_batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "class_weight": "balanced",
            "class_weights": class_weights.tolist(),
            "accuracy": float(report["accuracy"]),
            "f1_hallucinated": float(report["hallucinated"]["f1-score"]),
            "roc_auc": auc,
            "classification_report": report,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        (variant_dir / "judge_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        results[variant] = metadata

    winner = max(results, key=lambda name: results[name]["roc_auc"])
    summary = {
        "answer_only_auc": results["answer_only"]["roc_auc"],
        "gold_answer_auc": results["gold_answer"]["roc_auc"],
        "tfidf_baseline_auc_from_specification": 0.82,
        "winner": winner,
        "winner_path": str(root / f"arabert_judge_{winner}"),
    }
    (root / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}", flush=True)
    print(f"answer-only AUC : {summary['answer_only_auc']:.3f}", flush=True)
    print(f"gold+answer AUC : {summary['gold_answer_auc']:.3f}", flush=True)
    print("TF-IDF baseline : 0.820", flush=True)
    print(f"winner          : {winner}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="judge_train.csv")
    parser.add_argument("--output", default="layer2_variants")
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-source-id", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.data, args.output, args.model, args.model_revision,
        args.model_source_id, args.epochs, args.batch_size,
        args.eval_batch_size)
