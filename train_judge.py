"""Fine-tune and evaluate AraBERT with question-disjoint data partitions."""

import argparse
import json
from pathlib import Path

from judge_data import grouped_train_validation_test_split, load_judge_rows


def _model_load_kwargs(model_name, model_revision):
    return {} if Path(model_name).is_dir() else {"revision": model_revision}


def _probabilities(logits, np):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, 1] / exponentials.sum(axis=1)


def _metrics_at_threshold(labels, probabilities, threshold, metrics):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(metrics["accuracy"](labels, predictions)), 6),
        "f1": round(float(metrics["f1"](labels, predictions)), 6),
        "roc_auc": round(float(metrics["roc_auc"](labels, probabilities)), 6),
    }


def _select_threshold(labels, probabilities, np, f1_score):
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [(float(f1_score(labels, probabilities >= value)), float(value))
              for value in candidates]
    best_f1 = max(score for score, _ in scored)
    tied = [value for score, value in scored if score == best_f1]
    return min(tied, key=lambda value: abs(value - 0.5))


def run(data_path, output_dir, test_fold=0, validation_fold=0, epochs=3,
        use_gold=True, batch_size=2, gradient_accumulation=8,
        model_name="aubmindlab/bert-base-arabertv2", model_revision=None,
        smoke_groups=None, model_source_id=None):
    import datasets
    import numpy as np
    import torch
    import transformers
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    rows = load_judge_rows(data_path)
    train_indices, validation_indices, test_indices = \
        grouped_train_validation_test_split(
            rows, test_fold=test_fold, validation_fold=validation_fold)

    if smoke_groups:
        def limit_groups(indices):
            chosen, groups = [], set()
            for index in indices:
                group = rows[index]["sample_index"]
                if group not in groups and len(groups) >= smoke_groups:
                    continue
                groups.add(group)
                chosen.append(index)
            return chosen
        train_indices = limit_groups(train_indices)
        validation_indices = limit_groups(validation_indices)
        test_indices = limit_groups(test_indices)
    load_kwargs = _model_load_kwargs(model_name, model_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)

    def make_dataset(indices):
        data = {
            "answer": [rows[i]["answer"] for i in indices],
            "gold": [rows[i]["gold_answer"] for i in indices],
            "label": [rows[i]["label"] for i in indices],
        }
        dataset = Dataset.from_dict(data)

        def tokenize(batch):
            if use_gold:
                return tokenizer(batch["gold"], batch["answer"],
                                 truncation=True, max_length=512)
            return tokenizer(batch["answer"], truncation=True, max_length=512)
        return dataset.map(tokenize, batched=True)

    train_dataset = make_dataset(train_indices)
    validation_dataset = make_dataset(validation_indices)
    test_dataset = make_dataset(test_indices)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, **load_kwargs)

    def trainer_metrics(output):
        probabilities = _probabilities(output.predictions, np)
        return _metrics_at_threshold(
            output.label_ids, probabilities, 0.5,
            {"accuracy": accuracy_score, "f1": f1_score,
             "roc_auc": roc_auc_score})

    output_path = Path(output_dir)
    arguments = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 4),
        gradient_accumulation_steps=gradient_accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=arguments,
        train_dataset=train_dataset, eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics)
    trainer.train()

    validation_output = trainer.predict(validation_dataset)
    validation_probabilities = _probabilities(validation_output.predictions, np)
    threshold = _select_threshold(
        validation_output.label_ids, validation_probabilities, np, f1_score)
    test_output = trainer.predict(test_dataset)
    test_probabilities = _probabilities(test_output.predictions, np)
    metric_functions = {"accuracy": accuracy_score, "f1": f1_score,
                        "roc_auc": roc_auc_score}

    def group_ids(indices):
        return sorted({rows[index]["sample_index"] for index in indices},
                      key=lambda value: int(value))

    metadata = {
        "base_model": model_source_id or model_name,
        "base_model_load_path": model_name,
        "base_model_revision": model_revision,
        "smoke_test": smoke_groups is not None,
        "input": "gold_answer + answer" if use_gold else "answer only",
        "data": data_path,
        "seed": 42,
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "decision_threshold": round(threshold, 6),
        "validation_metrics": _metrics_at_threshold(
            validation_output.label_ids, validation_probabilities, threshold,
            metric_functions),
        "test_metrics": _metrics_at_threshold(
            test_output.label_ids, test_probabilities, threshold,
            metric_functions),
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "n_test": len(test_indices),
        "train_question_ids": group_ids(train_indices),
        "validation_question_ids": group_ids(validation_indices),
        "test_question_ids": group_ids(test_indices),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    (output_path / "judge_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"decision_threshold": metadata["decision_threshold"],
                      "validation_metrics": metadata["validation_metrics"],
                      "test_metrics": metadata["test_metrics"]}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="judge_train.csv")
    parser.add_argument("--output", default="arabert_judge_gold_answer")
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--model-revision", default=None,
                        help="immutable Hugging Face commit for the base model")
    parser.add_argument("--model-source-id", default=None,
                        help="canonical model ID when --model is a local snapshot")
    parser.add_argument("--smoke-groups", type=int, default=None,
                        help="limit each partition to N question groups for plumbing tests")
    parser.add_argument("--answer-only", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.test_fold, args.validation_fold,
        args.epochs, not args.answer_only, args.batch_size,
        args.gradient_accumulation, args.model, args.model_revision,
        args.smoke_groups, args.model_source_id)
