"""Loading and leakage-safe splitting for AraHalluEval judge training data."""

import csv

LABEL_COLUMNS = [
    "Named-Entity Hallucination",
    "Temporal/Number Hallucination",
    "Factual Contradiction",
    "Conflict Hallucination",
    "K0wledge Source Conflict",  # Typo is present in the source dataset.
    "Grammar Hallucination",
    "Generic/Imprecise Hallucination",
    "Instruction Inconsistency",
    "Code-Switching",
]


def valid_labels(row):
    return all(row.get(column, "").strip() in {"", "0", "1"}
               for column in LABEL_COLUMNS)


def binary_label(row):
    return int(any(row.get(column, "").strip() == "1"
                   for column in LABEL_COLUMNS))


def load_judge_rows(path="Judge_train.csv.csv"):
    # utf-8-sig removes the BOM present before sample_index in the source CSV.
    with open(path, encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row.get("answer", "").strip() and valid_labels(row)]
    for row in rows:
        row["label"] = binary_label(row)
    return rows


def grouped_split(rows, n_splits=5, fold=0, random_state=42):
    """Split by question ID so answers to one question never cross folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = [row["label"] for row in rows]
    groups = [row["sample_index"] for row in rows]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(rows, labels, groups))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be between 0 and {len(splits) - 1}")
    return splits[fold]


def grouped_train_validation_test_split(rows, test_fold=0, validation_fold=0,
                                        random_state=42):
    """Return disjoint row indices for training, selection, and final testing."""
    train_validation, test = grouped_split(
        rows, n_splits=5, fold=test_fold, random_state=random_state)
    inner_rows = [rows[index] for index in train_validation]
    inner_train, inner_validation = grouped_split(
        inner_rows, n_splits=4, fold=validation_fold,
        random_state=random_state + 1)
    train = [train_validation[index] for index in inner_train]
    validation = [train_validation[index] for index in inner_validation]
    return train, validation, list(test)
