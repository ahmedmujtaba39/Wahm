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
    has_type_columns = any(column in row for column in LABEL_COLUMNS)
    derived = int(any(row.get(column, "").strip() == "1"
                      for column in LABEL_COLUMNS))
    supplied = row.get("hallucinated", "").strip()
    if not supplied:
        return derived
    if supplied not in {"0", "1"}:
        raise ValueError(f"invalid binary hallucinated label: {supplied!r}")
    label = int(supplied)
    if has_type_columns and label != derived:
        raise ValueError("binary hallucinated label disagrees with type columns")
    return label


def load_judge_rows(path="judge_train.csv"):
    # utf-8-sig also supports the BOM in the unprocessed source CSV.
    with open(path, encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row.get("answer", "").strip() and valid_labels(row)]
    question_groups = {}
    for row in rows:
        row["label"] = binary_label(row)
        if not row.get("sample_index", "").strip():
            question = row.get("question", "").strip()
            if not question:
                raise ValueError("judge row is missing both sample_index and question")
            if question not in question_groups:
                question_groups[question] = str(len(question_groups))
            row["sample_index"] = question_groups[question]
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
