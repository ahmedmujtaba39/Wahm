"""Shared Arabic text utilities used by translation QC and scoring."""

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def normalize_arabic(text):
    """Normalize Arabic for conservative lexical comparison."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _DIACRITICS.sub("", text)
    for source, target in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
                           ("ى", "ي"), ("ة", "ه")):
        text = text.replace(source, target)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def token_f1(left, right):
    """Symmetric set-token F1; suitable as a transparent QC proxy."""
    left_tokens = set(normalize_arabic(left).split())
    right_tokens = set(normalize_arabic(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2 * precision * recall / (precision + recall)
