"""WAHM Judge v2 Layer 1: mechanical routing and diagnostics only."""

import argparse
import csv
import re

from wahm_text import normalize_arabic

ROLE_MARKER = re.compile(r"(?:^|\n)\s*(?:assistant|user|system)\s*:", re.I)
FOREIGN_GARBAGE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
CODE_FENCE = re.compile(r"```")


def gold_variants(gold):
    """Return non-empty reference variants from the benchmark's CSV field."""
    # Protect decimal commas such as 37,5, then split list separators.
    protected = re.sub(r"(?<=\d),(?=\d)", "<DECIMAL_COMMA>", gold)
    variants = [part.strip() for part in re.split(
        r"\s*(?:,|;|\||،)\s*", protected)]
    variants = [variant.replace("<DECIMAL_COMMA>", ",") for variant in variants]
    return [variant for variant in variants if variant] or [gold.strip()]


def token_coverage(answer, gold):
    """Maximum fraction of a gold variant's tokens found in the answer."""
    answer_tokens = normalize_arabic(answer).split()

    def forms(token):
        """Include conservative one-letter Arabic proclitic variants."""
        result = {token}
        current = token
        for _ in range(2):
            if len(current) > 3 and current[0] in "وفبكل":
                current = current[1:]
                result.add(current)
            else:
                break
        return result

    answer_forms = set().union(*(forms(token) for token in answer_tokens)) \
        if answer_tokens else set()
    scores = []
    for variant in gold_variants(gold):
        reference_tokens = set(normalize_arabic(variant).split())
        if reference_tokens:
            matches = sum(bool(forms(token) & answer_forms)
                          for token in reference_tokens)
            scores.append(matches / len(reference_tokens))
    return max(scores, default=0.0)


def heavy_repetition(text, minimum_repeats=4):
    tokens = normalize_arabic(text).split()
    if len(tokens) < minimum_repeats:
        return False
    for width in (1, 2, 3):
        for start in range(len(tokens) - width * minimum_repeats + 1):
            chunk = tokens[start:start + width]
            if all(tokens[start + i * width:start + (i + 1) * width] == chunk
                   for i in range(minimum_repeats)):
                return True
    return False


def degeneration_reasons(answer, generation_error=""):
    reasons = []
    if generation_error.strip():
        reasons.append("generation_error")
    if not answer.strip():
        reasons.append("empty_answer")
    if FOREIGN_GARBAGE.search(answer):
        reasons.append("foreign_script")
    if ROLE_MARKER.search(answer):
        reasons.append("role_marker")
    if CODE_FENCE.search(answer):
        reasons.append("code_fence")
    if heavy_repetition(answer):
        reasons.append("heavy_repetition")
    return reasons


def score_answer(answer, gold, generation_error=""):
    """Return diagnostic coverage and a mechanical route.

    Coverage never assigns a factual label. Mechanical invalidity is preserved
    as ``degeneration``; every other answer is deferred to the factual judge.
    """
    reasons = degeneration_reasons(answer, generation_error)
    coverage = token_coverage(answer, gold)
    decision = "degeneration" if reasons else "defer"
    return coverage, decision, reasons


def canonical_generations(rows):
    """Keep one row per experiment key, preferring the latest success."""
    canonical = {}
    for row in rows:
        key = (row.get("qid"), row.get("variety"),
               row.get("condition"), row.get("model"))
        current = canonical.get(key)
        success = bool(row.get("answer", "").strip()
                       and not row.get("error", "").strip())
        current_success = bool(current and current.get("answer", "").strip()
                               and not current.get("error", "").strip())
        if current is None or success or not current_success:
            canonical[key] = row
    return list(canonical.values())


def run(input_path="generations.csv", output_path="scores_layer1.csv"):
    with open(input_path, encoding="utf-8", newline="") as source:
        raw_rows = list(csv.DictReader(source))
    rows = canonical_generations(raw_rows)
    if not rows:
        raise SystemExit(f"ERROR: {input_path} contains no generations")

    fields = list(rows[0]) + ["layer1_coverage", "layer1_decision",
                              "degeneration_reasons"]
    with open(output_path, "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            coverage, decision, reasons = score_answer(
                row.get("answer", ""), row.get("gold_answer", ""),
                row.get("error", ""))
            row.update(layer1_coverage=f"{coverage:.3f}",
                       layer1_decision=decision,
                       degeneration_reasons="|".join(reasons))
            writer.writerow(row)

    counts = {name: 0 for name in ("degeneration", "defer")}
    with open(output_path, encoding="utf-8", newline="") as scored:
        for row in csv.DictReader(scored):
            counts[row["layer1_decision"]] += 1
    print(f"wrote {output_path}: {counts} "
          f"({len(raw_rows)} attempts -> {len(rows)} canonical generations)")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generations.csv")
    parser.add_argument("--output", default="scores_layer1.csv")
    args = parser.parse_args()
    run(args.input, args.output)
