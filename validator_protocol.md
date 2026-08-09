# WAHM — Native Speaker Validator Protocol

## Overview

WAHM uses 8 native Arabic speakers (2 per dialect) across three stages of the
project. Having two speakers per dialect enables inter-annotator agreement
measurement, which is the standard reviewers expect for validating both
translation quality and annotation reliability.

## Dialect Coverage

| Dialect    | Validators | Sub-variety (to be confirmed) |
|------------|------------|-------------------------------|
| Gulf       | 2          |                               |
| Egyptian   | 2          |                               |
| Levantine  | 2          |                               |
| Sudanese   | 2          |                               |

Each validator should note their specific sub-variety (e.g. Najdi, Cairene,
Palestinian, Khartoum) on their first deliverable. This is reported in the
paper's methods section.

## Where Native Speakers Are Needed

### Round 1: Few-Shot Exemplar Authoring

**What:** Each validator translates 10 pre-selected MSA questions into their
dialect by hand. These 10 pairs become the few-shot examples that anchor
GPT-4o's translations for all 300 questions.

**Why it matters:** These exemplars are the single highest-leverage quality
control in the project. Every downstream translation is shaped by them.

**Time commitment:** ~30 minutes per validator.

**Deliverable:** Completed `exemplars_<dialect>.csv` with dialect translations,
sub-variety noted, following CODA orthography guidelines.

**Inter-annotator use:** Both validators per dialect independently translate the
same 10 questions. Agreement between the two versions measures how consistent
the dialect target is (are both speakers producing recognizably similar Gulf,
or are sub-variety differences large enough to note?).

### Round 2: Translation Validation

**What:** GPT-4o translates all 300 MSA questions into each dialect using the
few-shot exemplars. Each validator reviews the machine output for their dialect
and either approves or rewrites each translation.

**Why it matters:** This is what makes the benchmark human-validated rather than
raw machine translation. The human acceptance rate, edit distance, and fluency
ratings become the translation-quality table in the paper.

**Time commitment:** ~2-3 hours per validator (300 rows, most are approve-only,
rewrites take longer).

**Deliverable:** Completed `validation_<dialect>.csv` with APPROVE/REWRITE
decisions, rewrites where needed, and a 1-5 fluency/naturalness score.

**Inter-annotator use:** Both validators review the same 300 translations
independently. Cohen's kappa on their approve/rewrite decisions is reported as
translation-quality inter-annotator agreement. Where the two disagree, a
reconciliation pass produces the final version (take the rewrite if either
validator rewrote, or discuss).

**Workload split option:** If 300 rows twice is too much volunteer time, a
practical compromise is: both validators review a shared overlap set of 50-75
questions (for the agreement number), and split the remaining 225-250 between
them (for coverage). The paper reports agreement on the overlap and notes the
split.

### Round 3: Hallucination Annotation (Layer 3)

**What:** After the models generate answers and the automatic judge (Layers 1
and 2) scores them, a stratified sample of ~150 scored outputs per dialect is
drawn. Validators hand-label each as hallucinated or clean, without seeing the
automatic label.

**Why it matters:** This is the proof that the MSA-trained automatic judge still
works on dialect outputs. Cohen's kappa between the automatic pipeline and
human labels, per dialect, is the trust number for the entire scoring system.
If kappa drops on Sudanese relative to Gulf, that itself is a finding about
where the judge struggles.

**Time commitment:** ~1-1.5 hours per validator (75 outputs each if split, or
~150 each if both review the full sample for agreement).

**Deliverable:** Completed `layer3_<dialect>.csv` with binary hallucinated/clean
labels per output.

**Inter-annotator use:** Both validators label the same sample independently.
Cohen's kappa on their hallucination labels is the annotation-reliability
number. This is separate from and additional to the pipeline-vs-human kappa.

## Summary of Time Commitment Per Validator

| Round                      | Estimated time | When needed           |
|----------------------------|----------------|-----------------------|
| 1. Exemplar authoring      | 30 min         | Now                   |
| 2. Translation validation  | 2-3 hours      | After translations    |
| 3. Hallucination annotation| 1-1.5 hours    | After model answers   |
| **Total**                  | **~4-5 hours**  | Spread across weeks   |

## Key Principles

1. Validators work independently within each round (no discussing answers
   before submitting). Agreement is measured, not manufactured.
2. All deliverables are CSV files with clear column headers and pre-filled
   content where possible, to minimize friction.
3. Validators are credited as contributors in the paper's acknowledgments
   (or as co-authors if their contribution warrants it, to be discussed with
   Dr. Kamran).
4. No specialized NLP knowledge is required. The only requirement is native
   fluency in the target dialect and basic Arabic literacy.
