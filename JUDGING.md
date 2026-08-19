# WAHM judging pipeline

This document is the operational specification for assigning hallucination
labels. It separates judge development evidence from WAHM's final dialect
evaluation.

## Source labels

`Judge_train.csv.csv` is the 4,200-row raw AraHalluEval source. The training
input is the validated `judge_train.csv`: five empty answers are removed,
leaving 4,195 examples from 300 questions. Its `hallucinated` column is the
binary training target; the nine source type columns remain only for auditing:

- `clean = 0` when every type is zero;
- `hallucinated = 1` when any type is one.

The source column `K0wledge Source Conflict` is intentionally spelled as it is
in the CSV. Renaming it without migrating the dataset would silently discard
that label.

## Layer 1

`score_layer1.py` is a mechanical router, not a factual judge. It sends an
answer to the separate `degeneration` outcome only for observable generation
failures: an explicit generation error, an empty answer, foreign CJK/Hangul
script above the conservative density threshold, role markers, code fences,
or heavy repeated token sequences. A stray foreign token remains an audit
signal rather than automatically invalidating the response. Every
other answer is deferred to Layer 2.

Normalized gold-token coverage is retained as a diagnostic feature only. It
must never determine the label: zero coverage can be caused by morphology,
spelling, or dialectal wording, while high coverage can coexist with an
unsupported extra claim. The pre-v2 coverage rules and their source-label
agreement in `results/layer1_arahallueval.json` are historical development
results and must not be used for final WAHM scores.

## Layer 2 development split

The processed file omits the raw `sample_index`, so the loader deterministically
reconstructs it from the 300 unique question strings. All answers for one
question remain in the same partition. With seed 42,
test fold 0, and validation fold 0:

| Partition | Questions | Answers | Hallucinated rate |
|---|---:|---:|---:|
| Train | 180 | 2,518 | 55.7% |
| Validation | 60 | 839 | 56.6% |
| Test | 60 | 838 | 56.0% |

The validation partition selects the best checkpoint and the binary decision
threshold. The test partition is evaluated exactly once after selection. The
saved `judge_metadata.json` contains the complete question-ID manifest,
threshold, package versions, validation metrics, and test metrics.

The old notebook's row-random split leaked questions across partitions and
must not be cited. Its TF-IDF AUC of 0.82 does not survive question-disjoint
evaluation; five-fold grouped TF-IDF yields mean AUC 0.578.

### Layer 2 input ablation

The final input representation is selected from three prespecified variants:
answer only, reference answer plus candidate answer, and question plus
reference answer plus candidate answer. All variants use the identical frozen
question-grouped split and training configuration. Each variant selects its
checkpoint by validation ROC-AUC and calibrates its threshold by validation
F1. The variants are then ranked by validation ROC-AUC, with validation F1 as
the tie-break. Test metrics are unavailable during this selection step; the
held-out test partition is evaluated exactly once for the frozen winner.

Metadata additionally records PR-AUC, the confusion matrix components, and
recall for each of the nine overlapping AraHalluEval source tags. Per-tag
figures are recalls for a binary classifier, not nine-class accuracy.

## Training

On a CUDA training machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-judge.txt
python train_judge.py --output arabert_judge_gold_answer
```

Run any prespecified input variant separately:

```powershell
python train_judge.py --input-variant question_gold_answer --output arabert_judge_question_gold_answer
```

The accepted question-aware checkpoint was trained on NRP Nautilus from
commit `3411cf028f7148a49b6b5ddaa73c7d829663701a` using `judge_train.csv`.
Validation selected `question_gold_answer` over `gold_answer` and
`answer_only`, with ROC-AUC 0.8995 and PR-AUC 0.9252. Its validation-selected
threshold is 0.135. The single held-out test evaluation obtained F1 0.8617,
ROC-AUC 0.9180, and PR-AUC 0.9331. The checkpoint, predictions, training
history, and ablation metadata are packaged in `results/judge_v2_3411cf0/`;
the weight hash is
`ed0bb0387eed6a94519aea596795e2f5520f552b9c55f35343cd6e343f01c751`.
The 541 MB checkpoint is distributed as a GitHub release asset because public
forks cannot add objects to the upstream repository's Git LFS store.

Training loss decreased across all three epochs. Validation loss was lowest
at epoch 2 and increased at epoch 3, while validation ROC-AUC increased from
0.8977 to 0.8995. This is recorded as mild loss-level overfitting; the final
checkpoint follows the prespecified validation ROC-AUC selection rule.

The local environment passed an end-to-end tiny-model smoke test, whose metrics
are not research results. An earlier full run used the raw-source filename and
is not accepted after the binary training schema was formalized.

## Combined scoring

```powershell
python score_layer1.py
python score_layer2.py --model arabert_judge_gold_answer
```

Layer 2 scores only deferred rows. Its threshold is read from the model's
`judge_metadata.json`; silently falling back to 0.5 is prohibited. The
resulting `scores_combined.csv` contains the route, Layer 2 probability where
applicable, threshold, final decision, and labels. Final decisions are
`clean`, `factual_hallucination`, or `degeneration`. `combined_label` is the
binary factual label and is intentionally blank for degeneration;
`headline_failure_label` treats either factual hallucination or degeneration
as a failed response.

## Human validation

After all WAHM generations are rescored with one frozen Judge v2 checkpoint:

```powershell
python prepare_judge_audit.py --inputs results/*/scores_combined.csv --size 250
```

This creates two independently shuffled, blinded annotation sheets and a
private key. Sampling is stratified across variety, judge route, predicted
label, and distance from the Layer 2 threshold. Validators assign one of
`clean`, `factual_hallucination`, or `degeneration` and may record a factual
error type. The packet must contain 200--300 rows; the default is 250.

After both validators return their files:

```powershell
python evaluate_judge_audit.py `
  --key audit/judge_v2_audit_key.csv `
  --validator-a audit/judge_v2_validator_a.csv `
  --validator-b audit/judge_v2_validator_b.csv `
  --output results/judge_v2_audit_metrics.json
```

Report inter-annotator kappa, raw agreement, three-class macro F1 and per-class
precision/recall/F1, the confusion matrix, annotation coverage, and the number
requiring adjudication. Do not inspect the private automatic-label key while
annotating.

## Final analysis

`analyze_results.py` calculates paired factual hallucination rates, HDS,
degeneration rates, paired transitions, deterministic paired-bootstrap 95%
confidence intervals, and exact two-sided McNemar tests. Every dialect arm is
compared with MSA on the exact same QIDs. Factual HR and HDS exclude pairs where
either response is degeneration; headline failure rates retain them. The MSA
`direct` arm is the baseline for every dialect condition.
