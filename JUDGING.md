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

`score_layer1.py` applies two transparent checks:

1. Degeneration detection: foreign CJK/Hangul script, role markers, code
   fences, or heavy repeated token sequences are labeled hallucinated.
2. Gold-token coverage after Arabic normalization and conservative proclitic
   handling:
   - coverage >= 0.9: clean;
   - coverage = 0: hallucinated;
   - otherwise: defer to Layer 2.

Against all 4,195 usable source labels, this implementation resolves 2,580
answers (61.5%) with 81.8% accuracy on resolved rows. These values are recorded
in `results/layer1_arahallueval.json`. They measure agreement with the noisy
source labels, not dialect generalization.

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

## Training

On a CUDA training machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-judge.txt
python train_judge.py --output arabert_judge_gold_answer
```

Run the answer-only ablation separately:

```powershell
python train_judge.py --answer-only --output arabert_judge_answer_only
```

The local environment passed an end-to-end tiny-model smoke test. Its metrics
are not research results. An earlier full run used the raw-source filename and
is not accepted as the final Layer 2 artifact after the binary training schema
was formalized. The next NRP Nautilus run must use `judge_train.csv`; only its
saved `judge_metadata.json` may be cited.

## Combined scoring

```powershell
python score_layer1.py
python score_layer2.py --model arabert_judge_gold_answer
```

Layer 2 scores only deferred rows. Its threshold is read from the model's
`judge_metadata.json`; silently falling back to 0.5 is prohibited. Layer 1
labels pass through unchanged. The resulting `scores_combined.csv` contains the
route, Layer 2 probability where applicable, final decision, and binary label.

## Layer 3 dialect validation

After WAHM generations are scored:

```powershell
python prepare_layer3.py --per-dialect 150
```

For every dialect this creates two independently shuffled, blinded annotation
sheets and a private key. Validator sheets contain the question, MSA gold, and
model answer but not the model identity, automatic label, probability, or judge
route.

After both validators return their files:

```powershell
python evaluate_layer3.py `
  --key layer3/layer3_gulf_key.csv `
  --validator-a layer3/layer3_gulf_validator_a.csv `
  --validator-b layer3/layer3_gulf_validator_b.csv `
  --output results/layer3_gulf_agreement.json
```

Report inter-annotator kappa, raw agreement, pipeline agreement with each
validator, pipeline agreement on human-consensus rows, annotation coverage,
and the number requiring adjudication. Do not inspect the private automatic
label key while annotating.

## Final analysis

`analyze_results.py` calculates hallucination rates, HDS, hallucination-set IoU,
and degeneration rate. Every dialect arm is compared with MSA on the exact same
QIDs. The MSA `direct` arm is the baseline for every dialect condition.
