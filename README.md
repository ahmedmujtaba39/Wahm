# WAHM

WAHM studies whether open-generation factual hallucination changes when the
same Arabic question is asked in Modern Standard Arabic (MSA) versus Gulf,
Egyptian, Levantine, or Sudanese Arabic.

The primary metric is the Hallucination Drift Score:

`HDS(model, dialect) = HR(model, dialect) - HR(model, MSA)`

A positive HDS means hallucination increased in dialect; a negative HDS is a
valid result and means it decreased.

## Current status (August 2026)

| Component | Status | Evidence in this repository |
|---|---|---|
| MSA seed | Complete, pending a source/data audit | `wahm_seed_msa.csv` (300 unique QIDs) |
| Gulf exemplars | Confirmed | `exemplars_gulf.csv` |
| Other dialect exemplars | Collected externally; not yet in this checkout | Awaiting repository files |
| Dialect translations | Not generated/validated | Translation, QC, and finalization scripts only |
| Model generations | Not run | `generate.py` |
| Layer 1 | Implemented; source-label evaluation saved | `score_layer1.py`, `results/layer1_arahallueval.json` |
| Layer 2 | Binary training data validated; clean AraBERT rerun pending | `judge_train.csv`, `train_judge.py`, `score_layer2.py`, `infra/nautilus/` |
| Layer 3 human validation | Blinded preparation/evaluation implemented | `prepare_layer3.py`, `evaluate_layer3.py` |

The earlier extensionless `layer 1` file is a failed AraBERT environment
notebook, not a working Layer 1 implementation. Its row-random split would put
answers to the same question into train and test sets. The new judge scripts
split by question (the processed file's deterministic replacement for raw
`sample_index`), preventing that leakage. Consequently, the
previously stated TF-IDF 75% accuracy/0.82 AUC must not be used. The verified
five-fold question-grouped result is **57.9% mean accuracy and 0.578 mean AUC**;
it is saved in `results/tfidf_grouped_cv.json`.

The implemented Layer 1 resolves **61.5%** of the 4,195 usable AraHalluEval
answers at **81.8% accuracy on resolved rows**. This is an in-domain source-label
evaluation, not proof that it generalizes to dialectal model outputs.

## Pipeline

1. Add the already-collected Egyptian, Levantine, and Sudanese exemplar files
   to the checkout when they are ready to be shared.
2. Install the API dependency and run translation, QC, and finalization:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:OPENAI_API_KEY = "..."
   python translate.py gulf --limit 20
   python qc.py gulf
   python finalize.py gulf
   ```

   `finalize.py` intentionally refuses to produce a final arm until the
   validation sheet contains human decisions.
3. Pilot generations before the full API run:

   ```powershell
   python generate.py --models gpt4o --varieties msa gulf --limit 20
   python score_layer1.py
   ```

4. Establish the leakage-safe Layer 2 baseline and train AraBERT on a GPU:

   ```powershell
   pip install -r requirements-judge.txt
   python baseline_tfidf.py  # five question-grouped folds
   python train_judge.py
   python score_layer2.py
   python analyze_results.py
   python prepare_layer3.py
   ```

## Generation conditions

- `direct`: zero-shot factual answer.
- `msa_answer`: dialect question, answer constrained to MSA (measurement control).
- `dialect_aware`: system prompt identifies the dialect.
- `msa_pivot`: a first call translates to MSA, then a second call answers it.
- `msa_restate`: the response visibly restates the question in MSA before answering.

Raw outputs are append-only and resumable. The pivot text is stored alongside
the answer so the mitigation can be audited rather than treated as hidden work.
Both requested and provider-resolved model IDs are recorded. Translation is
also resumable and rejects attempts to append a different model/prompt setup to
an existing candidate file.

`analyze_results.py` computes HDS against the MSA `direct` baseline using only
QIDs present in both arms. It also reports hallucination-set IoU and dialect
degeneration rate. This paired restriction prevents translation filtering from
changing the question mix between MSA and a dialect.

`prepare_layer3.py` creates two independently shuffled, blinded validator files
per dialect plus a private key containing the automatic decisions. After both
files return, `evaluate_layer3.py` reports inter-annotator kappa, each
validator's agreement with the pipeline, consensus agreement, and the number
of disagreements requiring adjudication.

## Tests

The local suite does not call paid APIs:

```powershell
python -m unittest discover -s tests -v
```

## Research gates before making claims

- Audit the 300 seed questions and acceptable gold variants against the stated
  TyDiQA provenance; several rows visibly need spelling/variant review.
- Rerun TF-IDF and AraBERT with question-grouped splits and save metrics.
- Calibrate Layer 1's 0.9/0 thresholds against held-out human labels. Zero
  lexical overlap is a hypothesis, not automatically a trustworthy label for
  paraphrastic answers.
- Complete two-speaker translation agreement and dialect authenticity checks.
- Validate the combined automatic judge separately for every dialect before
  reporting HDS.

See `validator_protocol.md` for the three native-speaker rounds.
See `JUDGING.md` for the exact judge split, calibration, inference, and Layer 3
agreement protocol.
See `infra/nautilus/README.md` for the established NRP Nautilus execution path.
