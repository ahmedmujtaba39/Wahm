# Jais 2 8B Chat evaluation

This directory contains the raw generations and two-layer judge outputs for
`inceptionai/Jais-2-8B-Chat` at pinned Hugging Face revision
`f7df1cb035424ed345b5ee2b04f114721c8951f2`.

The run used vLLM 0.27.1, bfloat16 weights, temperature 0, the `direct`
condition, and 300 questions for each of MSA, Gulf, Egyptian, Levantine, and
Sudanese. The MSA arm came from `wahm_seed_msa.csv` (SHA-256
`5533844b360e762b1014a95d7ba6367605904631cc58236f8b6aac25ff120bcb`).
The four dialect arms came from the candidate files recorded in
`input_manifest.json`; they remain explicitly marked as provisional,
unreviewed candidates.

Transformers 4.46+ identified the model tokenizer's saved Mistral regex as
incorrect for some mixed text, including Arabic followed by digits. The run
therefore used `fix_mistral_regex=True`. Hashes of the derived tokenizer files
are recorded in `input_manifest.json`.

Artifacts:

- `generations.csv`: unchanged raw model generations (1,500 rows).
- `scores_layer1.csv`: deterministic Layer 1 outputs.
- `scores_combined.csv`: Layer 1 plus Layer 2 probabilities and final labels.
- `input_manifest.json`: pinned inputs and inference configuration.
- `run_summary.json`: counts, checkpoint provenance, and artifact hashes.

Layer 1 labeled 362 rows clean and 478 hallucinated, deferring 660. Layer 2
scored exactly those 660 deferred rows using threshold 0.5 and checkpoint
weights SHA-256
`57f2d72bad55cfebd521da993e1e38f5fd301bc9ac7a358c546d84ce80f3d81a`.
The combined result contains 758 clean and 742 hallucinated labels.

These are automatic judge outputs, not human-validated ground truth. Retain
the raw generations and probabilities when auditing or reporting results.
