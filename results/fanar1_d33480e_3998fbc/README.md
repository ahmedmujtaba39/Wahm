# Fanar 1 9B evaluation

This directory contains the raw generations and two-layer judge outputs for
`QCRI/Fanar-1-9B` at pinned Hugging Face revision
`d33480ed13158518a0902a1d4ac00d66cf4086a9`.

The linked checkpoint is the base model, not `Fanar-1-9B-Instruct`. Following
its model card, the run therefore supplied each question directly through the
OpenAI-compatible text-completions endpoint rather than applying a chat
template. It used vLLM 0.27.1, bfloat16 weights, temperature 0, the `direct`
condition, and 300 questions for each of MSA, Gulf, Egyptian, Levantine, and
Sudanese. Sixteen concurrent requests improved GPU utilization without
changing the prompt or decoding configuration.

The four dialect arms came from the candidate files and hashes recorded in
`input_manifest.json`; they remain explicitly marked as provisional,
unreviewed candidates.

Artifacts:

- `generations.csv`: unchanged raw model generations (1,500 rows).
- `scores_layer1.csv`: deterministic Layer 1 outputs.
- `scores_combined.csv`: Layer 1 plus Layer 2 probabilities and final labels.
- `input_manifest.json`: pinned inputs and inference configuration.
- `run_summary.json`: counts, checkpoint provenance, and artifact hashes.

Layer 1 labeled 97 rows clean and 942 hallucinated, deferring 461. Layer 2
scored exactly those 461 deferred rows using threshold 0.5 and checkpoint
weights SHA-256
`57f2d72bad55cfebd521da993e1e38f5fd301bc9ac7a358c546d84ce80f3d81a`.
The combined result contains 264 clean and 1,236 hallucinated labels.

These are automatic judge outputs, not human-validated ground truth. The base
checkpoint frequently produces long continuations, so this result should not
be presented as an evaluation of the separately released instruction-tuned
Fanar checkpoint. Retain the raw generations and probabilities when auditing
or reporting results.
