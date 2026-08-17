# Fanar 1 9B Instruct evaluation

This directory contains the raw generations and two-layer judge outputs for
`QCRI/Fanar-1-9B-Instruct` at pinned Hugging Face revision
`d30938d0efb1fd251727ee96df6987802ff84662`.

The run used the checkpoint's official chat template through vLLM 0.27.1,
bfloat16 weights, temperature 0, the `direct` condition, and 300 questions for
each of MSA, Gulf, Egyptian, Levantine, and Sudanese. Sixteen concurrent
requests improved GPU utilization without changing prompts or decoding.

The four dialect arms came from the candidate files and hashes recorded in
`input_manifest.json`; they remain explicitly marked as provisional,
unreviewed candidates.

Artifacts:

- `generations.csv`: unchanged raw model generations (1,500 rows).
- `scores_layer1.csv`: deterministic Layer 1 outputs.
- `scores_combined.csv`: Layer 1 plus Layer 2 probabilities and final labels.
- `input_manifest.json`: pinned inputs and inference configuration.
- `run_summary.json`: counts, checkpoint provenance, and artifact hashes.

Layer 1 labeled 378 rows clean and 370 hallucinated, deferring 752. Layer 2
scored exactly those 752 deferred rows using threshold 0.5 and checkpoint
weights SHA-256
`57f2d72bad55cfebd521da993e1e38f5fd301bc9ac7a358c546d84ce80f3d81a`.
The combined result contains 854 clean and 646 hallucinated labels, an
automatic hallucination rate of 43.1%.

Unlike the separately retained base-model ablation, this instruction-tuned run
showed no cases where the question or an identical output line was repeated
ten or more times. Its median answer length was 202 characters.

These are automatic judge outputs, not human-validated ground truth. Retain
the raw generations and probabilities when auditing or reporting results.
