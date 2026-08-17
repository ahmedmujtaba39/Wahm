# GPT-5.5 evaluation

This directory contains the raw generations and two-layer judge outputs for
the Azure OpenAI `gpt-5.5` deployment, model version `2026-04-24`.

The run used a Global Standard deployment through the OpenAI v1 Responses
API, the `direct` condition, eight concurrent requests, and 300 questions for
each of MSA, Gulf, Egyptian, Levantine, and Sudanese. GPT-5.5 rejects the
`temperature` parameter, so it was omitted rather than silently reported as
zero. Requests started with a 2,048-token output budget; empty responses were
retried with budgets of 4,096 and then 8,192 tokens.

The four dialect arms came from the candidate files and hashes recorded in
`input_manifest.json`; they remain explicitly marked as provisional,
unreviewed candidates.

Artifacts:

- `generations.csv`: append-only generation log (1,504 attempts for 1,500
  canonical answers). Four empty attempts are retained before their successful
  retries.
- `scores_layer1.csv`: deterministic Layer 1 outputs for 1,500 canonical rows.
- `scores_combined.csv`: Layer 1 plus Layer 2 probabilities and final labels.
- `input_manifest.json`: pinned inputs and inference configuration.
- `run_summary.json`: counts, judge provenance, and artifact hashes.

Layer 1 labeled 594 rows clean and 229 hallucinated, deferring 677. Layer 2
scored exactly those 677 deferred rows using threshold 0.5 and checkpoint
weights SHA-256
`57f2d72bad55cfebd521da993e1e38f5fd301bc9ac7a358c546d84ce80f3d81a`.
The combined result contains 1,003 clean and 497 hallucinated labels, an
automatic hallucination rate of 33.1%.

Per-variety automatic hallucination rates are 31.3% for MSA, 33.0% for Gulf,
32.0% for Egyptian, 33.7% for Levantine, and 35.7% for Sudanese. Relative to
MSA, the corresponding dialect gaps are +1.7, +0.7, +2.3, and +4.3 percentage
points.

These are automatic judge outputs, not human-validated ground truth. Retain
the raw generations and probabilities when auditing or reporting results.
