# Question-aware Judge v2 rescoring results

These are fresh scores for all existing generations using the frozen
`question_gold_answer` Judge v2 checkpoint. No model answers were regenerated.

- Judge training commit: `3411cf028f7148a49b6b5ddaa73c7d829663701a`
- Scoring commit: `5ca2255f1b7a361180c44ce71098a516e42fc475`
- Decision threshold: `0.135`
- Full paired runs: 6 models × 5 varieties × 300 questions = 9,000 rows
- Legacy ALLAM MSA-only run: 300 rows, retained in `allam_msa/` but excluded from paired analysis
- Analysis: 24 MSA-to-dialect arms, 2,000 paired bootstrap samples per arm, exact McNemar tests, and Benjamini-Hochberg correction across all 24 tests

After false-discovery-rate correction at 0.05, four positive automated shifts
remain: ALLAM–Sudanese (HDS 0.1014, 95% CI 0.0405–0.1622),
Fanar-base–Sudanese (0.1055, 0.0391–0.1720), GPT-5.5–Sudanese
(0.0767, 0.0267–0.1267), and Qwen3-8B–Levantine
(0.0648, 0.0239–0.1092). Their BH-adjusted McNemar q-values are all 0.0267.

These remain machine-judged findings, not final paper claims. The blinded
250-item human audit in `results/judge_v2_audit_3411cf0/` must be completed
before interpreting the absolute hallucination rates or dialect shifts.
Fanar base and Qwen3 8B still have unusually high automated absolute rates;
the audit is specifically needed to estimate whether those are real model
failures or residual judge-domain error.

Each model directory contains Layer 1 diagnostics, combined tri-class scores,
and a run summary with exact judge metadata, weight, and score hashes.
`analysis.csv` and `analysis.json` contain paired rates, degeneration rates,
transition counts, confidence intervals, raw p-values, and BH-adjusted
q-values.
