# Judge v2 rescoring results

These are Judge v2 scores for the existing generations. No model answers were regenerated.

- Judge checkpoint: `e9c5abbf2048c7e2d5762036b722aebc154e7295`
- Decision threshold: `0.235`
- Full paired runs: 6 models x 5 varieties x 300 questions = 9,000 scored rows
- Additional legacy ALLAM MSA-only run: 300 rows (retained in `allam_msa/`, excluded from the joint paired analysis)
- Analysis: 24 MSA-to-dialect arms, paired by QID, with 2,000 paired bootstrap samples and exact McNemar tests

`analysis.csv` and `analysis.json` report factual hallucination rates after excluding pairs with degeneration. Headline failure rates keep degeneration as a separate failure mode. The individual model directories contain Layer 1 diagnostics, combined tri-class decisions, and hash-verified run summaries.

At an uncorrected alpha of 0.05, the automated Judge v2 analysis finds two positive dialect shifts whose bootstrap intervals exclude zero and whose exact McNemar tests are below 0.05: ALLAM on Sudanese (HDS 0.0642, 95% CI 0.0134 to 0.1182, p=0.0226) and Qwen3 8B on Levantine (HDS 0.0451, 95% CI 0.0074 to 0.0865, p=0.0428). These are provisional machine-judged findings, not final paper claims. Confirm them with the blinded human audit before interpretation or multiple-comparison correction.

The high automated rates for Qwen3 8B and Fanar base remain after removing the old Layer 1 semantic shortcuts and using the selected validation threshold. That rules out those two specific implementation causes, but it does not establish that the rates are accurate. The stratified human audit is required to estimate judge precision, recall, and F1 on these outputs.
