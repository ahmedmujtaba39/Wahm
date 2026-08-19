# Judge v2 manual audit

This packet contains 250 blinded rows sampled with seed 42.

Allowed `human_label` values: `clean | factual_hallucination | degeneration`.

Suggested `error_type` values: `wrong_entity | wrong_number_date | contradiction | unsupported_elaboration | generic_imprecise | instruction_mismatch | degeneration | other`.

Validators must not inspect `judge_v2_audit_key.csv` until both blinded files are complete. Resolve disagreements through adjudication before reporting final pipeline precision, recall, or F1.
