# Qwen3-8B five-variety evaluation

This directory contains the raw generations and automatic Layer 1/Layer 2
hallucination scores for `Qwen/Qwen3-8B`.

## Configuration

- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Generation commit: `64776fa24d4308a5339f6961439a33deaaf0810e`
- Score commit: `d4211eb483f53b3ed5dc4c48f0fc2d5c1542c55c`
- Run ID: `qwen3-8b-b968826-64776fa`
- Interface: vLLM OpenAI-compatible chat completions
- Temperature: `0.0`
- Qwen thinking mode: disabled through `chat_template_kwargs`
- Condition: direct, zero-shot, no retrieved context
- Layer 2 threshold: `0.5`
- Layer 2 run: `c234f9c79a394f1d415314217c93af1326f41e48`

The dialect inputs are recorded in `input_manifest.json` as provisional
unreviewed candidates. The raw answers and judge probabilities are retained so
the automatic labels can be audited.

## Generation integrity

- 1,500/1,500 successful canonical generations
- 300 generations each for MSA, Gulf, Egyptian, Levantine, and Sudanese
- Zero blank answers and zero request errors
- Zero emitted `<think>` tags

## Automatic judging

Layer 1:

- Clean: 180
- Hallucinated: 595
- Deferred: 725

Layer 2 scored exactly all 725 deferred rows.

Combined:

- Clean: 338
- Hallucinated: 1,162 (77.5%)

Per variety:

| Variety | Clean | Hallucinated | Rate |
| --- | ---: | ---: | ---: |
| MSA | 68 | 232 | 77.3% |
| Gulf | 70 | 230 | 76.7% |
| Egyptian | 65 | 235 | 78.3% |
| Levantine | 69 | 231 | 77.0% |
| Sudanese | 66 | 234 | 78.0% |

This unexpectedly high automatic rate is reported without a causal claim. The
answers are verbose (mean 686.6 characters for clean labels and 841.8 for
hallucinated labels), and 97 answers triggered degeneration checks. A manual
error audit is required before attributing the rate to factual accuracy alone.

## Files

- `generations.csv`: raw model outputs
- `scores_layer1.csv`: deterministic Layer 1 decisions
- `scores_combined.csv`: Layer 1 decisions plus Layer 2 probabilities and labels
- `input_manifest.json`: model and input provenance
- `run_summary.json`: aggregate counts and artifact hashes
