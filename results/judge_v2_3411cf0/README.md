# Judge v2 question-aware checkpoint

This directory contains the frozen Layer 2 winner from the prespecified,
question-grouped input ablation. All three variants used seed 42, identical
train/validation/test question partitions, and deterministic model
initialization. Variants were ranked using validation ROC-AUC only; the test
fold was evaluated once after `question_gold_answer` was frozen as the winner.

## Input ablation

| Input | Threshold | Validation F1 | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|---:|---:|
| answer only | 0.425 | 0.7636 | 0.7870 | 0.8380 |
| gold + answer | 0.355 | 0.8466 | 0.8869 | 0.9170 |
| question + gold + answer | 0.135 | 0.8473 | **0.8995** | **0.9252** |

The frozen winner obtained test F1 0.8617, ROC-AUC 0.9180, and PR-AUC
0.9331 on 838 held-out examples. Its threshold-selected test confusion matrix
is TN=262, FP=107, FN=33, TP=436.

`epoch_metrics.csv` contains training and validation loss and validation
metrics for every epoch. Validation loss reached its minimum at epoch 2 and
rose at epoch 3, while validation ROC-AUC still increased from 0.8977 to
0.8995. That is mild loss-level overfitting, not a collapse in ranking
quality; checkpoint selection followed the prespecified ROC-AUC criterion.

The 541 MB `model.safetensors` file is intentionally excluded from Git. Its
SHA-256 is
`ed0bb0387eed6a94519aea596795e2f5520f552b9c55f35343cd6e343f01c751`.
It is published as the `model.safetensors` asset on the
[`judge-v2-3411cf0`](https://github.com/cyberuniversal/Wahm/releases/tag/judge-v2-3411cf0)
release.
The remaining files record the tokenizer, full training configuration,
per-epoch history, validation/test predictions, per-tag recalls, split
manifest, and ablation selection evidence.
