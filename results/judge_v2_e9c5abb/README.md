# Judge v2 accepted checkpoint

This is the accepted AraBERT gold-answer factual judge trained from commit
`e9c5abbf2048c7e2d5762036b722aebc154e7295`. It uses the binary
`judge_train.csv` labels and question-disjoint train/validation/test groups.

## Held-out results

| Partition | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8212 | 0.8392 | 0.8463 | 0.8428 | 0.8856 |
| Test | 0.8329 | 0.8283 | 0.8849 | 0.8557 | 0.9122 |

The validation-selected decision threshold is **0.235**. Test labels were not
used for checkpoint or threshold selection.

## Training curve

| Epoch | Train loss | Validation loss | Validation F1 | Validation ROC-AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.4821 | 0.5041 | 0.7746 | 0.8421 |
| 2 | 0.3108 | 0.4651 | 0.8174 | 0.8799 |
| 3 | 0.2339 | 0.4730 | 0.8317 | 0.8856 |

Validation loss reaches its minimum at epoch 2 and rises slightly at epoch 3,
while validation F1 and ROC-AUC continue to improve. Report this as mild loss
divergence and retain the validation-ROC-AUC-selected checkpoint.

## Integrity

- The checkpoint is published as the `model.safetensors` asset on the
  [`judge-v2-e9c5abb` GitHub release](https://github.com/cyberuniversal/Wahm/releases/tag/judge-v2-e9c5abb).
- `model.safetensors`: 540,803,072 bytes, SHA-256
  `04115c581e64be35a1e4ffc705e0ff4bec8d77110a6395dfe144b6eeb3e7c895`
- `judge_metadata.json`: SHA-256
  `579273c192d3a39092e31c60b8dc64dc5c637d46c8b7f485b33e0e5bedba3305`
- `epoch_metrics.csv` contains the exported per-epoch curve.
- `training_config.json` contains the split, hyperparameters, versions, and
  threshold-selection rule.
- Validation and test prediction CSVs permit independent metric reproduction.
