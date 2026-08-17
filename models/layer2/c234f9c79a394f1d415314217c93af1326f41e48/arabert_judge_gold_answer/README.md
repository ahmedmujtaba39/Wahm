# WAHM Layer 2 gold+answer judge

This is the selected AraBERT v2 binary hallucination-judge checkpoint from
WAHM run `c234f9c79a394f1d415314217c93af1326f41e48`.

- Base model: `aubmindlab/bert-base-arabertv2`
- Inputs: MSA gold answer and candidate answer
- ROC-AUC: `0.9319397222`
- Accuracy: `0.8605482718`
- Hallucinated-class F1: `0.8790072389`
- `model.safetensors` SHA-256:
  `57f2d72bad55cfebd521da993e1e38f5fd301bc9ac7a358c546d84ce80f3d81a`

The 540.8 MB weights are published as the `model.safetensors` asset on the
[GitHub checkpoint release](https://github.com/cyberuniversal/Wahm/releases/tag/layer2-judge-c234f9c).
Download that asset into this directory before running:

```powershell
gh release download layer2-judge-c234f9c `
  --repo cyberuniversal/Wahm `
  --pattern model.safetensors `
  --dir models/layer2/c234f9c79a394f1d415314217c93af1326f41e48/arabert_judge_gold_answer

python score_layer2.py `
  --model models/layer2/c234f9c79a394f1d415314217c93af1326f41e48/arabert_judge_gold_answer
```

See `judge_metadata.json` for the saved evaluation metadata.
