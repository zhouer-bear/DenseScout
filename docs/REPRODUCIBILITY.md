# Reproducibility

The alpha training utility implements synchronized geometric flipping, AdamW, and cosine annealing. The exact historical HSV jitter parameters were not retained consistently and are therefore not reconstructed in this source-only prerelease.

The original DenseScout experiments were completed before a standardized per-run experiment ledger was introduced. Exact epoch counts, random seeds, and some optimizer hyperparameters were not retained consistently for every historical run. This repository distinguishes verified historical settings from recommended reproduction settings.

Recommended public reproduction command:

```bash
python tools/train.py --train-records data/visdrone/train_records.json --val-records data/visdrone/val_records.json --epochs 60 --batch-size 32 --lr 1e-3 --device cuda --output-dir outputs/visdrone_release
```

Evaluation uses selector Recall with a square 64x64 cell in the 640 proxy coordinate space:

```text
abs(x_pred - x_gt) <= 32 and abs(y_pred - y_gt) <= 32
```

The 640x640 crop used by a downstream detector is a deployment/perception input unit and must not be mixed with the selector hit criterion.
