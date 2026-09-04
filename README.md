# DenseScout v0.1.0-alpha.1

Source-only prerelease.

Paper: https://doi.org/10.1145/3767308.3836001  
Code: https://github.com/zhouer-bear/DenseScout  

DenseScout is a budgeted patch-center selector for tiny-object perception on edge platforms. It is not a final detector: it ranks patch centers, and downstream detectors may run only on selected crops.

## Alpha Scope

The current alpha release provides source-code and pipeline validation only. It does not include a paper-exact checkpoint or a public release checkpoint, and it does not claim to reproduce the paper's full quantitative results. The included VisDrone sanity commands verify parsing, inference, and metric code paths; they must not be reported as benchmark results.

This prerelease includes the core model, loss, decoder, VisDrone preparation utility, training/inference/evaluation interfaces, and tests. It excludes private data, paper-exact weights, historical traces, and the proprietary board-level Copy-Avoidance wrapper.

## Install

For CPU-only usage, install PyTorch first from the CPU wheel index, then install the local package.

```bash
git clone https://github.com/zhouer-bear/DenseScout.git
cd DenseScout
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
```

## Quick Checks

```bash
pip check
pytest -q
python tools/smoke_test.py --device cpu
```

## Inference

This source-only alpha does not bundle a checkpoint. The following command is an API sanity check with randomly initialized weights only.

```bash
python tools/infer_image.py \
  --image assets/demo.jpg \
  --topk 9 \
  --device cpu \
  --output outputs/demo.json
```

After you provide a compatible checkpoint, pass it explicitly:

```bash
python tools/infer_image.py \
  --image assets/demo.jpg \
  --checkpoint checkpoints/densescout_visdrone.pt \
  --topk 9 \
  --device cpu \
  --output outputs/demo_with_checkpoint.json
```

## Public Benchmark Evaluation

The following command checks the evaluation API on a prepared public VisDrone records file. Without a checkpoint, the output is not a benchmark result.

```bash
python tools/eval_recall.py \
  --config configs/visdrone_eval.yaml \
  --records data/visdrone/val_records.json \
  --limit-samples 2 \
  --device cpu \
  --output outputs/visdrone_eval_sanity.json
```

After you provide a compatible checkpoint, pass it explicitly and remove `--limit-samples` for full public-val evaluation.

```bash
python tools/eval_recall.py \
  --config configs/visdrone_eval.yaml \
  --records data/visdrone/val_records.json \
  --checkpoint checkpoints/densescout_visdrone.pt \
  --device cpu \
  --output outputs/visdrone_eval.json
```

## Protocol

DenseScout outputs one 80x80 logit map for a 640x640 proxy image. `decode_heatmap` follows the paper-default order: logits -> sigmoid score -> local-max filtering with default kernel size 7 -> sparse score map > 0 -> descending Top-K. It does not threshold raw logits at 0, reports `K_eff`, and never pads missing candidates with zero-score or uniform centers.

The selector localization metric uses a square 64x64 cell:

```text
abs(x_pred - x_gt) <= 32 and abs(y_pred - y_gt) <= 32
```

This 64x64 selector cell is not the same as the 640x640 backend crop. The backend crop is used for downstream perception/deployment experiments.

## Checkpoints

No checkpoint is bundled in this source-only prerelease.

```text
paper-exact checkpoint: unavailable / unverified
release checkpoint: not included yet
```

Without a public release checkpoint and public-data evaluation artifact, this package should be treated as `v0.1.0-alpha.1` rather than a full quantitative reproduction release.

## Data Scope

The proprietary Lab_Dataset used for a subset of fixed-K comparisons and ablations is not publicly released because it contains defense-related and geographically sensitive information.

InsPLAD is a public third-party dataset; it is not redistributed and is not fully supported by the current source-only alpha workflow.

The current public workflow focuses on the core DenseScout implementation and VisDrone preparation/evaluation interfaces.

## Training Scope

The alpha training utility implements synchronized geometric flipping, AdamW, and cosine annealing. The exact historical HSV jitter parameters were not retained consistently and are therefore not reconstructed in this source-only prerelease.

The original DenseScout experiments were completed before a standardized per-run experiment ledger was introduced. Exact epoch counts, random seeds, and some optimizer hyperparameters were not retained consistently for every historical run. This repository distinguishes verified historical settings from recommended reproduction settings.

## Deployment Scope

This repository does not claim to release the complete proprietary board-level Copy-Avoidance pipeline. Vendor-specific and internal deployment wrappers are excluded. Included examples are labeled according to their actual scope.

## Reproducibility Status

See `docs/REPRODUCIBILITY_STATUS.md` for the public reproducibility boundary of this alpha release.

## Citation

See `CITATION.cff` for software metadata and the preferred ACM MM 2026 paper citation.

## Demo Asset

`assets/demo.jpg` is a synthetic grid image generated for this release package. It contains no private data and no third-party image content.

## License

Apache-2.0. See `THIRD_PARTY.md` for dependency and licensing notes.
