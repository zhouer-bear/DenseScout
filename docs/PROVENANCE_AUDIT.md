# Provenance Audit

This public-safe provenance note summarizes what was verified from the internal archival copy before preparing DenseScout v0.1. It intentionally redacts internal absolute paths, usernames, and dataset locations.

## Verified implementation evidence

The release implementation was cross-checked against the archived Phase 5 DenseScout files:

- historical `train_dense_fpn.py`: MobileNetV3-Small + Tiny-FPN + heatmap head;
- historical `step1_densescout_selector.py`: Top-K patch-center inference wrapper;
- historical `dataset_dense_unified.py`: public-records training adapter;
- historical benchmark protocol note: fixed patch-budget selection and target recall definitions.

The public release rewrites these components into path-free modules under `densescout/` and `tools/`.

## Architecture

Verified from the archived implementation and reimplemented in `densescout/model.py`:

- MobileNetV3-Small backbone from `timm`;
- C3/C4/C5 features corresponding to stride 8/16/32;
- 64-channel Tiny-FPN with top-down fusion;
- single heatmap head producing `[B, 1, 80, 80]` for `[B, 3, 640, 640]` input;
- parameter count verified by `tools/smoke_test.py`: 1,005,729 parameters.

## Historical checkpoints

Historical checkpoints were found in the private research workspace for VisDrone, DOTA, AI-TOD, and private inspection-data ablations. They are not bundled in this public repository.

paper-exact checkpoint: unavailable / unverified

Evidence: historical checkpoint files exist and several historical metrics were reproduced during internal audits, but exact per-run ledgers for every paper table were not retained consistently. This repository must not present retrained checkpoints as paper-exact.

release checkpoint: retrained with documented public configuration, if included separately

## Verified historical training settings

Verified from archived training scripts:

- one historical tuned recipe used 80 epochs, batch size 32, learning rate 1e-3, AdamW with weight decay 1e-4, and cosine annealing to 1e-6;
- one earlier FPN recipe used 40 epochs, batch size 32, learning rate 5e-4, and Adam;
- some older ablations used the proprietary Lab_Dataset, which is not part of the public benchmark closure. InsPLAD is a public third-party dataset, but it is not redistributed or supported by this source-only alpha package.

Unverified / unavailable:

- random seeds for all historical runs;
- exact epoch counts for every checkpoint;
- exact configs for every paper table;
- complete public ledger tying every metric row to a checkpoint and command.

## Sensitive/proprietary data scope

The proprietary laboratory inspection dataset used for a subset of fixed-K comparisons and ablations is not publicly released because it contains defense-related and geographically sensitive information. The public repository focuses on the core DenseScout implementation and public-benchmark evaluation.
