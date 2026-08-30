# Reproducibility Status

DenseScout v0.1.0-alpha.1 is a source-only prerelease. It makes the core implementation inspectable and runnable, but it does not claim full paper-result reproduction.

## Publicly Verifiable in This Alpha

- Core architecture: MobileNetV3-Small, Tiny-FPN, and a single heatmap head.
- Model shape: a 640x640 RGB input produces a 1x80x80 response map.
- Model parameter count: approximately 1.01M parameters.
- Reported FLOPs: 0.72 GFLOPs is verified under the THOP convention that reports one MAC as one operation.
- Decoder behavior: sigmoid scores, local-max filtering, positive sparse peaks, descending Top-K, and explicit K_eff.
- Hit criterion: the selector uses a 64x64 square cell, not a circular radius and not a 640x640 backend crop.
- VisDrone preparation pipeline: public annotation parsing and records generation can be inspected and run by users with the official dataset.
- CPU smoke tests and unit tests.

## Not Reproduced by This Alpha

- Paper-exact quantitative results are not reproduced because no paper-exact checkpoint is bundled.
- No public release checkpoint is included in this alpha.
- Lab_Dataset results are private-data dependent and are not publicly reproducible.
- InsPLAD is a public third-party dataset, but this alpha does not redistribute it or provide a complete InsPLAD workflow.
- Full board-level Copy-Avoidance wrappers, vendor-specific deployment code, and raw hardware traces are not included.

## Recommended Use

Use this alpha to inspect the model, loss, decoder, coordinate mapping, VisDrone conversion logic, and evaluation interface. Treat all example commands without a checkpoint as API sanity checks rather than benchmark reproduction.
