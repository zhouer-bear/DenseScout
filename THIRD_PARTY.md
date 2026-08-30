# Third-Party and Licensing Notes

DenseScout v0.1.0-alpha.1 source code in this repository is licensed under Apache-2.0.

## Repository-owned files

- `densescout/`, `tools/`, `tests/`, `configs/`, and `docs/`: DenseScout project source and documentation, Apache-2.0.
- `assets/demo.jpg`: synthetic demonstration image generated specifically for this release package; no private data, no third-party image source.
- No checkpoint files are bundled in this source-only alpha.

## Runtime dependencies

These packages are installed by users and are not vendored in this repository:

| Dependency | Purpose | License note |
|---|---|---|
| PyTorch | model runtime and training | BSD-style license |
| torchvision | optional tensor/image utilities | BSD-style license |
| timm | MobileNetV3-Small backbone implementation | Apache-2.0 |
| numpy | numerical arrays | BSD-style license |
| OpenCV Python | image I/O and resizing | Apache-2.0 |
| Pillow | image metadata in dataset preparation | HPND/PIL Software License |
| PyYAML | optional config parsing | MIT |
| pytest | tests | MIT |

## Not bundled

The following third-party or vendor-specific systems were used in historical experiments or optional internal probes but are not copied into this release tree:

- Ultralytics YOLO: optional downstream detector experiments only.
- NanoDet: historical detector baseline only.
- DPR: historical selector baseline only.
- TensorRT: optional deployment/profiling environment only.
- RKNN/RGA/DMA wrappers: vendor/internal deployment probes only.

Because these codebases are not redistributed here, users should follow each upstream project's license and installation instructions if they reproduce optional baselines.
