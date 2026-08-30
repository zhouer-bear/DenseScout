# Datasets

## VisDrone

Use the official VisDrone DET layout:

```text
VisDrone2019-DET-train/images/*.jpg
VisDrone2019-DET-train/annotations/*.txt
VisDrone2019-DET-val/images/*.jpg
VisDrone2019-DET-val/annotations/*.txt
```

Convert annotations to DenseScout records:

```bash
python tools/prepare_visdrone.py --root /path/to/visdrone --split train --output data/visdrone/train_records.json
python tools/prepare_visdrone.py --root /path/to/visdrone --split val --output data/visdrone/val_records.json
```

Records schema:

```json
{
  "image_id": "string",
  "file_name": "/absolute/or/relative/image.jpg",
  "width": 1920,
  "height": 1080,
  "dataset": "visdrone",
  "objects": [{"bbox_xyxy": [x1, y1, x2, y2], "category_id": 4}]
}
```

Ignored VisDrone regions are skipped by default: category outside 1..10 or score flag 0 is excluded. Bbox centers are mapped from original pixels to the 640 proxy and then to the 80x80 training grid.

### VisDrone val annotation count audit

The release converter follows the public VisDrone DET convention and keeps only non-ignored official object classes 1--10. A read-only audit of the public val annotations found:

| Quantity | Count | Meaning |
|---|---:|---|
| total annotation rows | 40,169 | all rows in val annotation txt files |
| ignored-region rows | 1,410 | rows with ignore flag/score 0 |
| ignored class-0 rows | 1,378 | ordinary ignored regions |
| ignored class-11 rows | 32 | score-0 ignored/other rows, outside official classes 1--10 |
| removed invalid boxes | 0 | no non-positive width/height rows among retained classes |
| removed categories | class 0: 1,378; class 11: 32 | both excluded by the public converter |
| final retained targets | 38,759 | official non-ignored classes 1--10 |

This explains the historical 38,791 vs. 38,759 discrepancy: 38,791 equals the count after excluding only the 1,378 class-0 ignored rows, while 38,759 additionally excludes the 32 score-0 class-11 ignored rows. Public v0.1.0-alpha.1 reports 38,759 as the effective evaluation target count for official VisDrone classes 1--10.

## DOTA and InsPLAD

DOTA support is planned unless the slicing, overlap, boundary handling, and records generation are fully reproduced in a public workflow. InsPLAD is a public third-party dataset and is not redistributed in this repository. The proprietary Lab_Dataset is not publicly released. The current v0.1.0-alpha.1 workflow focuses on VisDrone.
