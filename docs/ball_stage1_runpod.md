# Ball Stage 1 On RunPod

This guide builds the first expanded ball-detection dataset by combining:

1. the existing Zenodo ball source already used for the baseline
2. one large Roboflow ball source

Recommended Roboflow source for Stage 1:

- `padel-vyxal/padel-dataset-6ywe6`

Do not merge both `padel-vyxal/padel-dataset-6ywe6` and `padelsense/padel-dataset-j1pcr` in the same first pass. They appear very similar in description and size, so treat them as likely duplicates until inspected.

## Goal

Improve ball detection while keeping the first expansion controlled and interpretable.

## 1. Activate environment

```bash
cd /workspace/Padel-Analytics-System
source .venv/bin/activate
```

## 2. Keep the current Zenodo ball dataset

This assumes you already have:

- `data/prepared/zenodo_ball_f1/zenodo_ball_f1.yaml`

Validate it again if needed:

```bash
python scripts/validate_yolo_dataset.py \
  --data data/prepared/zenodo_ball_f1/zenodo_ball_f1.yaml \
  --task detect
```

## 3. Download the Roboflow ball dataset

From the Roboflow dataset page, choose `YOLOv8` or `Ultralytics YOLO` download format and copy the generated ZIP URL.

Dataset page:

- `https://universe.roboflow.com/padel-vyxal/padel-dataset-6ywe6`

Download and extract:

```bash
python scripts/download_and_extract_zip.py \
  --url "<ROBOFLOW_ZIP_URL_PADEL_DATASET_6YWE6>" \
  --output-dir data/raw/roboflow_ball_stage1 \
  --archive-name roboflow_ball_stage1.zip
```

## 4. Normalize the Roboflow export

```bash
python scripts/prepare_yolo_dataset.py \
  --source-dir data/raw/roboflow_ball_stage1 \
  --output-dir data/prepared/roboflow_ball_stage1_raw \
  --task detect \
  --dataset-name roboflow_ball_stage1_raw
```

## 5. Filter to ball-only labels

The `padel dataset` source is a 2-class dataset with `ball` and `player`.
Keep only `ball` before merging it with the Zenodo ball dataset.

```bash
python scripts/filter_yolo_classes.py \
  --data data/prepared/roboflow_ball_stage1_raw/roboflow_ball_stage1_raw.yaml \
  --output-dir data/prepared/roboflow_ball_stage1_ball_only \
  --dataset-name roboflow_ball_stage1_ball_only \
  --keep-classes ball
```

## 6. Validate the filtered Roboflow dataset

```bash
python scripts/validate_yolo_dataset.py \
  --data data/prepared/roboflow_ball_stage1_ball_only/roboflow_ball_stage1_ball_only.yaml \
  --task detect
```

## 7. Merge Zenodo + Roboflow ball-only

```bash
python scripts/merge_yolo_datasets.py \
  --inputs \
    data/prepared/zenodo_ball_f1 \
    data/prepared/roboflow_ball_stage1_ball_only \
  --output-dir data/merged_ball_stage1 \
  --dataset-name merged_ball_stage1 \
  --task detect \
  --class-names ball
```

## 8. Validate the merged dataset

```bash
python scripts/validate_yolo_dataset.py \
  --data data/merged_ball_stage1/merged_ball_stage1.yaml \
  --task detect
```

## 9. Train the Stage 1 ball model

Recommended first run:

```bash
python scripts/train_yolo.py \
  --preset players-detection \
  --model yolov8m.pt \
  --data data/merged_ball_stage1/merged_ball_stage1.yaml \
  --epochs 100 \
  --batch 64 \
  --imgsz 1280 \
  --device 0 \
  --workers 4 \
  --name ball_detection_stage1_zenodo_plus_rf
```

Recommended `nohup` version:

```bash
nohup python scripts/train_yolo.py \
  --preset players-detection \
  --model yolov8m.pt \
  --data data/merged_ball_stage1/merged_ball_stage1.yaml \
  --epochs 100 \
  --batch 64 \
  --imgsz 1280 \
  --device 0 \
  --workers 4 \
  --name ball_detection_stage1_zenodo_plus_rf \
  > train_ball_stage1_zenodo_plus_rf.log 2>&1 &
```

Monitor:

```bash
tail -f train_ball_stage1_zenodo_plus_rf.log
```

## 10. Summarize the new ball run

```bash
python scripts/summarize_baselines.py \
  --run ball_stage1=runs/detect/runs/train/ball_detection_stage1_zenodo_plus_rf \
  --output artifacts/reports/ball_detection_stage1_zenodo_plus_rf.md
```

## Notes

- If the Roboflow class is named something other than `ball`, inspect `data.yaml` first and change `--keep-classes`.
- If the Stage 1 overlay improves ball recall but adds new false positives, stop before adding more datasets and inspect label quality.
- Add only one extra dataset after Stage 1 so the effect of each new source stays measurable.
