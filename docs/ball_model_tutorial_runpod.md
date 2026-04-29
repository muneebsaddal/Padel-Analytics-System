# RunPod Tutorial: Download and Prepare Ball Detection Datasets

This tutorial covers downloading all 5 Roboflow ball detection datasets, validating them, preparing YAMLs, and merging for training a YOLOv8 detection model.

## 0) Setup

```bash
cd /workspace/Padel-Analytics-System
source .venv/bin/activate
python scripts/check_training_env.py
```

## 1) Download all 5 Roboflow datasets (bl01-bl05)

Generate fresh signed export URLs for each dataset from Roboflow before running. Replace `<BL_LINK_X_FRESH>` with actual URLs.

**Dataset mappings:**
- bl01: https://universe.roboflow.com/plaimaker/padel-mhxdf
- bl02: https://universe.roboflow.com/joshs-workspace-p1aa0/padel-court-detection
- bl03: https://universe.roboflow.com/testing-5xxjo/padel-court-fmfv8
- bl04: https://universe.roboflow.com/joaquns-workspace/padel-keypoints-court
- bl05: https://universe.roboflow.com/plaimaker/padel-mhxdf

```bash
mkdir -p data/raw/roboflow_ball

python scripts/download_and_extract_zip.py --url "<BL_LINK_1_FRESH>" --output-dir data/raw/roboflow_ball/bl01 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_2_FRESH>" --output-dir data/raw/roboflow_ball/bl02 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_3_FRESH>" --output-dir data/raw/roboflow_ball/bl03 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_4_FRESH>" --output-dir data/raw/roboflow_ball/bl04 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl05 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl06 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_6_FRESH>" --output-dir data/raw/roboflow_ball/bl07 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl08 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl09 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl10 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl11 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<BL_LINK_5_FRESH>" --output-dir data/raw/roboflow_ball/bl12 --archive-name yolov8.zip
```

## 2) Verify all downloads completed

```bash
for i in 01 02 03 04 05; do
  echo "===== bl$i"
  ls -lh data/raw/roboflow_ball/bl$i/
  [ -f "data/raw/roboflow_ball/bl$i/data.yaml" ] && echo "✓ data.yaml found" || echo "✗ data.yaml MISSING"
done
```

## 3) Check class names in all datasets

```bash
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  echo "===== bl$i data.yaml ====="
  sed -n '1,150p' data/raw/roboflow_ball/bl$i/data.yaml
  echo
done
```

Record the output to determine:
- How many classes each dataset has
- Whether class is `ball` or something else
- Total image count per split

## 4) Build path-aware YAML for all datasets (bl01-05)

```bash
python - <<'PY'
from pathlib import Path
import yaml

for key in ["bl01", "bl02", "bl03", "bl04", "bl05", "bl06", "bl07", "bl08", "bl09", "bl10", "bl11"]:
    root = Path("data/raw/roboflow_ball") / key
    if not (root / "data.yaml").exists():
        print(f"SKIP {key}: data.yaml not found")
        continue
    
    src = yaml.safe_load((root / "data.yaml").read_text())
    names = src.get("names", [])
    if isinstance(names, list):
        names_map = {i: n for i, n in enumerate(names)}
    else:
        names_map = {int(k): v for k, v in names.items()}

    out = {
        "path": str(root.resolve()),
        "train": str(src["train"]).replace("../", ""),
        "val": str(src["val"]).replace("../", ""),
        "test": str(src["test"]).replace("../", ""),
        "names": names_map,
        "nc": src.get("nc", 1),
    }

    p = root / "source_with_path.yaml"
    p.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"✓ wrote {p}")
PY
```

## 5) Validate raw datasets before filtering

```bash
for i in 01 02 03 04 05; do
  echo "===== Validating bl$i ====="
  python scripts/validate_yolo_dataset.py \
    --data data/raw/roboflow_ball/bl$i/source_with_path.yaml \
    --task detect 2>&1 | head -30
  echo
done
```

## 6) Filter each dataset to ball-only detect datasets

Keep only `ball` class for all datasets.

```bash
# bl01 - ball only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_ball/bl01/source_with_path.yaml \
  --output-dir data/prepared/roboflow_ball/bl01_ball_only \
  --dataset-name bl01_ball_only \
  --keep-classes ball

# bl02 - ball only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_ball/bl02/source_with_path.yaml \
  --output-dir data/prepared/roboflow_ball/bl02_ball_only \
  --dataset-name bl02_ball_only \
  --keep-classes ball

# bl03 - ball only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_ball/bl03/source_with_path.yaml \
  --output-dir data/prepared/roboflow_ball/bl03_ball_only \
  --dataset-name bl03_ball_only \
  --keep-classes ball

# bl04 - ball only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_ball/bl04/source_with_path.yaml \
  --output-dir data/prepared/roboflow_ball/bl04_ball_only \
  --dataset-name bl04_ball_only \
  --keep-classes ball

# bl05 - ball only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_ball/bl05/source_with_path.yaml \
  --output-dir data/prepared/roboflow_ball/bl05_ball_only \
  --dataset-name bl05_ball_only \
  --keep-classes ball
```

## 7) Validate all prepared datasets

```bash
for i in 01 02 03 04 05; do
  echo "===== Validating bl${i}_ball_only ====="
  python scripts/validate_yolo_dataset.py \
    --data data/prepared/roboflow_ball/bl${i}_ball_only/bl${i}_ball_only.yaml \
    --task detect 2>&1 | head -30
  echo
done
```

## 8) Rebuild RF-only merge (all bl01-05)

```bash
python scripts/merge_yolo_datasets.py \
  --inputs \
    data/prepared/roboflow_ball/bl01_ball_only \
    data/prepared/roboflow_ball/bl02_ball_only \
    data/prepared/roboflow_ball/bl03_ball_only \
    data/prepared/roboflow_ball/bl04_ball_only \
    data/prepared/roboflow_ball/bl05_ball_only \
  --output-dir data/merged_ball_rf_v1 \
  --dataset-name merged_ball_rf_v1 \
  --task detect \
  --class-names ball

python scripts/validate_yolo_dataset.py --data data/merged_ball_rf_v1/merged_ball_rf_v1.yaml --task detect
```

## 9) (Optional) Rebuild Zenodo + RF merge if Zenodo data exists

If you have Zenodo ball detection data at `data/prepared/zenodo_ball_detection_f1`:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import yaml

inputs=[Path('data/prepared/zenodo_ball_detection_f1'), Path('data/merged_ball_rf_v1')]
out=Path('data/merged_ball_zenodo_rf_v1_fast')

for split in ['train','val','test']:
    (out/split/'images').mkdir(parents=True, exist_ok=True)
    (out/split/'labels').mkdir(parents=True, exist_ok=True)

counts={s:0 for s in ['train','val','test']}
for root in inputs:
    if not root.exists():
        print(f"SKIP {root}: does not exist")
        continue
    for split in ['train','val','test']:
        in_img=root/split/'images'
        in_lbl=root/split/'labels'
        if not in_img.is_dir():
            continue
        for img in sorted(in_img.iterdir()):
            if not img.is_file():
                continue
            stem=f"{root.name}_{img.stem}"
            out_img=out/split/'images'/f"{stem}{img.suffix.lower()}"
            out_lbl=out/split/'labels'/f"{stem}.txt"
            try:
                if out_img.exists(): out_img.unlink()
                out_img.hardlink_to(img)
            except Exception:
                shutil.copy2(img,out_img)
            lbl=in_lbl/f"{img.stem}.txt"
            if lbl.is_file():
                try:
                    if out_lbl.exists(): out_lbl.unlink()
                    out_lbl.hardlink_to(lbl)
                except Exception:
                    shutil.copy2(lbl,out_lbl)
            else:
                out_lbl.write_text('',encoding='utf-8')
            counts[split]+=1

yaml_path=out/'merged_ball_zenodo_rf_v1_fast.yaml'
yaml_content={
    'path': str(out.resolve()),
    'train':'train/images',
    'val':'val/images',
    'test':'test/images',
    'names':{0:'ball'},
}
yaml_path.write_text(yaml.safe_dump(yaml_content,sort_keys=False),encoding='utf-8')
print('Merged dataset written to:', out.resolve())
print('Dataset YAML written to:', yaml_path.resolve())
for s in ['train','val','test']:
    print(f"{s}: {counts[s]} images")
PY

python scripts/validate_yolo_dataset.py --data data/merged_ball_zenodo_rf_v1_fast/merged_ball_zenodo_rf_v1_fast.yaml --task detect
```

## 10) Training commands (when ready)

RF-only:

```bash
nohup python scripts/train_yolo.py \
  --preset ball-detection \
  --model yolov8m.pt \
  --data data/merged_ball_rf_v1/merged_ball_rf_v1.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name ball_detection_rf_v1 \
  > logs/train_ball_detection_rf_v1.log 2>&1 &
```

Zenodo+RF (if available):

```bash
nohup python scripts/train_yolo.py \
  --preset ball-detection \
  --model yolov8m.pt \
  --data data/merged_ball_zenodo_rf_v1_fast/merged_ball_zenodo_rf_v1_fast.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name ball_detection_zenodo_rf_v1_fast \
  > logs/train_ball_detection_zenodo_rf_v1_fast.log 2>&1 &
```






# RunPod Tutorial: Ball Detection Dataset Pipeline (Corrected)

This version follows the same workflow as the player pipeline:
filter → validate → merge → train

---

## 0) Setup

```bash
cd /workspace/Padel-Analytics-System
source .venv/bin/activate
python scripts/check_training_env.py
```

---

## 1) Build path-aware YAML (bl01–bl11)

```bash
python - <<'PY'
from pathlib import Path
import yaml

for key in ["bl01","bl02","bl03","bl04","bl05","bl06","bl07","bl08","bl09","bl10","bl11"]:
    root = Path("data/raw/roboflow_ball") / key
    if not (root / "data.yaml").exists():
        print(f"SKIP {key}")
        continue
    
    src = yaml.safe_load((root / "data.yaml").read_text())
    names = src.get("names", [])
    names_map = {i:n for i,n in enumerate(names)} if isinstance(names,list) else {int(k):v for k,v in names.items()}

    out = {
        "path": str(root.resolve()),
        "train": str(src["train"]).replace("../",""),
        "val": str(src["val"]).replace("../",""),
        "test": str(src["test"]).replace("../",""),
        "names": names_map,
        "nc": src.get("nc",1),
    }

    (root/"source_with_path.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
    )
    print(f"✓ {key}")
PY
```

---

## 2) Filter ALL datasets → ball-only detection

This is the critical step.

```bash
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  echo "===== Filtering bl$i ====="

  python scripts/filter_yolo_classes.py \
    --data data/raw/roboflow_ball/bl$i/source_with_path.yaml \
    --output-dir data/prepared/roboflow_ball/bl${i}_ball_only \
    --dataset-name bl${i}_ball_only \
    --keep-classes ball \
    --truncate-values 5

done
```

✔ Handles:

* segmentation → converted
* pose → truncated
* multi-class → filtered

---

## 3) Validate cleaned datasets

```bash
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  echo "===== Validating bl${i}_ball_only ====="

  python scripts/validate_yolo_dataset.py \
    --data data/prepared/roboflow_ball/bl${i}_ball_only/bl${i}_ball_only.yaml \
    --task detect

done
```

👉 Now validation should pass cleanly

---

## 4) Merge all datasets

```bash
python scripts/merge_yolo_datasets.py \
  --inputs \
    data/prepared/roboflow_ball/bl01_ball_only \
    data/prepared/roboflow_ball/bl02_ball_only \
    data/prepared/roboflow_ball/bl03_ball_only \
    data/prepared/roboflow_ball/bl04_ball_only \
    data/prepared/roboflow_ball/bl05_ball_only \
    data/prepared/roboflow_ball/bl06_ball_only \
    data/prepared/roboflow_ball/bl07_ball_only \
    data/prepared/roboflow_ball/bl08_ball_only \
    data/prepared/roboflow_ball/bl09_ball_only \
    data/prepared/roboflow_ball/bl10_ball_only \
    data/prepared/roboflow_ball/bl11_ball_only \
  --output-dir data/merged_ball_rf_v2 \
  --dataset-name merged_ball_rf_v2 \
  --task detect \
  --class-names ball
```

---

## 5) Validate merged dataset

```bash
python scripts/validate_yolo_dataset.py \
  --data data/merged_ball_rf_v2/merged_ball_rf_v2.yaml \
  --task detect
```

---

## 6) Train

```bash
nohup python scripts/train_yolo.py \
  --preset ball-detection \
  --model yolov8m.pt \
  --data data/merged_ball_rf_v2/merged_ball_rf_v2.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 \
  --name ball_detection_rf_v2 \
  > logs/train_ball_detection_rf_v2.log 2>&1 &
```
