# RunPod Tutorial: Download and Prepare Court Keypoints Datasets

This tutorial covers downloading all 5 Roboflow court keypoints datasets, validating them, preparing YAMLs, and merging for training a YOLOv8 pose model.

## 0) Setup

```bash
cd /workspace/Padel-Analytics-System
source .venv/bin/activate
python scripts/check_training_env.py
```

## 1) Download all 5 Roboflow datasets (ck01-ck05)

Generate fresh signed export URLs for each dataset from Roboflow before running. Replace `<CK_LINK_X_FRESH>` with actual URLs.

**Dataset mappings:**
- ck01: https://universe.roboflow.com/plaimaker/padel-mhxdf
- ck02: https://universe.roboflow.com/joshs-workspace-p1aa0/padel-court-detection
- ck03: https://universe.roboflow.com/testing-5xxjo/padel-court-fmfv8
- ck04: https://universe.roboflow.com/joaquns-workspace/padel-keypoints-court
- ck05: https://universe.roboflow.com/plaimaker/padel-mhxdf

```bash
mkdir -p data/raw/roboflow_court_keypoints

python scripts/download_and_extract_zip.py --url "<CK_LINK_1_FRESH>" --output-dir data/raw/roboflow_court_keypoints/ck01 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<CK_LINK_2_FRESH>" --output-dir data/raw/roboflow_court_keypoints/ck02 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<CK_LINK_3_FRESH>" --output-dir data/raw/roboflow_court_keypoints/ck03 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<CK_LINK_4_FRESH>" --output-dir data/raw/roboflow_court_keypoints/ck04 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<CK_LINK_5_FRESH>" --output-dir data/raw/roboflow_court_keypoints/ck05 --archive-name yolov8.zip
```

## 2) Verify all downloads completed

```bash
for i in 01 02 03 04 05; do
  echo "===== ck$i"
  ls -lh data/raw/roboflow_court_keypoints/ck$i/
  [ -f "data/raw/roboflow_court_keypoints/ck$i/data.yaml" ] && echo "✓ data.yaml found" || echo "✗ data.yaml MISSING"
done
```

## 3) Check class names and keypoint shapes in all datasets

```bash
for i in 01 02 03 04 05; do
  echo "===== ck$i data.yaml ====="
  sed -n '1,150p' data/raw/roboflow_court_keypoints/ck$i/data.yaml
  echo
done
```

Record the output to determine:
- How many classes each dataset has
- Whether they are detection-only or pose format
- What the `kpt_shape` is (if pose)

## 4) Build path-aware YAML for all datasets (ck01-05)

```bash
python - <<'PY'
from pathlib import Path
import yaml

for key in ["ck01", "ck02", "ck03", "ck04", "ck05"]:
    root = Path("data/raw/roboflow_court_keypoints") / key
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
    
    # Add kpt_shape if pose dataset
    if "kpt_shape" in src:
        out["kpt_shape"] = src["kpt_shape"]

    p = root / "source_with_path.yaml"
    p.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"✓ wrote {p}")
PY
```

## 5) Validate raw datasets before filtering

```bash
for i in 01 02 03 04 05; do
  echo "===== Validating ck$i ====="
  python scripts/validate_yolo_dataset.py \
    --data data/raw/roboflow_court_keypoints/ck$i/source_with_path.yaml \
    --task pose 2>&1 | head -30
  echo
done
```

## 6) Filter each dataset to court-only and prepare for pose training

For each dataset, determine from step 3 if it's detection or pose. If pose, keep kpt_shape. If detection-only or has mixed classes, keep only `court` class.

```bash
# ck01 - court only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_court_keypoints/ck01/source_with_path.yaml \
  --output-dir data/prepared/roboflow_court_keypoints/ck01_court_pose \
  --dataset-name ck01_court_pose \
  --keep-classes court

# ck02 - court only (detection format)
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_court_keypoints/ck02/source_with_path.yaml \
  --output-dir data/prepared/roboflow_court_keypoints/ck02_court_only_det \
  --dataset-name ck02_court_only_det \
  --keep-classes court

# ck03 - court only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_court_keypoints/ck03/source_with_path.yaml \
  --output-dir data/prepared/roboflow_court_keypoints/ck03_court_pose \
  --dataset-name ck03_court_pose \
  --keep-classes court

# ck04 - court only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_court_keypoints/ck04/source_with_path.yaml \
  --output-dir data/prepared/roboflow_court_keypoints/ck04_court_pose \
  --dataset-name ck04_court_pose \
  --keep-classes court

# ck05 - court only
python scripts/filter_yolo_classes.py \
  --data data/raw/roboflow_court_keypoints/ck05/source_with_path.yaml \
  --output-dir data/prepared/roboflow_court_keypoints/ck05_court_pose \
  --dataset-name ck05_court_pose \
  --keep-classes court
```

If any dataset is pose-style with extra columns, add `--truncate-values 12` (for 4 keypoints * 3 = 12 values):

```bash
# Example for a dataset with pose data:
# python scripts/filter_yolo_classes.py \
#   --data data/raw/roboflow_court_keypoints/ckXX/source_with_path.yaml \
#   --output-dir data/prepared/roboflow_court_keypoints/ckXX_court_pose \
#   --dataset-name ckXX_court_pose \
#   --keep-classes court \
#   --truncate-values 12
```

## 7) Validate all prepared datasets

```bash
for i in 01 02 03 04 05; do
  if [ "$i" = "02" ]; then
    suffix="det"
  else
    suffix="pose"
  fi
  echo "===== Validating ck${i}_court_${suffix} ====="
  python scripts/validate_yolo_dataset.py \
    --data data/prepared/roboflow_court_keypoints/ck${i}_court_${suffix}/ck${i}_court_${suffix}.yaml \
    --task pose 2>&1 | head -30
  echo
done
```

## 8) Rebuild RF-only merge (all ck01-05)

```bash
python scripts/merge_yolo_datasets.py \
  --inputs \
    data/prepared/roboflow_court_keypoints/ck01_court_pose \
    data/prepared/roboflow_court_keypoints/ck02_court_only_det \
    data/prepared/roboflow_court_keypoints/ck03_court_pose \
    data/prepared/roboflow_court_keypoints/ck04_court_pose \
    data/prepared/roboflow_court_keypoints/ck05_court_pose \
  --output-dir data/merged_court_keypoints_rf_v1 \
  --dataset-name merged_court_keypoints_rf_v1 \
  --task pose \
  --class-names court \
  --kpt-shape 4 3

python scripts/validate_yolo_dataset.py --data data/merged_court_keypoints_rf_v1/merged_court_keypoints_rf_v1.yaml --task pose
```

## 9) (Optional) Rebuild Zenodo + RF merge if Zenodo data exists

If you have Zenodo court keypoints data at `data/prepared/zenodo_court_keypoints_pose`:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import yaml

inputs=[Path('data/prepared/zenodo_court_keypoints_pose'), Path('data/merged_court_keypoints_rf_v1')]
out=Path('data/merged_court_keypoints_zenodo_rf_v1_fast')

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

yaml_path=out/'merged_court_keypoints_zenodo_rf_v1_fast.yaml'
yaml_content={
    'path': str(out.resolve()),
    'train':'train/images',
    'val':'val/images',
    'test':'test/images',
    'names':{0:'court'},
    'kpt_shape': [4, 3],
    'nc': 1,
}
yaml_path.write_text(yaml.safe_dump(yaml_content,sort_keys=False),encoding='utf-8')
print('Merged dataset written to:', out.resolve())
print('Dataset YAML written to:', yaml_path.resolve())
for s in ['train','val','test']:
    print(f"{s}: {counts[s]} images")
PY

python scripts/validate_yolo_dataset.py --data data/merged_court_keypoints_zenodo_rf_v1_fast/merged_court_keypoints_zenodo_rf_v1_fast.yaml --task pose
```

## 10) Training commands (when ready)

RF-only:

```bash
nohup python scripts/train_yolo.py \
  --preset court-keypoints \
  --model yolov8m-pose.pt \
  --data data/merged_court_keypoints_rf_v1/merged_court_keypoints_rf_v1.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name court_keypoints_rf_v1 \
  > logs/train_court_keypoints_rf_v1.log 2>&1 &
```

Zenodo+RF (if available):

```bash
nohup python scripts/train_yolo.py \
  --preset court-keypoints \
  --model yolov8m-pose.pt \
  --data data/merged_court_keypoints_zenodo_rf_v1_fast/merged_court_keypoints_zenodo_rf_v1_fast.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name court_keypoints_zenodo_rf_v1_fast \
  > logs/train_court_keypoints_zenodo_rf_v1_fast.log 2>&1 &
```