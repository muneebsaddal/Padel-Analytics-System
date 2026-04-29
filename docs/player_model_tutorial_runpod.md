# RunPod Tutorial: Continue From Existing Persistent Data

This version assumes your RunPod storage persists and the repo/data are still there.

## 0) Resume on new pod

```bash
cd /workspace/Padel-Analytics-System
source .venv/bin/activate
python scripts/check_training_env.py
```

## 1) Verify existing data is still present

```bash
for d in \
  data/raw/roboflow_players/rf01 \
  data/raw/roboflow_players/rf02 \
  data/raw/roboflow_players/rf03 \
  data/raw/roboflow_players/rf04 \
  data/prepared/roboflow_players/rf01_player_only \
  data/prepared/roboflow_players/rf02_player_only_det \
  data/prepared/roboflow_players/rf03_player_only \
  data/prepared/roboflow_players/rf04_player_only \
  data/merged_players_rf_v1 \
  data/merged_players_zenodo_rf_v1_fast
  do
  [ -d "$d" ] && echo "OK: $d" || echo "MISSING: $d"
done
```

If these are present, continue below and only process remaining sets.

## 2) Download only remaining Roboflow sets (rf05, rf06, rf07)

Generate fresh signed export URLs for datasets 5-7 right before running.

```bash
python scripts/download_and_extract_zip.py --url "<RF_LINK_5_FRESH>" --output-dir data/raw/roboflow_players/rf05 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<RF_LINK_6_FRESH>" --output-dir data/raw/roboflow_players/rf06 --archive-name yolov8.zip
python scripts/download_and_extract_zip.py --url "<RF_LINK_7_FRESH>" --output-dir data/raw/roboflow_players/rf07 --archive-name yolov8.zip
```

## 3) Build path-aware YAML only for rf05-07

```bash
python - <<'PY'
from pathlib import Path
import yaml

for key in ["rf05", "rf06", "rf07"]:
    root = Path("data/raw/roboflow_players") / key
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
    }

    p = root / "source_with_path.yaml"
    p.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print("wrote", p)
PY
```

## 4) Check class names in rf05-07

```bash
for i in 05 06 07; do
  echo "===== rf$i"
  sed -n '1,120p' data/raw/roboflow_players/rf$i/data.yaml
  echo
done
```

## 5) Filter rf05-07 to player-only detect datasets

If class is literally `player`:

```bash
python scripts/filter_yolo_classes.py --data data/raw/roboflow_players/rf05/source_with_path.yaml --output-dir data/prepared/roboflow_players/rf05_player_only --dataset-name rf05_player_only --keep-classes player
python scripts/filter_yolo_classes.py --data data/raw/roboflow_players/rf06/source_with_path.yaml --output-dir data/prepared/roboflow_players/rf06_player_only --dataset-name rf06_player_only --keep-classes player
python scripts/filter_yolo_classes.py --data data/raw/roboflow_players/rf07/source_with_path.yaml --output-dir data/prepared/roboflow_players/rf07_player_only --dataset-name rf07_player_only --keep-classes player
```

If any dataset is pose-style (extra columns), rerun that dataset with `--truncate-values 5`.

Example:

```bash
python scripts/filter_yolo_classes.py --data data/raw/roboflow_players/rfXX/source_with_path.yaml --output-dir data/prepared/roboflow_players/rfXX_player_only_det --dataset-name rfXX_player_only_det --keep-classes player --truncate-values 5
```

## 6) Validate new datasets (rf05-07)

```bash
python scripts/validate_yolo_dataset.py --data data/prepared/roboflow_players/rf05_player_only/rf05_player_only.yaml --task detect
python scripts/validate_yolo_dataset.py --data data/prepared/roboflow_players/rf06_player_only/rf06_player_only.yaml --task detect
python scripts/validate_yolo_dataset.py --data data/prepared/roboflow_players/rf07_player_only/rf07_player_only.yaml --task detect
```

## 7) Rebuild RF-only merge including old + new

Use rf01..rf04 existing plus rf05..rf07 new.

```bash
python scripts/merge_yolo_datasets.py \
  --inputs \
    data/prepared/roboflow_players/rf01_player_only \
    data/prepared/roboflow_players/rf02_player_only_det \
    data/prepared/roboflow_players/rf03_player_only \
    data/prepared/roboflow_players/rf04_player_only \
    data/prepared/roboflow_players/rf05_player_only \
    data/prepared/roboflow_players/rf06_player_only \
    data/prepared/roboflow_players/rf07_player_only \
  --output-dir data/merged_players_rf_v2 \
  --dataset-name merged_players_rf_v2 \
  --task detect \
  --class-names player

python scripts/validate_yolo_dataset.py --data data/merged_players_rf_v2/merged_players_rf_v2.yaml --task detect
```

## 8) Rebuild Zenodo + RF merge (fast)

```bash
python - <<'PY'
from pathlib import Path
import shutil
import yaml

inputs=[Path('data/prepared/zenodo_players_detection_f1'), Path('data/merged_players_rf_v2')]
out=Path('data/merged_players_zenodo_rf_v2_fast')

for split in ['train','val','test']:
    (out/split/'images').mkdir(parents=True, exist_ok=True)
    (out/split/'labels').mkdir(parents=True, exist_ok=True)

counts={s:0 for s in ['train','val','test']}
for root in inputs:
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

yaml_path=out/'merged_players_zenodo_rf_v2_fast.yaml'
yaml_content={
    'path': str(out.resolve()),
    'train':'train/images',
    'val':'val/images',
    'test':'test/images',
    'names':{0:'player'},
}
yaml_path.write_text(yaml.safe_dump(yaml_content,sort_keys=False),encoding='utf-8')
print('Merged dataset written to:', out.resolve())
print('Dataset YAML written to:', yaml_path.resolve())
for s in ['train','val','test']:
    print(f"{s}: {counts[s]} images")
PY

python scripts/validate_yolo_dataset.py --data data/merged_players_zenodo_rf_v2_fast/merged_players_zenodo_rf_v2_fast.yaml --task detect
```

## 9) Training commands (when you decide to train)

Zenodo baseline:

```bash
nohup python scripts/train_yolo.py \
  --preset players-detection \
  --model yolov8m.pt \
  --data data/prepared/zenodo_players_detection_f1/zenodo_players_detection_f1.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name players_detection_zenodo_comparator_v2 \
  > logs/train_players_detection_zenodo_comparator_v2.log 2>&1 &
```

RF-only:

```bash
nohup python scripts/train_yolo.py \
  --preset players-detection \
  --model yolov8m.pt \
  --data data/merged_players_rf_v2/merged_players_rf_v2.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name players_detection_rf_v2 \
  > logs/train_players_detection_rf_v2.log 2>&1 &
```

Zenodo+RF:

```bash
nohup python scripts/train_yolo.py \
  --preset players-detection \
  --model yolov8m.pt \
  --data data/merged_players_zenodo_rf_v2_fast/merged_players_zenodo_rf_v2_fast.yaml \
  --epochs 100 --batch 160 --imgsz 640 --device 0 --workers 4 \
  --name players_detection_zenodo_rf_v2_fast \
  > logs/train_players_detection_zenodo_rf_v2_fast.log 2>&1 &
```

