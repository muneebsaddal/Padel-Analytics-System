# Training Workflow

This repo now includes a training-oriented workflow for the three YOLO-based models already used by inference:

- `players-detection`: YOLO object detection
- `players-keypoints`: YOLO pose with 13 keypoints
- `court-keypoints`: YOLO pose with one keypoint per class

The custom ball tracker is not included in this first training scaffold. It uses a separate TrackNet/InpaintNet pipeline and should be treated as a second phase.

## Recommended directory layout

Use one dataset root per task:

```text
data/
  players_detection/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
  players_keypoints/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
  court_keypoints/
    train/
      images/
      labels/
    val/
      images/
      labels/
    test/
      images/
      labels/
```

Dataset YAML templates live in:

- `datasets/players_detection_template.yaml`
- `datasets/players_keypoints_template.yaml`
- `datasets/court_keypoints_template.yaml`

Copy and edit the right YAML before training.

## Training commands

### Players detection

```powershell
python .\scripts\train_yolo.py --preset players-detection --data .\datasets\players_detection_template.yaml --epochs 100 --batch 8 --device cpu
```

### Players keypoints

```powershell
python .\scripts\train_yolo.py --preset players-keypoints --data .\datasets\players_keypoints_template.yaml --epochs 100 --batch 8 --device cpu
```

### Court keypoints

```powershell
python .\scripts\train_yolo.py --preset court-keypoints --data .\datasets\court_keypoints_template.yaml --epochs 100 --batch 8 --device cpu
```

On a CUDA machine, replace `--device cpu` with `--device 0`.

## Roboflow setup

For your case, Roboflow is the best first source because it can export directly in Ultralytics YOLO format.

Recommended workflow:

1. Create one Roboflow project per task.
2. Export in `YOLOv8` or `Ultralytics YOLO` format.
3. Unzip into a local folder under `data/`.
4. Point one of the dataset YAML files in `datasets/` at that folder.

Example local layout:

```text
data/
  roboflow_players_detection/
    train/
      images/
      labels/
    valid/
      images/
      labels/
    test/
      images/
      labels/
    data.yaml
```

Roboflow often names the validation split `valid` instead of `val`.
You can either:

1. Rename `valid` to `val`
2. Or edit the YAML so `val: valid/images`

Example detection YAML:

```yaml
path: C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/data/roboflow_players_detection
train: train/images
val: valid/images
test: test/images
names:
  0: player
```

Example first training command:

```powershell
python .\scripts\train_yolo.py --preset players-detection --data .\data\roboflow_players_detection\data.yaml --epochs 100 --batch 8 --device 0 --workers 2
```

For pose datasets, do one extra check before training:

1. Confirm the keypoint count matches this repo.
2. Confirm the keypoint order matches the YAML `flip_idx` assumptions.

Repo expectations:

- Players pose: 13 keypoints
- Court pose baseline: 18 non-cage classes with `kpt_shape: [1, 3]`
- Older 26-class court datasets can be filtered with `scripts/filter_yolo_classes.py`

If Roboflow keypoint names or ordering differ, the dataset must be remapped before merging it with CVAT or Zenodo-derived data.

## Roboflow preprocessing

If you export from Roboflow in Ultralytics YOLO format:

- Detection export: little to no preprocessing is needed.
- Pose export: little to no preprocessing is needed if the keypoint order already matches the YAML.

Usually you only need to:

1. Unzip the export into `data/<task_name>/`
2. Check `data.yaml`
3. Fix `path:` to point at your local dataset root
4. Verify class names and keypoint order

For the public Roboflow 6-sets workflow, this is the best first source to use because it already aligns with YOLO training.

## Zenodo preprocessing

Zenodo datasets such as PadelShot24 usually need preprocessing before training in this repo.

Typical work needed:

1. Extract frames if the source is video-based.
2. Convert annotations into YOLO detect or YOLO pose label files.
3. Build `train/val/test` splits.
4. Check class mapping.
5. Check keypoint name order against this repo.

For PadelShot24 specifically, expect preprocessing unless it already ships as Ultralytics-ready YOLO detect/pose folders.

If Zenodo ships in COCO, MOT, CSV, or custom JSON, you will need a format conversion step before training.

## CVAT XML preprocessing

Client CVAT XMLs do need preprocessing.

This repo now includes two converters:

- `scripts/convert_cvat_xml_to_yolo_detection.py`
- `scripts/convert_cvat_xml_to_yolo_pose.py`

### Detection conversion example

```powershell
python .\scripts\convert_cvat_xml_to_yolo_detection.py `
  --xml C:\data\client\annotations.xml `
  --images-dir C:\data\client\images `
  --output-dir C:\data\converted\players_detection `
  --dataset-name client_players `
  --class-names player
```

### Pose conversion example

Players keypoint order used by this repo:

```text
left_foot,right_foot,torso,right_shoulder,left_shoulder,head,neck,left_hand,right_hand,right_knee,left_knee,right_elbow,left_elbow
```

Example command:

```powershell
python .\scripts\convert_cvat_xml_to_yolo_pose.py `
  --xml C:\data\client\pose_annotations.xml `
  --images-dir C:\data\client\images `
  --output-dir C:\data\converted\players_keypoints `
  --dataset-name client_players_pose `
  --keypoint-names left_foot,right_foot,torso,right_shoulder,left_shoulder,head,neck,left_hand,right_hand,right_knee,left_knee,right_elbow,left_elbow `
  --flip-idx 1,0,2,4,3,5,6,8,7,10,9,12,11
```

For the current non-cage court baseline, use the exact class list from [datasets/court_keypoints_template.yaml](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/datasets/court_keypoints_template.yaml).

Important CVAT notes:

- Detection XML must contain `<box>` annotations.
- Pose XML should contain either grouped `<points>` annotations or `<skeleton>` annotations.
- If no bounding boxes exist for pose, the converter derives them from the visible keypoints.
- You must keep keypoint names and their order consistent across all datasets you plan to merge.

## Merging multiple datasets

After converting or downloading multiple datasets for the same task, merge them into one YOLO dataset:

```powershell
python .\scripts\merge_yolo_datasets.py `
  --inputs C:\data\rf_players C:\data\client_players `
  --output-dir C:\data\merged_players `
  --dataset-name merged_players `
  --task detect `
  --class-names player
```

## Court schema filtering

If you have older 26-class court datasets and want to train the stronger non-cage baseline, filter out all `cage_*` classes first:

```powershell
python .\scripts\filter_yolo_classes.py `
  --data C:\data\court_keypoints\court_keypoints.yaml `
  --output-dir C:\data\court_keypoints_no_cage `
  --dataset-name court_keypoints_no_cage `
  --drop-prefixes cage_
```

This keeps class ids aligned after filtering so the reduced-schema dataset can be merged and trained normally.

For pose:

```powershell
python .\scripts\merge_yolo_datasets.py `
  --inputs C:\data\rf_players_pose C:\data\client_players_pose `
  --output-dir C:\data\merged_players_pose `
  --dataset-name merged_players_pose `
  --task pose `
  --class-names player `
  --kpt-shape 13,3 `
  --flip-idx 1,0,2,4,3,5,6,8,7,10,9,12,11
```

## Recommended next step

Start with this order:

1. `players-detection`
2. `players-keypoints`
3. `court-keypoints`
4. ball model later

That gets you the fastest path to measurable model improvement using Roboflow first, then Zenodo and CVAT once converted into the same training format.

If you plan to train on RunPod, use [docs/runpod_training.md](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/docs/runpod_training.md) as the main end-to-end setup guide.

## GPU setup on your machine

Your current virtualenv is using a CPU-only PyTorch build, so training is not reaching the GTX 1060 yet.

You can verify the environment with:

```powershell
python .\scripts\check_training_env.py
```

Based on the current official PyTorch install page for Windows pip installs, CUDA wheels are available for multiple CUDA versions, including CUDA 11.8 and newer. For a GTX 1060, CUDA 11.8 is a safe choice. Source: https://pytorch.org/get-started/locally

Recommended install command inside `padelenv`:

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then verify:

```powershell
python .\scripts\check_training_env.py
```

You want to see:

- `torch cuda available: True`
- `cuda:0: NVIDIA GeForce GTX 1060 ...`

After that, train with:

```powershell
python .\scripts\train_yolo.py --preset players-detection --data .\data\roboflow_players_detection\data.yaml --epochs 100 --batch 8 --device 0 --workers 2
```

GTX 1060 practical notes:

- 6 GB VRAM is enough for training, but keep batch sizes modest.
- Good starting points:
  - detection at `imgsz 640`, `batch 8`
  - pose at `imgsz 640` or `imgsz 960`, `batch 2` to `4`
- If you hit out-of-memory, reduce `--batch` first, then reduce `--imgsz`.
