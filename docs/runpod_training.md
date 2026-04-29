# RunPod Training Setup

This is the full training path for using this repo on RunPod.

## Scope

This setup is complete for the three YOLO-based tasks already present in the repo:

- players detection
- players keypoints
- court keypoints

The current recommended court baseline is the reduced non-cage schema. It keeps the core court and service/net points and drops the `cage_*` labels that were hurting overall quality.

## 1. Launch a pod

Recommended:

- choose a PyTorch or CUDA-enabled RunPod image
- attach enough disk for datasets and checkpoints
- use a GPU with more VRAM than the GTX 1060 when possible

## 2. Clone the repo

```bash
git clone <your-repo-url>
cd padel_analytics
```

## 3. Set up the training environment

```bash
bash scripts/setup_runpod_training.sh
```

This script:

- creates `.venv`
- installs CUDA-enabled PyTorch
- installs training dependencies from `requirements-training.txt`
- verifies the GPU environment

Manual verification:

```bash
source .venv/bin/activate
python scripts/check_training_env.py
```

You want:

- `torch cuda available: True`
- at least one CUDA device listed

## 4. Download data

### Option A: Roboflow ZIP or export link

If Roboflow gives you a download URL to a ZIP:

```bash
source .venv/bin/activate
python scripts/download_and_extract_zip.py \
  --url "<roboflow-zip-url>" \
  --output-dir data/raw/roboflow_players_detection \
  --archive-name roboflow_players_detection.zip
```

Then normalize it:

```bash
python scripts/prepare_yolo_dataset.py \
  --source-dir data/raw/roboflow_players_detection \
  --output-dir data/players_detection \
  --task detect \
  --dataset-name players_detection
```

### Option B: Zenodo ZIP or archive link

If Zenodo provides a ZIP archive:

```bash
python scripts/download_and_extract_zip.py \
  --url "<zenodo-zip-url>" \
  --output-dir data/raw/zenodo_source \
  --archive-name zenodo_source.zip
```

If it is not already YOLO format, convert it first, then normalize it.

For COCO-style Zenodo annotations tied to a video, use the sequential converter.
It reads the video once from start to finish and writes only annotated frames.
This avoids slow random seeks and avoids storing every extracted frame.

Players pose example:

```bash
python scripts/convert_coco_video_to_yolo.py \
  --json data/raw/zenodo/padel-data-labels/labels/2022_BCN_FinalF_1_pose.json \
  --video data/raw/zenodo/padel-data-labels/2022_BCN_FinalF_1.mp4 \
  --output-dir data/prepared/zenodo_players_pose_f1 \
  --dataset-name zenodo_players_pose_f1 \
  --task pose \
  --include-category-ids 1 \
  --class-names player \
  --flip-idx 0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15 \
  --log-every 1000
```

Ball detection example:

```bash
python scripts/convert_coco_video_to_yolo.py \
  --json data/raw/zenodo/padel-data-labels/labels/2022_BCN_FinalF_1_ball.json \
  --video data/raw/zenodo/padel-data-labels/2022_BCN_FinalF_1.mp4 \
  --output-dir data/prepared/zenodo_ball_f1 \
  --dataset-name zenodo_ball_f1 \
  --task detect \
  --include-category-ids 1 \
  --class-names ball \
  --log-every 1000
```

### Option C: Client CVAT XMLs

Detection:

```bash
python scripts/convert_cvat_xml_to_yolo_detection.py \
  --xml /workspace/client/annotations.xml \
  --images-dir /workspace/client/images \
  --output-dir data/client_players_detection \
  --dataset-name client_players_detection \
  --class-names player
```

Pose:

```bash
python scripts/convert_cvat_xml_to_yolo_pose.py \
  --xml /workspace/client/pose_annotations.xml \
  --images-dir /workspace/client/images \
  --output-dir data/client_players_keypoints \
  --dataset-name client_players_keypoints \
  --keypoint-names left_foot,right_foot,torso,right_shoulder,left_shoulder,head,neck,left_hand,right_hand,right_knee,left_knee,right_elbow,left_elbow \
  --flip-idx 1,0,2,4,3,5,6,8,7,10,9,12,11
```

### Option D: Filter older court datasets into the non-cage schema

If you already have a 26-class court dataset, filter out all `cage_*` classes before merging:

```bash
python scripts/filter_yolo_classes.py \
  --data data/court_keypoints/court_keypoints.yaml \
  --output-dir data/filtered/court_keypoints_no_cage \
  --dataset-name court_keypoints_no_cage \
  --drop-prefixes cage_
```

## 5. Validate data before training

Detection:

```bash
python scripts/validate_yolo_dataset.py \
  --data data/players_detection/players_detection.yaml \
  --task detect
```

Pose:

```bash
python scripts/validate_yolo_dataset.py \
  --data data/players_keypoints/players_keypoints.yaml \
  --task pose
```

Zenodo pose validation example:

```bash
python scripts/validate_yolo_dataset.py \
  --data data/prepared/zenodo_players_pose_f1/zenodo_players_pose_f1.yaml \
  --task pose
```

## 6. Merge datasets if needed

Detection example:

```bash
python scripts/merge_yolo_datasets.py \
  --inputs data/players_detection data/client_players_detection \
  --output-dir data/merged_players_detection \
  --dataset-name merged_players_detection \
  --task detect \
  --class-names player
```

Pose example:

```bash
python scripts/merge_yolo_datasets.py \
  --inputs data/players_keypoints data/client_players_keypoints \
  --output-dir data/merged_players_keypoints \
  --dataset-name merged_players_keypoints \
  --task pose \
  --class-names player \
  --kpt-shape 13,3 \
  --flip-idx 1,0,2,4,3,5,6,8,7,10,9,12,11
```

## 7. Train

Activate the environment:

```bash
source .venv/bin/activate
```

Players detection:

```bash
python scripts/train_yolo.py \
  --preset players-detection \
  --data data/merged_players_detection/merged_players_detection.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --workers 4
```

Players keypoints:

```bash
python scripts/train_yolo.py \
  --preset players-keypoints \
  --data data/merged_players_keypoints/merged_players_keypoints.yaml \
  --epochs 100 \
  --batch 8 \
  --device 0 \
  --workers 4
```

Court keypoints:

```bash
python scripts/train_yolo.py \
  --preset court-keypoints \
  --data data/merged_court_keypoints_no_cage/merged_court_keypoints_no_cage.yaml \
  --epochs 100 \
  --batch 192 \
  --device 0 \
  --workers 8 \
  --name court_keypoints_no_cage_v1
```

## Data expectations

### Roboflow

Usually minimal preprocessing if exported in Ultralytics YOLO format.

Double-check:

- `valid` vs `val`
- keypoint order
- class names
- whether the dataset is detect or pose

### Zenodo

Often requires preprocessing.

Double-check:

- whether frames need to be extracted from video
- whether annotations are YOLO-ready
- whether the task is detect or pose
- whether keypoint names match this repo

For cost control on RunPod:

- process one video at a time
- prefer the sequential converter for video-backed COCO data
- it writes only annotated frames
- reruns can resume because completed outputs are skipped

### CVAT

Preprocessing is required.

Double-check:

- detection uses boxes
- pose uses grouped points or skeletons
- keypoint names are stable and ordered

## Common mistakes to avoid

- mixing keypoint orders across datasets
- mixing `valid` and `val` folder names without rewriting YAML
- using one class mapping in Roboflow and another in CVAT
- training pose with the wrong `kpt_shape`
- skipping validation before training

## Recommended first path

1. start with Roboflow players detection
2. validate it
3. train it
4. then merge in CVAT or Zenodo-derived data
5. repeat for players keypoints
