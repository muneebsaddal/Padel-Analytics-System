# Ball Data Plan

This is the next data-improvement step after the court baseline reached a strong non-cage result.

## Goal

Improve ball detection on real padel videos, especially:

- glass reflections mistaken for the ball
- shoe or clothing highlights mistaken for the ball
- tiny far-court ball misses
- motion-blurred ball misses

## Recommended source order

Start with this order:

1. Zenodo `padel-data-labels.zip`
2. one large Roboflow padel-ball source
3. one smaller diversity source
4. benchmark-driven cleanup after overlay review

## Primary sources

### 1. Zenodo ball labels

Use the existing Zenodo conversion path first because it already matches the current project workflow and produced the baseline ball model.

### 2. `padel dataset` by `padel-vyxal`

- link: `https://universe.roboflow.com/padel-vyxal/padel-dataset-6ywe6`
- size seen in Roboflow search results: about `9.2k` images
- classes: `ball`, `player`

Why use it:

- it is the largest clearly ball-relevant padel source we verified
- it appears aimed at ball position diversity across the court

Risk:

- it is multi-class, so ball labels should be inspected before merging
- deduplication against other Roboflow copies may be needed

### 3. `Padel Ball Detector`

- link: `https://universe.roboflow.com/padel/padel-ball-detector`
- size seen in Roboflow search results: about `1.4k` images
- classes: `Tennis Ball`

Why use it:

- single-class ball focus
- useful as a smaller diversity source after the larger dataset

### 4. User-provided Roboflow dataset

- link: `https://universe.roboflow.com/rengos-workspace/my-first-project-bejal`

Why use it:

- this is already on your shortlist

Risk:

- I have not yet verified its exact class list and image count from public search indexing
- inspect it before merging

## Lower-priority sources

These are worth checking, but only after the primary sources:

- `https://universe.roboflow.com/padelsense/padel-dataset-j1pcr`
- `https://universe.roboflow.com/padeltracking/padel-aszsm`
- `https://universe.roboflow.com/kk-workspace/tracking-padel-ball`

Reasons they are lower priority:

- smaller size
- possible duplication with the `padel dataset` family
- unclear label consistency

## Merge strategy

Use a staged merge instead of combining everything at once.

### Stage 1

- Zenodo ball dataset
- `padel-vyxal/padel-dataset-6ywe6`

### Stage 2

Add one of:

- `padel/padel-ball-detector`
- `rengos-workspace/my-first-project-bejal`

### Stage 3

Keep only the additions that improve the benchmark or overlay review.

Do not assume more images always help. Reflection-heavy false positives can get worse if the extra dataset has looser labels.

## Practical checks before merging

For every new ball dataset:

1. confirm the class list
2. keep only the ball class if the dataset is multi-class
3. inspect at least 50 random labels
4. check whether reflections are labeled as ball
5. check whether shoe highlights are mislabeled
6. validate the YOLO dataset before merging

## Repo tools to use

- download: [scripts/download_and_extract_zip.py](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/download_and_extract_zip.py)
- normalize: [scripts/prepare_yolo_dataset.py](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/prepare_yolo_dataset.py)
- filter classes: [scripts/filter_yolo_classes.py](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/filter_yolo_classes.py)
- merge: [scripts/merge_yolo_datasets.py](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/merge_yolo_datasets.py)
- validate: [scripts/validate_yolo_dataset.py](C:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/validate_yolo_dataset.py)

## Suggested next action

Build a first merged ball dataset from:

1. the current Zenodo ball baseline source
2. `padel-vyxal/padel-dataset-6ywe6`

Then inspect overlay behavior before adding more sources.
