# Padel Analytics System

End-to-end computer vision system for padel video analysis, model training, evaluation, annotation handoff, and lightweight deployment packaging.

## Project Overview

This repository documents a full applied CV workflow built around padel match footage. The work focused on improving model quality across four tasks while also making the surrounding tooling production-friendly:

- player detection
- player pose estimation
- ball detection
- court keypoint estimation

The repo is structured as a working training and inference lab rather than a single notebook or demo. It includes dataset conversion and filtering utilities, training entrypoints, evaluation scripts, CVAT-oriented review tools, and deployable model packaging.

## Highlights

- Built a unified YOLO-based workflow for training and comparing player, ball, and court models
- Merged Zenodo and Roboflow sources into cleaner task-specific datasets
- Improved court keypoint quality by removing noisy `cage_*` labels and retraining on a cleaner schema
- Improved inference stability with court-aware filtering, temporal ball smoothing, anti-jump logic, and pose-to-player consistency checks
- Added benchmark clip evaluation utilities for comparing predictions against manual annotations
- Added annotation handoff tooling for CVAT review and correction loops
- Packaged the latest deployable model set in `release_models/`

## System Capabilities

### Training Pipeline

- dataset conversion to YOLO format
- class filtering and label cleanup
- dataset validation and merge utilities
- task-specific YOLO training presets

### Inference Pipeline

- combined multi-model video overlay runner
- batch clip inference support
- optional CVAT XML export
- deployable runtime bundle preparation

### Evaluation And Annotation

- clip trimming for focused benchmarks
- evaluation against manual Zenodo annotations
- preview rendering for pose datasets
- CVAT-ready frame and XML generation

## Repo Structure

- `scripts/` training, inference, evaluation, packaging, and utility scripts
- `trackers/` task-level tracking and model orchestration modules
- `analytics/` analytics and projected-court helpers
- `docs/` runbooks, tutorials, and handoff notes
- `release_models/` latest packaged model weights intended for delivery

## Main Deliverables

- a reusable data-prep and training workflow for padel CV models
- final packaged weights for the strongest current models
- a combined overlay inference runner for full-video analysis
- supporting tools for annotation review, benchmarking, and deployment handoff

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- YAML-based dataset configs
- CVAT-compatible export tooling
- RunPod-based training workflow

## Current State

The strongest results in this repo are player detection and player pose, with improved court keypoints and a substantially better ball pipeline than the original baseline. The codebase is organized so future iterations can continue from inference review back into annotation and retraining.
