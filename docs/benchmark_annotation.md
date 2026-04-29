# Benchmark Annotation Guide

This document defines the first unseen benchmark annotation package for the current four-model baseline stack:

1. players detection
2. players pose
3. ball detection
4. court keypoints

Use this guide when preparing a benchmark clip for CVAT and when briefing a non-technical annotator.

## Goal

The benchmark should answer one question clearly:

How well do the current baselines perform on a real unseen fixed-camera padel rally video?

This benchmark is for evaluation first, not training first. Keep the benchmark video separate from all training and validation sources already used for Zenodo and Roboflow.

## Benchmark Video Requirements

Choose one clip with all of the following:

- unseen by the current training pipeline
- fixed-camera full-court view
- no cuts, replays, zooms, or camera switches
- at least 2 minutes and ideally no more than 5 minutes
- normal rally play with both teams visible for most of the clip
- reflections on glass are allowed and are actually useful because they expose a known failure mode

Avoid clips with:

- broadcast overlays covering the ball or players for long stretches
- large camera shake
- severe motion blur for most of the clip
- heavy occlusion for almost the entire video

Suggested package layout:

```text
benchmark/
  source_video/
    benchmark_clip.mp4
  extracted_frames/
  cvat_exports/
  notes/
    benchmark_notes.md
```

## Recommended CVAT Setup

Use three CVAT tasks against the same frame set:

1. `players_detection`
2. `players_pose`
3. `ball_and_court`

This keeps the UI simpler for the annotator while still covering the full benchmark.

If the annotator is comfortable with CVAT and the frame count is modest, `ball_and_court` can be split into two tasks:

1. `ball_detection`
2. `court_keypoints`

## Frame Strategy

Use one of these two strategies and keep it consistent for the full benchmark:

1. Annotate every frame if the clip is short enough and annotation time is acceptable.
2. Annotate every `n`th frame for the main benchmark and add dense extra frames around fast ball exchanges.

Recommended starting point:

- extract at 5 FPS for the main benchmark pass
- add extra frames around difficult ball segments if needed

Assumption:
This benchmark is primarily for model comparison and failure analysis, so a clean, consistent sampled benchmark is more valuable than an inconsistent full-frame annotation pass.

## Labels And Schema

### Task 1: Players detection

Use one box class only:

- `player`

Rules:

- annotate every real player fully or partially visible on court
- annotate players even if partially occluded by another player, net, or glass structure
- do not annotate reflections in glass
- do not annotate spectators, coaches, referees, or people outside the playable court area
- do not annotate posters, shadows, or silhouettes

Bounding box rules:

- fit the visible person as tightly as practical
- include the full visible body, including racket hand if it is clearly part of the player silhouette
- if only a small body fragment is visible and identity as a player is unclear, skip it

### Task 2: Players pose

Use one skeleton label:

- `player`

Use the standard 17-keypoint COCO order:

```text
nose,
left_eye,
right_eye,
left_ear,
right_ear,
left_shoulder,
right_shoulder,
left_elbow,
right_elbow,
left_wrist,
right_wrist,
left_hip,
right_hip,
left_knee,
right_knee,
left_ankle,
right_ankle
```

Rules:

- annotate each on-court player with one skeleton
- place keypoints only on the real player, never on reflections
- if a keypoint is fully occluded or impossible to localize, leave it absent or marked not visible in CVAT
- keep left and right from the player’s perspective, not from the camera’s perspective
- if the person is too small or blurred to place a reliable skeleton, skip the pose but still keep the detection box in the detection task

### Task 3: Ball detection

Use one box class only:

- `ball`

Rules:

- annotate the real ball whenever it is visible with reasonable confidence
- do not annotate likely reflections in glass
- do not annotate shoe highlights, logos, or bright clothing spots
- do not hallucinate the ball when it is fully invisible
- if the ball is motion-blurred, box the visible blur region tightly

Special instruction for known failure modes:

- if a bright round object is attached to a shoe or clothing, do not label it
- if a bright round object appears only in the glass reflection and not in the actual court space, do not label it
- if two candidate balls are visible, label only the physically plausible real ball in the playing area

### Task 4: Court keypoints

Use the exact same 26 court point names and ordering as the current trained court baseline.

Do not invent new court labels for the benchmark.
Do not rename labels to make them easier for annotation.
The benchmark must stay schema-compatible with the existing court model and downstream overlay code.

At minimum, confirm these four polygon anchor labels exist exactly with these names:

```text
court_bottom_left_far
court_bottom_right_far
court_bottom_right_close
court_bottom_left_close
```

Before annotation starts, the technical owner should create the CVAT point-label list by copying the exact 26 names from the court dataset YAML or from the trained model metadata used by the current baseline.

Rules:

- annotate the visible court point closest to the true painted-line or structure intersection for that label
- keep label names consistent across every frame
- if a point is fully invisible or impossible to locate, leave it absent
- do not place points on reflections
- do not move labels across semantically different court corners or net/cage points just to keep all labels filled

## Non-Technical Annotator Instructions

Share this short version with the annotator:

1. Label only real objects on the actual court, not reflections in glass.
2. Players: box every real player on court.
3. Player pose: add the skeleton only when body joints can be placed with confidence.
4. Ball: label the real ball only when it is actually visible.
5. Court points: use the exact point names already provided in the task and do not rename them.
6. If you are unsure, leave a note rather than guessing.

## QA Checklist

Review a sample of early annotations before the full task continues.

Check for:

- reflections labeled as players
- reflections labeled as balls
- shoe highlights labeled as balls
- left and right body joints swapped in player pose
- missing obvious players
- court labels drifting between frames
- inconsistent handling of occluded points

## Technical Export Notes

Recommended exports from CVAT:

- players detection as CVAT XML or COCO boxes
- players pose as CVAT XML skeletons or grouped points
- ball detection as CVAT XML or COCO boxes
- court keypoints as CVAT XML points or skeleton-style grouped points

Current repo conversion support:

- detection: [scripts/convert_cvat_xml_to_yolo_detection.py](/c:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/convert_cvat_xml_to_yolo_detection.py)
- pose: [scripts/convert_cvat_xml_to_yolo_pose.py](/c:/Users/munee/Documents/Code/Upwork-contracts/UP-Contract-6--Padel-Analytics-System/padel_analytics/scripts/convert_cvat_xml_to_yolo_pose.py)

Use separate exports per task unless the CVAT setup has been tested end to end with the current converters.

## Benchmark Notes Template

Create a short note file with:

- source URL or source file name
- clip duration
- frame extraction rate
- whether annotation is full-frame or sampled
- known hard segments
- whether reflections are strong, moderate, or low
- any uncertain court labels that need technical review

## Next Step After Annotation

Once the benchmark is annotated:

1. run the combined overlay on the same video
2. compare predictions against ground truth
3. group errors into reflections, small-ball misses, pose misses under occlusion, and court-point confusion
4. use those error groups to define the next data collection and labeling sprint
