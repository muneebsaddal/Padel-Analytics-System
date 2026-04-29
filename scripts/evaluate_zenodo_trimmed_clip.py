from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from convert_coco_video_to_yolo import build_label_lines, frame_index_from_name, parse_int_csv


def parse_time_to_seconds(value: str) -> float:
    text = value.strip()
    if ":" not in text:
        return float(text)

    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return (minutes * 60.0) + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (hours * 3600.0) + (minutes * 60.0) + seconds
    raise ValueError(f"Unsupported time format: {value}")


def require_file(path_str: str, label: str) -> Path:
    path = Path(path_str).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate current models on a trimmed Zenodo clip using the original manual annotations.",
    )
    parser.add_argument("--video", required=True, help="Original full Zenodo video path.")
    parser.add_argument("--start", required=True, help='Clip start in seconds or "MM:SS" or "HH:MM:SS".')
    parser.add_argument("--end", help='Clip end in seconds or "MM:SS" or "HH:MM:SS".')
    parser.add_argument("--duration", help='Clip duration in seconds or "MM:SS" or "HH:MM:SS".')
    parser.add_argument("--players-json", required=True, help="Zenodo pose/player COCO JSON.")
    parser.add_argument("--ball-json", required=True, help="Zenodo ball COCO JSON.")
    parser.add_argument("--players-detection-model", required=True)
    parser.add_argument("--players-pose-model", required=True)
    parser.add_argument("--ball-model", required=True)
    parser.add_argument("--output-dir", default="artifacts/eval/zenodo_trimmed_clip")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280, help="Ball validation image size.")
    parser.add_argument("--players-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser


def build_clip_dataset(
    *,
    json_path: Path,
    video_path: Path,
    output_dir: Path,
    task: str,
    include_category_ids: list[int],
    class_names: list[str],
    flip_idx: list[int] | None,
    start_frame: int,
    end_frame: int,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    categories_by_id = {category["id"]: category for category in categories}
    category_to_class_id = {
        category_id: class_id
        for class_id, category_id in enumerate(include_category_ids)
    }

    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id not in category_to_class_id:
            continue
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    kpt_shape = None
    if task == "pose":
        first_category = categories_by_id[include_category_ids[0]]
        keypoints_names = first_category.get("keypoints", [])
        if not keypoints_names:
            raise ValueError(f"Pose conversion requires keypoints in {json_path}")
        kpt_shape = [len(keypoints_names), 3]

    targets_by_frame: dict[int, dict] = {}
    for fallback_index, image in enumerate(images):
        frame_index = frame_index_from_name(image["file_name"], fallback_index=fallback_index)
        if frame_index < start_frame or frame_index >= end_frame:
            continue

        image_annotations = annotations_by_image_id.get(image["id"], [])
        if not image_annotations:
            continue

        label_lines = build_label_lines(
            image_annotations=image_annotations,
            task=task,
            kpt_shape=kpt_shape,
            width=float(image["width"]),
            height=float(image["height"]),
            category_to_class_id=category_to_class_id,
        )
        if not label_lines:
            continue

        clip_frame_index = frame_index - start_frame
        output_stem = f"frame_{clip_frame_index:06d}"
        targets_by_frame[frame_index] = {
            "output_image_path": output_dir / "val" / "images" / f"{output_stem}.png",
            "label_path": output_dir / "val" / "labels" / f"{output_stem}.txt",
            "label_lines": label_lines,
        }

    if not targets_by_frame:
        raise ValueError(f"No annotated frames found in requested clip range for {json_path.name}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    written = 0
    while current_frame < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        target = targets_by_frame.get(current_frame)
        if target is not None:
            cv2.imwrite(str(target["output_image_path"]), frame)
            target["label_path"].write_text("\n".join(target["label_lines"]) + "\n", encoding="utf-8")
            written += 1
        current_frame += 1

    capture.release()

    yaml_content = {
        "path": str(output_dir),
        "train": "val/images",
        "val": "val/images",
        "test": "val/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if task == "pose":
        yaml_content["kpt_shape"] = kpt_shape
        yaml_content["flip_idx"] = flip_idx or []

    yaml_path = output_dir / f"{output_dir.name}.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False)

    print(f"Prepared clip dataset: {output_dir}")
    print(f"Annotated frames written: {written}")
    print(f"Dataset YAML written to: {yaml_path}")
    return yaml_path


def extract_metrics(metrics) -> dict[str, float | None]:
    results = getattr(metrics, "results_dict", {}) or {}
    output: dict[str, float | None] = {}
    for key, value in results.items():
        if isinstance(value, (int, float)):
            output[str(key)] = float(value)
    return output


def main() -> None:
    args = build_parser().parse_args()

    video_path = require_file(args.video, "Video")
    players_json_path = require_file(args.players_json, "Players JSON")
    ball_json_path = require_file(args.ball_json, "Ball JSON")
    players_detection_model_path = require_file(args.players_detection_model, "Players detection model")
    players_pose_model_path = require_file(args.players_pose_model, "Players pose model")
    ball_model_path = require_file(args.ball_model, "Ball model")

    start_seconds = parse_time_to_seconds(args.start)
    end_seconds = parse_time_to_seconds(args.end) if args.end else None
    duration_seconds = parse_time_to_seconds(args.duration) if args.duration else None
    if end_seconds is None and duration_seconds is None:
        raise ValueError("Provide either --end or --duration.")
    if end_seconds is not None and duration_seconds is not None:
        raise ValueError("Provide only one of --end or --duration.")
    if duration_seconds is not None:
        end_seconds = start_seconds + duration_seconds
    if end_seconds is None or end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time.")

    output_root = Path(args.output_dir).resolve()
    if args.force_rebuild and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)

    players_det_yaml = build_clip_dataset(
        json_path=players_json_path,
        video_path=video_path,
        output_dir=output_root / "players_detection",
        task="detect",
        include_category_ids=[1],
        class_names=["player"],
        flip_idx=None,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    players_pose_yaml = build_clip_dataset(
        json_path=players_json_path,
        video_path=video_path,
        output_dir=output_root / "players_pose",
        task="pose",
        include_category_ids=[1],
        class_names=["player"],
        flip_idx=parse_int_csv("0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15"),
        start_frame=start_frame,
        end_frame=end_frame,
    )
    ball_yaml = build_clip_dataset(
        json_path=ball_json_path,
        video_path=video_path,
        output_dir=output_root / "ball_detection",
        task="detect",
        include_category_ids=[1],
        class_names=["ball"],
        flip_idx=None,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    players_det_metrics = YOLO(str(players_detection_model_path)).val(
        data=str(players_det_yaml),
        split="val",
        device=args.device,
        conf=args.players_conf,
        verbose=False,
    )
    players_pose_metrics = YOLO(str(players_pose_model_path)).val(
        data=str(players_pose_yaml),
        split="val",
        device=args.device,
        conf=args.pose_conf,
        verbose=False,
    )
    ball_metrics = YOLO(str(ball_model_path)).val(
        data=str(ball_yaml),
        split="val",
        imgsz=args.imgsz,
        device=args.device,
        conf=args.ball_conf,
        verbose=False,
    )

    summary = {
        "clip": {
            "video": str(video_path),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "players_detection": extract_metrics(players_det_metrics),
        "players_pose": extract_metrics(players_pose_metrics),
        "ball_detection": extract_metrics(ball_metrics),
    }
    summary_path = output_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Clip evaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
