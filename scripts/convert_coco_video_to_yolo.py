from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml


FRAME_PATTERN = re.compile(r"(\d+)")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    if total == 0:
        return "train"
    threshold_train = int(total * train_ratio)
    threshold_val = int(total * (train_ratio + val_ratio))
    if index < threshold_train:
        return "train"
    if index < threshold_val:
        return "val"
    return "test"


def frame_index_from_name(file_name: str, fallback_index: int) -> int:
    matches = FRAME_PATTERN.findall(Path(file_name).stem)
    if not matches:
        return fallback_index
    return int(matches[-1])


def coco_bbox_to_yolo(
    bbox: list[float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x_min, y_min, box_width, box_height = bbox
    center_x = (x_min + box_width / 2.0) / width
    center_y = (y_min + box_height / 2.0) / height
    return center_x, center_y, box_width / width, box_height / height


def normalize_keypoints(
    keypoints: list[float],
    width: float,
    height: float,
) -> list[str]:
    output: list[str] = []
    for idx in range(0, len(keypoints), 3):
        x_coord = keypoints[idx]
        y_coord = keypoints[idx + 1]
        visibility = int(keypoints[idx + 2])
        if visibility <= 0:
            output.extend(["0.000000", "0.000000", "0"])
        else:
            output.extend(
                [
                    f"{x_coord / width:.6f}",
                    f"{y_coord / height:.6f}",
                    str(min(visibility, 2)),
                ]
            )
    return output


def build_label_lines(
    image_annotations: list[dict],
    task: str,
    kpt_shape: list[int] | None,
    width: float,
    height: float,
    category_to_class_id: dict[int, int],
) -> list[str]:
    label_lines: list[str] = []
    for annotation in image_annotations:
        class_id = category_to_class_id[annotation["category_id"]]
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        center_x, center_y, box_width, box_height = coco_bbox_to_yolo(
            bbox, width, height
        )

        if task == "detect":
            label_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            )
            continue

        keypoints = annotation.get("keypoints", [])
        expected_keypoint_values = kpt_shape[0] * 3
        if len(keypoints) != expected_keypoint_values:
            continue

        normalized_keypoints = normalize_keypoints(keypoints, width, height)
        label_lines.append(
            " ".join(
                [
                    str(class_id),
                    f"{center_x:.6f}",
                    f"{center_y:.6f}",
                    f"{box_width:.6f}",
                    f"{box_height:.6f}",
                    *normalized_keypoints,
                ]
            )
        )

    return label_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert COCO-style annotations backed by a source video into YOLO detect or pose datasets using a single sequential pass.",
    )
    parser.add_argument("--json", required=True, help="Path to COCO-style annotation JSON.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the source video.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument(
        "--include-category-ids",
        default="",
        help="Optional comma-separated COCO category ids to keep.",
    )
    parser.add_argument(
        "--class-names",
        default="",
        help="Optional comma-separated YOLO class names.",
    )
    parser.add_argument(
        "--flip-idx",
        default="",
        help="Pose only, comma-separated flip indices.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-extension",
        default=".png",
        help="Output image extension, e.g. .png or .jpg",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete the output directory before conversion.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N decoded frames.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    json_path = Path(args.json).resolve()
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not json_path.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {json_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    if args.force_rebuild and output_dir.exists():
        shutil.rmtree(output_dir)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    include_category_ids = (
        set(parse_int_csv(args.include_category_ids))
        if args.include_category_ids
        else None
    )
    categories_by_id = {category["id"]: category for category in categories}

    if include_category_ids is None:
        kept_category_ids = sorted(categories_by_id.keys())
    else:
        kept_category_ids = sorted(
            category_id
            for category_id in include_category_ids
            if category_id in categories_by_id
        )

    if not kept_category_ids:
        raise ValueError("No valid category ids selected for conversion.")

    if args.class_names:
        class_names = parse_csv(args.class_names)
    else:
        class_names = [
            categories_by_id[category_id]["name"]
            for category_id in kept_category_ids
        ]

    category_to_class_id = {
        category_id: class_id
        for class_id, category_id in enumerate(kept_category_ids)
    }

    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id not in category_to_class_id:
            continue
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    selected_images = [image for image in images if annotations_by_image_id.get(image["id"])]
    random.Random(args.seed).shuffle(selected_images)

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    kpt_shape = None
    if args.task == "pose":
        first_category = categories_by_id[kept_category_ids[0]]
        keypoints_names = first_category.get("keypoints", [])
        if not keypoints_names:
            raise ValueError(
                "Pose conversion requires category keypoints in the COCO JSON."
            )
        kpt_shape = [len(keypoints_names), 3]

    # Precompute frame targets and output paths once so the video can be read sequentially.
    targets_by_frame: dict[int, dict] = {}
    for index, image in enumerate(selected_images):
        image_id = image["id"]
        width = float(image["width"])
        height = float(image["height"])
        label_lines = build_label_lines(
            image_annotations=annotations_by_image_id[image_id],
            task=args.task,
            kpt_shape=kpt_shape,
            width=width,
            height=height,
            category_to_class_id=category_to_class_id,
        )

        if not label_lines:
            continue

        split = split_name(index, len(selected_images), args.train_ratio, args.val_ratio)
        output_stem = Path(image["file_name"]).stem
        output_image_path = output_dir / split / "images" / f"{output_stem}{args.image_extension}"
        output_label_path = output_dir / split / "labels" / f"{output_stem}.txt"
        frame_index = frame_index_from_name(image["file_name"], fallback_index=index)

        targets_by_frame[frame_index] = {
            "label_lines": label_lines,
            "label_path": output_label_path,
            "output_image_path": output_image_path,
        }

    if not targets_by_frame:
        raise ValueError("No valid annotated frames were found for conversion.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_target_frame = max(targets_by_frame.keys())

    written_images = 0
    skipped_existing = 0
    current_frame = 0

    while current_frame <= max_target_frame:
        success, frame = capture.read()
        if not success:
            break

        target = targets_by_frame.get(current_frame)
        if target is not None:
            output_image_path = target["output_image_path"]
            output_label_path = target["label_path"]

            if output_image_path.exists() and output_label_path.exists():
                skipped_existing += 1
            else:
                if args.image_extension.lower() in (".jpg", ".jpeg"):
                    cv2.imwrite(
                        str(output_image_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                else:
                    cv2.imwrite(str(output_image_path), frame)
                output_label_path.write_text(
                    "\n".join(target["label_lines"]) + "\n",
                    encoding="utf-8",
                )
                written_images += 1

        current_frame += 1
        if args.log_every > 0 and current_frame % args.log_every == 0:
            if total_frames > 0:
                print(
                    f"Processed frame {current_frame}/{min(total_frames, max_target_frame + 1)}"
                )
            else:
                print(f"Processed frame {current_frame}")

    capture.release()

    yaml_content = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if args.task == "pose":
        yaml_content["kpt_shape"] = kpt_shape
        yaml_content["flip_idx"] = parse_int_csv(args.flip_idx)

    yaml_path = output_dir / f"{args.dataset_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False)

    print(f"New files written: {written_images}")
    print(f"Existing files skipped: {skipped_existing}")
    print(f"Total target frames: {len(targets_by_frame)}")
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
