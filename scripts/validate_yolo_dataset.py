from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_line(line: str) -> list[float]:
    return [float(value) for value in line.strip().split()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO detect or pose dataset for common structural mistakes.",
    )
    parser.add_argument("--data", required=True, help="Path to the dataset YAML.")
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--strict-labels", action="store_true", help="Fail if any image is missing a label file.")
    return parser


def validate_split(root_path: Path, split_value: str, task: str, class_count: int, keypoint_count: int | None, strict_labels: bool) -> tuple[int, int]:
    images_dir = root_path / Path(split_value)
    labels_dir = images_dir.parent / "labels"

    if not images_dir.is_dir():
        return 0, 0

    image_files = [
        image_path
        for image_path in sorted(images_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ]

    missing_labels = 0
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            if strict_labels:
                raise FileNotFoundError(f"Missing label file for image: {image_path}")
            continue

        with open(label_path, "r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                values = parse_label_line(line)
                class_id = int(values[0])
                if class_id < 0 or class_id >= class_count:
                    raise ValueError(f"Invalid class id {class_id} in {label_path}:{line_number}")

                if task == "detect":
                    if len(values) != 5:
                        raise ValueError(f"Expected 5 values in detection label, got {len(values)} in {label_path}:{line_number}")
                else:
                    expected_length = 5 + (keypoint_count * 3)
                    if len(values) != expected_length:
                        raise ValueError(
                            f"Expected {expected_length} values in pose label, got {len(values)} in {label_path}:{line_number}"
                        )

    return len(image_files), missing_labels


def main() -> None:
    args = build_parser().parse_args()
    yaml_path = Path(args.data).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as file:
        data_yaml = yaml.safe_load(file) or {}

    root_path = Path(data_yaml["path"]).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root_path}")

    names = data_yaml.get("names", {})
    class_count = len(names)
    keypoint_count = None
    if args.task == "pose":
        kpt_shape = data_yaml.get("kpt_shape")
        if not kpt_shape:
            raise ValueError("Pose validation requires kpt_shape in the dataset YAML.")
        keypoint_count = int(kpt_shape[0])

    total_images = 0
    total_missing_labels = 0
    for split in ("train", "val", "test"):
        split_value = data_yaml.get(split)
        if not split_value:
            continue
        image_count, missing_labels = validate_split(
            root_path=root_path,
            split_value=split_value,
            task=args.task,
            class_count=class_count,
            keypoint_count=keypoint_count,
            strict_labels=args.strict_labels,
        )
        total_images += image_count
        total_missing_labels += missing_labels
        print(f"{split}: {image_count} images, {missing_labels} missing label files")

    print(f"total_images: {total_images}")
    print(f"total_missing_labels: {total_missing_labels}")


if __name__ == "__main__":
    main()
