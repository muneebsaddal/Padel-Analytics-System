from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, sort_keys=False)


def merge_split(input_root: Path, output_root: Path, split: str) -> int:
    input_images = input_root / split / "images"
    input_labels = input_root / split / "labels"
    if not input_images.is_dir():
        return 0

    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in sorted(input_images.iterdir()):
        if not image_path.is_file():
            continue

        stem = f"{input_root.name}_{image_path.stem}"
        new_image_path = output_images / f"{stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, new_image_path)

        label_path = input_labels / f"{image_path.stem}.txt"
        new_label_path = output_labels / f"{stem}.txt"
        if label_path.is_file():
            shutil.copy2(label_path, new_label_path)
        else:
            new_label_path.write_text("", encoding="utf-8")
        copied += 1

    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO-format datasets into one train/val/test dataset.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input YOLO dataset roots. Each root should contain train/val/test folders.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--class-names", required=True, help="Comma-separated class names.")
    parser.add_argument(
        "--kpt-shape",
        help='Required for pose datasets, e.g. "13,3" or "12,3".',
    )
    parser.add_argument(
        "--flip-idx",
        default="",
        help="Comma-separated keypoint flip indices for pose datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_roots = [Path(path).resolve() for path in args.inputs]
    for input_root in input_roots:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {split: 0 for split in SPLITS}
    for input_root in input_roots:
        for split in SPLITS:
            split_counts[split] += merge_split(input_root, output_dir, split)

    class_names = parse_csv(args.class_names)
    yaml_content = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }

    if args.task == "pose":
        if not args.kpt_shape:
            raise ValueError("--kpt-shape is required when task=pose")
        kpt_shape = [int(value) for value in parse_csv(args.kpt_shape)]
        yaml_content["kpt_shape"] = kpt_shape
        yaml_content["flip_idx"] = [int(value) for value in parse_csv(args.flip_idx)]

    yaml_path = output_dir / f"{args.dataset_name}.yaml"
    write_yaml(yaml_path, yaml_content)

    print(f"Merged dataset written to: {output_dir}")
    print(f"Dataset YAML written to: {yaml_path}")
    for split, count in split_counts.items():
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
