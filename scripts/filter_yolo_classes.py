from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        pairs = sorted((int(key), str(value)) for key, value in raw_names.items())
        return [value for _, value in pairs]
    raise ValueError("Dataset YAML must contain names as a list or mapping.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter classes from a YOLO detect or pose dataset and rewrite class ids.",
    )
    parser.add_argument("--data", required=True, help="Path to the source dataset YAML.")
    parser.add_argument("--output-dir", required=True, help="Directory for the filtered dataset.")
    parser.add_argument("--dataset-name", required=True, help="Name for the rewritten dataset YAML.")
    parser.add_argument("--keep-classes", default="", help="Comma-separated class names to keep.")
    parser.add_argument("--drop-classes", default="", help="Comma-separated class names to drop.")
    parser.add_argument(
        "--drop-prefixes",
        default="",
        help='Comma-separated class-name prefixes to drop, e.g. "cage_".',
    )
    parser.add_argument(
        "--truncate-values",
        type=int,
        default=0,
        help="Optional output width for each label row. For example, use 5 to keep YOLO detection fields only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    yaml_path = Path(args.data).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    source_yaml = load_yaml(yaml_path)
    source_root = Path(source_yaml["path"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {source_root}")

    source_names = normalize_names(source_yaml.get("names", {}))
    keep_classes = set(parse_csv(args.keep_classes))
    drop_classes = set(parse_csv(args.drop_classes))
    drop_prefixes = tuple(parse_csv(args.drop_prefixes))

    if keep_classes and drop_classes:
        raise ValueError("Use either --keep-classes or --drop-classes, not both.")

    kept_source_ids: list[int] = []
    kept_names: list[str] = []
    for source_id, class_name in enumerate(source_names):
        keep = True
        if keep_classes:
            keep = class_name in keep_classes
        if class_name in drop_classes:
            keep = False
        if drop_prefixes and any(class_name.startswith(prefix) for prefix in drop_prefixes):
            keep = False
        if keep:
            kept_source_ids.append(source_id)
            kept_names.append(class_name)

    if not kept_names:
        raise ValueError("Filtering removed every class. Adjust keep/drop arguments.")

    class_id_map = {source_id: new_id for new_id, source_id in enumerate(kept_source_ids)}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    kept_annotations = 0
    for split in SPLITS:
        split_value = source_yaml.get(split)
        if not split_value:
            continue

        source_images = source_root / Path(split_value)
        source_labels = source_images.parent / "labels"
        target_images = output_dir / split / "images"
        target_labels = output_dir / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        if not source_images.is_dir():
            continue

        for image_path in sorted(source_images.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            shutil.copy2(image_path, target_images / image_path.name)
            copied_images += 1

            source_label_path = source_labels / f"{image_path.stem}.txt"
            target_label_path = target_labels / f"{image_path.stem}.txt"
            rewritten_lines: list[str] = []
            if source_label_path.is_file():
                with source_label_path.open("r", encoding="utf-8") as file:
                    for raw_line in file:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        source_class_id = int(float(parts[0]))
                        if source_class_id not in class_id_map:
                            continue
                        parts[0] = str(class_id_map[source_class_id])
                        if args.truncate_values:
                            if len(parts) < args.truncate_values:
                                raise ValueError(
                                    f"Cannot truncate label with only {len(parts)} values to {args.truncate_values}: {source_label_path}"
                                )
                            parts = parts[:args.truncate_values]
                        rewritten_lines.append(" ".join(parts))

            if rewritten_lines:
                target_label_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
                kept_annotations += len(rewritten_lines)
            else:
                target_label_path.write_text("", encoding="utf-8")

    filtered_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(kept_names)},
    }

    if "kpt_shape" in source_yaml:
        filtered_yaml["kpt_shape"] = source_yaml["kpt_shape"]
    if "flip_idx" in source_yaml:
        filtered_yaml["flip_idx"] = source_yaml["flip_idx"]

    output_yaml_path = output_dir / f"{args.dataset_name}.yaml"
    with output_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(filtered_yaml, file, sort_keys=False)

    print(f"Filtered dataset written to: {output_dir}")
    print(f"Dataset YAML written to: {output_yaml_path}")
    print(f"Kept classes ({len(kept_names)}): {', '.join(kept_names)}")
    print(f"Copied images: {copied_images}")
    print(f"Kept annotations: {kept_annotations}")


if __name__ == "__main__":
    main()
