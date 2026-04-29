from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid"),
    "test": ("test",),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_dataset_root(source_dir: Path) -> Path:
    if (source_dir / "data.yaml").is_file():
        return source_dir

    yaml_candidates = sorted(source_dir.rglob("data.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError(
            f"Could not find a data.yaml file under {source_dir}. "
            "Point --source-dir at the extracted YOLO dataset root."
        )
    return yaml_candidates[0].parent


def resolve_split_dir(dataset_root: Path, split: str) -> Path | None:
    for candidate in SPLIT_ALIASES[split]:
        split_dir = dataset_root / candidate
        if split_dir.is_dir():
            return split_dir
    return None


def copy_split(dataset_root: Path, output_dir: Path, split: str) -> int:
    source_split_dir = resolve_split_dir(dataset_root, split)
    target_images = output_dir / split / "images"
    target_labels = output_dir / split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    if source_split_dir is None:
        return 0

    source_images = source_split_dir / "images"
    source_labels = source_split_dir / "labels"
    if not source_images.is_dir():
        return 0

    copied = 0
    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file():
            continue
        shutil.copy2(image_path, target_images / image_path.name)
        label_path = source_labels / f"{image_path.stem}.txt"
        if label_path.is_file():
            shutil.copy2(label_path, target_labels / label_path.name)
        else:
            (target_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
        copied += 1
    return copied


def build_yaml_content(
    dataset_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    source_yaml_path = dataset_root / "data.yaml"
    with open(source_yaml_path, "r", encoding="utf-8") as file:
        source_yaml = yaml.safe_load(file) or {}

    names = source_yaml.get("names", {0: "object"})
    if args.class_names:
        parsed_names = parse_csv(args.class_names)
        names = {index: name for index, name in enumerate(parsed_names)}

    yaml_content = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }

    if args.task == "pose":
        if args.kpt_shape:
            yaml_content["kpt_shape"] = [int(value) for value in parse_csv(args.kpt_shape)]
        elif source_yaml.get("kpt_shape"):
            yaml_content["kpt_shape"] = source_yaml["kpt_shape"]
        else:
            raise ValueError("Pose datasets require --kpt-shape if the source YAML does not define it.")

        if args.flip_idx:
            yaml_content["flip_idx"] = [int(value) for value in parse_csv(args.flip_idx)]
        else:
            yaml_content["flip_idx"] = source_yaml.get("flip_idx", [])

    return yaml_content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize an extracted YOLO dataset into train/val/test with a rewritten local data.yaml.",
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--class-names", help="Override class names with a comma-separated list.")
    parser.add_argument("--kpt-shape", help='Pose only, e.g. "13,3" or "12,3".')
    parser.add_argument("--flip-idx", default="", help="Pose only, comma-separated flip indices.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    dataset_root = find_dataset_root(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        split: copy_split(dataset_root, output_dir, split)
        for split in ("train", "val", "test")
    }

    yaml_content = build_yaml_content(dataset_root, output_dir, args)
    yaml_path = output_dir / f"{args.dataset_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False)

    print(f"Prepared dataset written to: {output_dir}")
    print(f"Dataset YAML written to: {yaml_path}")
    for split, count in split_counts.items():
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
