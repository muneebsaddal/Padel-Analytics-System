from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def yolo_box(xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float) -> tuple[float, float, float, float]:
    center_x = ((xmin + xmax) / 2.0) / width
    center_y = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return center_x, center_y, box_width, box_height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML detection annotations to YOLO dataset format.",
    )
    parser.add_argument("--xml", required=True, help="Path to CVAT XML export.")
    parser.add_argument("--images-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="cvat_detection")
    parser.add_argument("--class-names", required=True, help="Comma-separated class names in YOLO order.")
    parser.add_argument(
        "--include-labels",
        help="Optional comma-separated CVAT labels to keep. Defaults to all class names.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    xml_path = Path(args.xml).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"CVAT XML not found: {xml_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    class_names = parse_csv(args.class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    include_labels = set(parse_csv(args.include_labels)) if args.include_labels else set(class_names)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_nodes = list(root.findall(".//image"))
    random.Random(args.seed).shuffle(image_nodes)

    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    kept_images = 0
    for index, image_node in enumerate(image_nodes):
        image_name = image_node.attrib["name"]
        width = float(image_node.attrib["width"])
        height = float(image_node.attrib["height"])
        annotations = []

        for box in image_node.findall("box"):
            label = box.attrib.get("label", "")
            if label not in include_labels or label not in class_to_index:
                continue
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            center_x, center_y, box_width, box_height = yolo_box(xtl, ytl, xbr, ybr, width, height)
            annotations.append(
                f"{class_to_index[label]} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            )

        if not annotations:
            continue

        source_image = images_dir / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Source image not found: {source_image}")

        split = split_name(index, len(image_nodes), args.train_ratio, args.val_ratio)
        target_image = output_dir / split / "images" / Path(image_name).name
        target_label = output_dir / split / "labels" / f"{Path(image_name).stem}.txt"
        shutil.copy2(source_image, target_image)
        target_label.write_text("\n".join(annotations) + "\n", encoding="utf-8")
        kept_images += 1

    yaml_content = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path = output_dir / f"{args.dataset_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False)

    print(f"Converted {kept_images} annotated images.")
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
