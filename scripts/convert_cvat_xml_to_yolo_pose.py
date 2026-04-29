from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_point_string(points_value: str) -> tuple[float, float]:
    x_value, y_value = points_value.split(",")
    return float(x_value), float(y_value)


def normalize_pose(
    keypoint_map: dict[str, tuple[float, float]],
    keypoint_names: list[str],
    width: float,
    height: float,
    bbox_padding: float,
) -> str | None:
    present_points = [keypoint_map[name] for name in keypoint_names if name in keypoint_map]
    if len(present_points) < 4:
        return None

    xs = [point[0] for point in present_points]
    ys = [point[1] for point in present_points]
    xmin = clamp(min(xs) - bbox_padding, 0.0, width)
    ymin = clamp(min(ys) - bbox_padding, 0.0, height)
    xmax = clamp(max(xs) + bbox_padding, 0.0, width)
    ymax = clamp(max(ys) + bbox_padding, 0.0, height)

    center_x = ((xmin + xmax) / 2.0) / width
    center_y = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    pose_values: list[str] = ["0", f"{center_x:.6f}", f"{center_y:.6f}", f"{box_width:.6f}", f"{box_height:.6f}"]
    for keypoint_name in keypoint_names:
        if keypoint_name in keypoint_map:
            x_coord, y_coord = keypoint_map[keypoint_name]
            pose_values.extend([f"{x_coord / width:.6f}", f"{y_coord / height:.6f}", "2"])
        else:
            pose_values.extend(["0.000000", "0.000000", "0"])
    return " ".join(pose_values)


def collect_skeletons(image_node: ET.Element) -> list[dict[str, tuple[float, float]]]:
    skeletons: list[dict[str, tuple[float, float]]] = []
    for skeleton in image_node.findall("skeleton"):
        keypoint_map: dict[str, tuple[float, float]] = {}
        for points_node in skeleton.findall("points"):
            if points_node.attrib.get("outside") == "1":
                continue
            label = points_node.attrib.get("label", "")
            if not label:
                continue
            keypoint_map[label] = parse_point_string(points_node.attrib["points"])
        if keypoint_map:
            skeletons.append(keypoint_map)
    return skeletons


def collect_grouped_points(image_node: ET.Element) -> list[dict[str, tuple[float, float]]]:
    grouped_points: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for points_node in image_node.findall("points"):
        if points_node.attrib.get("outside") == "1":
            continue
        group_id = points_node.attrib.get("group_id", "0")
        label = points_node.attrib.get("label", "")
        if not label:
            continue
        grouped_points[group_id][label] = parse_point_string(points_node.attrib["points"])
    return [group for group in grouped_points.values() if group]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML pose annotations to YOLO pose dataset format.",
    )
    parser.add_argument("--xml", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="cvat_pose")
    parser.add_argument("--keypoint-names", required=True, help="Comma-separated keypoint names in YOLO order.")
    parser.add_argument(
        "--flip-idx",
        default="",
        help="Comma-separated flip indices matching the keypoint order.",
    )
    parser.add_argument("--bbox-padding", type=float, default=10.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    xml_path = Path(args.xml).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    keypoint_names = parse_csv(args.keypoint_names)
    flip_idx = [int(value) for value in parse_csv(args.flip_idx)]

    if not xml_path.is_file():
        raise FileNotFoundError(f"CVAT XML not found: {xml_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

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

        pose_groups = collect_skeletons(image_node)
        if not pose_groups:
            pose_groups = collect_grouped_points(image_node)

        annotations = []
        for pose_group in pose_groups:
            line = normalize_pose(
                keypoint_map=pose_group,
                keypoint_names=keypoint_names,
                width=width,
                height=height,
                bbox_padding=args.bbox_padding,
            )
            if line is not None:
                annotations.append(line)

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
        "names": {0: "object"},
        "kpt_shape": [len(keypoint_names), 3],
        "flip_idx": flip_idx,
    }
    yaml_path = output_dir / f"{args.dataset_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False)

    print(f"Converted {kept_images} annotated images.")
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
