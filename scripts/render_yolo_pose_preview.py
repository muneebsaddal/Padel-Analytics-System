from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POINT_COLOR = (255, 140, 0)
BOX_COLOR = (0, 220, 120)
TEXT_COLOR = (255, 255, 255)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a short preview video from a YOLO pose dataset with labels drawn on sample frames.",
    )
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--output", required=True, help="Output preview MP4 path.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-images", type=int, default=60, help="Maximum number of images to include.")
    parser.add_argument("--fps", type=float, default=4.0, help="Preview video FPS.")
    return parser


def draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def main() -> None:
    args = build_parser().parse_args()

    yaml_path = Path(args.data).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    data_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data_yaml["path"]).resolve()
    split_value = data_yaml.get(args.split)
    if not split_value:
        raise ValueError(f"Split '{args.split}' is not defined in {yaml_path}")

    images_dir = dataset_root / Path(split_value)
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    names_raw = data_yaml.get("names", {})
    if isinstance(names_raw, dict):
        names = {int(key): str(value) for key, value in names_raw.items()}
    else:
        names = {index: str(value) for index, value in enumerate(names_raw)}

    image_paths = [
        image_path
        for image_path in sorted(images_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ][: args.max_images]

    if not image_paths:
        raise ValueError(f"No images found in split '{args.split}' at {images_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read image: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    rendered = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                values = [float(value) for value in line.split()]
                if len(values) < 8:
                    continue

                class_id = int(values[0])
                center_x = values[1] * width
                center_y = values[2] * height
                box_width = values[3] * width
                box_height = values[4] * height
                x1 = int(center_x - (box_width / 2.0))
                y1 = int(center_y - (box_height / 2.0))
                x2 = int(center_x + (box_width / 2.0))
                y2 = int(center_y + (box_height / 2.0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)

                keypoints = values[5:]
                for idx in range(0, len(keypoints), 3):
                    if idx + 2 >= len(keypoints):
                        break
                    x_norm, y_norm, visibility = keypoints[idx : idx + 3]
                    if visibility <= 0:
                        continue
                    x_coord = int(x_norm * width)
                    y_coord = int(y_norm * height)
                    cv2.circle(frame, (x_coord, y_coord), 4, POINT_COLOR, -1)
                    draw_label(
                        frame,
                        names.get(class_id, str(class_id)),
                        (x_coord + 6, max(18, y_coord - 6)),
                        TEXT_COLOR,
                    )

        draw_label(frame, image_path.name, (20, 30), TEXT_COLOR)
        writer.write(frame)
        rendered += 1

    writer.release()
    print(f"Preview written to: {output_path}")
    print(f"Frames rendered: {rendered}")


if __name__ == "__main__":
    main()
