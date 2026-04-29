from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PRESETS = {
    "players-detection": {
        "task": "detect",
        "default_model": "weights/players_detection/yolov8m.pt",
        "default_imgsz": 640,
        "default_name": "players_detection",
    },
    "players-keypoints": {
        "task": "pose",
        "default_model": "weights/players_keypoints_detection/best.pt",
        "default_imgsz": 1280,
        "default_name": "players_keypoints",
    },
    "court-keypoints": {
        "task": "pose",
        "default_model": "weights/court_keypoints_detection/best.pt",
        "default_imgsz": 640,
        "default_name": "court_keypoints",
    },
    "ball-detection": {
        "task": "detect",
        "default_model": "artifacts/final_models/zenodo_ball_detection_best.pt",
        "default_imgsz": 640,
        "default_name": "ball_detection",
    },
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO models for padel tasks using dataset YAML files.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        required=True,
        help="Training preset to use.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--model",
        help="Base model or checkpoint to finetune from. Defaults to the preset checkpoint.",
    )
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--batch", type=positive_int, default=8)
    parser.add_argument("--imgsz", type=positive_int)
    parser.add_argument(
        "--device",
        default="cpu",
        help='Ultralytics device value. Use "cpu" on Windows CPU-only or "0" for the first CUDA GPU.',
    )
    parser.add_argument("--workers", type=non_negative_int, default=0)
    parser.add_argument("--patience", type=non_negative_int, default=25)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", help="Run name. Defaults to a preset-specific name.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", type=non_negative_int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if args.model:
        model_spec = args.model
        model_path = Path(model_spec)
        if model_path.exists():
            model_spec = str(model_path.resolve())
    else:
        model_path = Path(preset["default_model"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Default preset checkpoint not found: {model_path}. "
                "Pass --model with a local checkpoint path or an Ultralytics model name such as "
                "'yolov8n-pose.pt'."
            )
        model_spec = str(model_path)

    imgsz = args.imgsz or preset["default_imgsz"]
    run_name = args.name or preset["default_name"]

    model = YOLO(model_spec, task=preset["task"])
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=run_name,
        cache=args.cache,
        resume=args.resume,
        exist_ok=args.exist_ok,
        seed=args.seed,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
    )

    print("Training finished.")
    if results is not None:
        save_dir = getattr(results, "save_dir", None)
        if save_dir is not None:
            print(f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
