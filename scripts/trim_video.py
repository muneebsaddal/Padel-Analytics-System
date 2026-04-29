from __future__ import annotations

import argparse
from pathlib import Path

import cv2


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
        description="Trim a video into a shorter clip using OpenCV.",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output clip path.")
    parser.add_argument("--start", required=True, help='Start time in seconds or "MM:SS" or "HH:MM:SS".')
    parser.add_argument("--end", help='End time in seconds or "MM:SS" or "HH:MM:SS".')
    parser.add_argument("--duration", help='Duration in seconds or "MM:SS" or "HH:MM:SS".')
    return parser


def main() -> None:
    args = build_parser().parse_args()

    video_path = require_file(args.video, "Video")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(start_seconds * fps))
    end_frame = min(total_frames, int(end_seconds * fps))
    if end_frame <= start_frame:
        raise ValueError("Trim range does not contain any frames.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written_frames = 0
    for frame_index in range(start_frame, end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written_frames += 1
        if written_frames % 100 == 0:
            print(f"Written {written_frames} frames")

    capture.release()
    writer.release()
    print(f"Trimmed clip written to: {output_path}")
    print(f"Frames written: {written_frames}")
    print(f"Approx duration: {written_frames / fps:.2f} seconds")


if __name__ == "__main__":
    main()
