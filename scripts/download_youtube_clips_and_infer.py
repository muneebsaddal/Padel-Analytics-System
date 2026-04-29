from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import ceil
from pathlib import Path

import cv2
import yt_dlp


DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=3EQGUMDBLR8&t=122s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video, split it into fixed-duration clips, and run batch inference with CVAT XML export.",
    )
    parser.add_argument("--url", default=DEFAULT_VIDEO_URL, help="YouTube video URL.")
    parser.add_argument(
        "--workspace-dir",
        default="data/youtube_pipeline",
        help="Base workspace directory for downloads, clips, and inference outputs.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional override for the raw downloaded video directory.",
    )
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Optional override for the generated clip directory.",
    )
    parser.add_argument(
        "--inference-dir",
        default=None,
        help="Optional override for inferred overlay video outputs.",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        help="Optional override for CVAT XML outputs.",
    )
    parser.add_argument(
        "--cvat-frames-dir",
        default=None,
        help="Optional directory for exporting per-frame images alongside XML for CVAT image-task import.",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=10.0,
        help="Clip duration in seconds.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Optional cap on the number of generated clips. Use 0 to process the whole video.",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=0,
        help="Optional frame cap forwarded to inference for quick tests. Use 0 to disable.",
    )
    parser.add_argument("--device", default="0", help='Inference device passed through to run_combined_overlay.py.')
    parser.add_argument("--overwrite", action="store_true", help="Redownload, regenerate clips, and overwrite outputs.")
    parser.add_argument(
        "--skip-download-if-exists",
        action="store_true",
        help="Reuse an existing downloaded video if present.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only download and split clips without running inference.",
    )
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dirs(args: argparse.Namespace, repo_root: Path) -> dict[str, Path]:
    workspace_dir = Path(args.workspace_dir)
    if not workspace_dir.is_absolute():
        workspace_dir = repo_root / workspace_dir
    workspace_dir = workspace_dir.resolve()

    download_dir = Path(args.download_dir).resolve() if args.download_dir else workspace_dir / "downloads"
    clips_dir = Path(args.clips_dir).resolve() if args.clips_dir else workspace_dir / "clips"
    inference_dir = Path(args.inference_dir).resolve() if args.inference_dir else workspace_dir / "inference"
    xml_dir = Path(args.xml_dir).resolve() if args.xml_dir else workspace_dir / "xml"

    paths = {
        "workspace_dir": ensure_dir(workspace_dir),
        "download_dir": ensure_dir(download_dir),
        "clips_dir": ensure_dir(clips_dir),
        "inference_dir": ensure_dir(inference_dir),
        "xml_dir": ensure_dir(xml_dir),
    }
    if args.cvat_frames_dir:
        paths["cvat_frames_dir"] = ensure_dir(Path(args.cvat_frames_dir).resolve())
    return paths


def find_downloaded_video(download_dir: Path, video_id: str) -> Path | None:
    matches = sorted(download_dir.glob(f"{video_id}.*"))
    for match in matches:
        if match.is_file() and match.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return match
    return None


def download_video(url: str, download_dir: Path, *, overwrite: bool, skip_if_exists: bool) -> tuple[Path, dict]:
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info["id"]
    existing_path = find_downloaded_video(download_dir, video_id)
    if existing_path is not None and (skip_if_exists or not overwrite):
        print(f"Using existing download: {existing_path}")
        return existing_path, info

    outtmpl = str(download_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }
    print(f"Downloading video: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloaded_path = find_downloaded_video(download_dir, info["id"])
    if downloaded_path is None:
        raise FileNotFoundError(f"Downloaded video for id {info['id']} was not found in {download_dir}")

    print(f"Downloaded video to: {downloaded_path}")
    return downloaded_path, info


def split_video_into_clips(
    video_path: Path,
    clips_dir: Path,
    *,
    clip_duration: float,
    max_clips: int,
    overwrite: bool,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_frames = max(1, int(round(clip_duration * fps)))
    total_clips = ceil(total_frames / clip_frames) if total_frames > 0 else 0
    if max_clips > 0:
        total_clips = min(total_clips, max_clips)

    print(
        "Splitting video into clips:",
        f"fps={fps:.2f}",
        f"frames={total_frames}",
        f"clip_frames={clip_frames}",
        f"planned_clips={total_clips}",
    )

    clip_paths: list[Path] = []
    for clip_index in range(total_clips):
        clip_path = clips_dir / f"{video_path.stem}_clip_{clip_index:04d}.mp4"
        clip_paths.append(clip_path)
        if clip_path.exists() and not overwrite:
            continue

        start_frame = clip_index * clip_frames
        end_frame = min(total_frames, start_frame + clip_frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open clip writer for: {clip_path}")

        written = 0
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.release()

        if written == 0:
            if clip_path.exists():
                clip_path.unlink()
            break

        if (clip_index + 1) % 25 == 0 or clip_index == total_clips - 1:
            print(f"Generated {clip_index + 1}/{total_clips} clips")

    capture.release()
    return [clip_path for clip_path in clip_paths if clip_path.is_file()]


def write_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest written to: {manifest_path}")


def run_batch_inference(
    repo_root: Path,
    clips_dir: Path,
    inference_dir: Path,
    xml_dir: Path,
    *,
    cvat_frames_dir: Path | None,
    device: str,
    max_frames_per_clip: int,
) -> None:
    command = [
        sys.executable,
        "scripts/run_combined_overlay.py",
        "--video-dir",
        str(clips_dir),
        "--output-dir",
        str(inference_dir),
        "--xml-dir",
        str(xml_dir),
        "--device",
        str(device),
    ]
    if cvat_frames_dir is not None:
        command.extend(["--cvat-frames-dir", str(cvat_frames_dir)])
    if max_frames_per_clip > 0:
        command.extend(["--max-frames", str(max_frames_per_clip)])

    print("Running batch inference on generated clips")
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths = resolve_dirs(args, repo_root)

    downloaded_video_path, info = download_video(
        args.url,
        paths["download_dir"],
        overwrite=args.overwrite,
        skip_if_exists=args.skip_download_if_exists,
    )
    clip_paths = split_video_into_clips(
        downloaded_video_path,
        paths["clips_dir"],
        clip_duration=args.clip_duration,
        max_clips=args.max_clips,
        overwrite=args.overwrite,
    )
    if not clip_paths:
        raise RuntimeError("No clips were generated.")

    manifest = {
        "url": args.url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "downloaded_video": str(downloaded_video_path),
        "clip_duration_seconds": args.clip_duration,
        "clip_count": len(clip_paths),
        "clips_dir": str(paths["clips_dir"]),
        "inference_dir": str(paths["inference_dir"]),
        "xml_dir": str(paths["xml_dir"]),
        "cvat_frames_dir": str(paths["cvat_frames_dir"]) if "cvat_frames_dir" in paths else None,
    }
    write_manifest(paths["workspace_dir"] / "workflow_manifest.json", manifest)

    if args.skip_inference:
        print("Skipping inference as requested.")
        return

    run_batch_inference(
        repo_root,
        paths["clips_dir"],
        paths["inference_dir"],
        paths["xml_dir"],
        cvat_frames_dir=paths.get("cvat_frames_dir"),
        device=args.device,
        max_frames_per_clip=args.max_frames_per_clip,
    )


if __name__ == "__main__":
    main()
