from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a ZIP file from a URL and extract it into a target directory.",
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-name", default="dataset.zip")
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--skip-download-if-exists", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / args.archive_name

    if archive_path.exists() and args.skip_download_if_exists:
        print(f"Using existing archive: {archive_path}")
    else:
        print(f"Downloading: {args.url}")
        with requests.get(args.url, stream=True, timeout=args.timeout) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)
        print(f"Saved archive to: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(output_dir)
    print(f"Extracted archive into: {output_dir}")


if __name__ == "__main__":
    main()
