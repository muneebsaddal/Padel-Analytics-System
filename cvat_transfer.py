import os
from pathlib import Path

import requests


CVAT_URL = os.environ.get("CVAT_URL", "https://app.cvat.ai")
CVAT_TOKEN = os.environ.get("CVAT_TOKEN")
TASK_ID = int(os.environ.get("CVAT_TASK_ID", "0"))
TAR_PATH = Path(
    os.environ.get(
        "CVAT_TAR_PATH",
        "/workspace/Padel-Analytics-System/data/youtube_batch/merged_frames.tar",
    )
)


def main() -> None:
    if not CVAT_TOKEN:
        raise EnvironmentError("Set CVAT_TOKEN before running this script.")
    if TASK_ID <= 0:
        raise EnvironmentError("Set CVAT_TASK_ID to a positive integer before running this script.")
    if not TAR_PATH.is_file():
        raise FileNotFoundError(f"TAR file not found: {TAR_PATH}")

    headers = {"Authorization": f"Bearer {CVAT_TOKEN}"}
    upload_url = f"{CVAT_URL}/api/tasks/{TASK_ID}/data"

    print("Uploading TAR...")
    with TAR_PATH.open("rb") as file_obj:
        response = requests.post(
            upload_url,
            headers=headers,
            files={"client_files[0]": ("merged_frames.tar", file_obj)},
            data={"image_quality": 70},
            timeout=3600,
        )

    print("Status:", response.status_code)
    print("Response:", response.text)


if __name__ == "__main__":
    main()
