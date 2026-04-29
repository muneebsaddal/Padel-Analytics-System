import os

import requests


CVAT_URL = os.environ.get("CVAT_URL", "https://app.cvat.ai")
CVAT_TOKEN = os.environ.get("CVAT_TOKEN")


def main() -> None:
    if not CVAT_TOKEN:
        raise EnvironmentError("Set CVAT_TOKEN before running this script.")

    response = requests.get(
        f"{CVAT_URL}/api/users/self",
        headers={"Authorization": f"Bearer {CVAT_TOKEN}"},
        timeout=60,
    )
    print("Status:", response.status_code)
    print("Response:", response.text)


if __name__ == "__main__":
    main()
