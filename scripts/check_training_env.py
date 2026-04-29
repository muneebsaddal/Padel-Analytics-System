from __future__ import annotations

import platform
import sys

import torch
import ultralytics


def main() -> None:
    print(f"platform: {platform.platform()}")
    print(f"python executable: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"ultralytics: {ultralytics.__version__}")

    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        for device_index in range(torch.cuda.device_count()):
            print(f"cuda:{device_index}: {torch.cuda.get_device_name(device_index)}")
    else:
        print("No CUDA device is available to PyTorch in this environment.")


if __name__ == "__main__":
    main()
