"""Download the pretrained volleyball ball detector.

The weights come from masouduut94/volleyball_analytics, which publishes a
YOLOv8n-seg model trained for 200 epochs at imgsz 1024 on a single `ball`
class (reported mAP50 0.948, precision 0.935, recall 0.909 on its own
validation split).

The Drive download is a zip of the whole training run, so this extracts the
checkpoint out of it and leaves the run directory in place — the training
curves and confusion matrix are worth a look before trusting the model.

Usage:
    python fetch_models.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DRIVE_FILE_ID = "1KXDunsC1ALOObb303n9j6HHO7Bxz1HR_"
MODELS_DIR = Path("models")
ARCHIVE = MODELS_DIR / "ball_seg_run.zip"
RUN_DIR = MODELS_DIR / "ball_run"
CHECKPOINT = RUN_DIR / "ball_segment" / "model1" / "weights" / "best.pt"
DESTINATION = MODELS_DIR / "volleyball_ball.pt"


def main() -> int:
    if DESTINATION.exists():
        print(f"{DESTINATION} already present; nothing to do.")
        return 0

    MODELS_DIR.mkdir(exist_ok=True)

    if not ARCHIVE.exists():
        try:
            import gdown  # noqa: F401
        except ImportError:
            print("gdown is required: pip install gdown", file=sys.stderr)
            return 1
        print("Downloading weights (~43 MB) from Google Drive...")
        subprocess.run(
            [sys.executable, "-m", "gdown", DRIVE_FILE_ID, "-O", str(ARCHIVE)],
            check=True,
        )

    if not zipfile.is_zipfile(ARCHIVE):
        print(f"{ARCHIVE} is not a zip archive; the download likely failed.",
              file=sys.stderr)
        return 1

    print(f"Extracting to {RUN_DIR}...")
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(RUN_DIR)

    if not CHECKPOINT.exists():
        print(f"Expected checkpoint missing: {CHECKPOINT}", file=sys.stderr)
        return 1

    shutil.copy2(CHECKPOINT, DESTINATION)
    print(f"\nReady: {DESTINATION}")
    print(f"  .venv/bin/python track.py --source clip.mp4 "
          f"--ball-model {DESTINATION} --ball-conf 0.25 --ball-imgsz 1024")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
