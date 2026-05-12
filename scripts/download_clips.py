"""
Download clips listed in data/clip_log.csv using yt-dlp.

Reads the clip log, downloads clips with status 'pending',
saves to data/raw_clips/, and updates the CSV afterward.
"""

import csv
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_LOG = PROJECT_ROOT / "data" / "clip_log.csv"
RAW_CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips"

# Max resolution to keep file sizes manageable
MAX_HEIGHT = 1080


def download_clip(clip_id: str, url: str, output_dir: Path) -> tuple[bool, str]:
    """Download a single clip using yt-dlp. Returns (success, local_filename)."""
    output_template = str(output_dir / f"{clip_id}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-overwrites",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Find the downloaded file
            for f in output_dir.iterdir():
                if f.stem == clip_id:
                    return True, f.name
            # Fallback filename
            return True, f"{clip_id}.mp4"
        else:
            print(f"  [ERROR] yt-dlp failed for {clip_id}:\n{result.stderr[:500]}")
            return False, ""
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Download timed out for {clip_id}")
        return False, ""
    except Exception as e:
        print(f"  [ERROR] Unexpected error for {clip_id}: {e}")
        return False, ""


def main():
    if not CLIP_LOG.exists():
        print(f"Clip log not found: {CLIP_LOG}")
        sys.exit(1)

    RAW_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Read clip log
    with open(CLIP_LOG, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    pending = [r for r in rows if r["download_status"] == "pending"]
    print(f"Found {len(pending)} pending clip(s) to download.\n")

    for row in rows:
        if row["download_status"] != "pending":
            print(f"[SKIP] {row['clip_id']} — status: {row['download_status']}")
            continue

        print(f"[DOWNLOADING] {row['clip_id']} from {row['url']}")
        success, filename = download_clip(row["clip_id"], row["url"], RAW_CLIPS_DIR)

        if success:
            row["download_status"] = "downloaded"
            row["local_filename"] = filename
            print(f"  [OK] Saved as {filename}\n")
        else:
            row["download_status"] = "failed"
            print(f"  [FAILED] {row['clip_id']}\n")

    # Write updated clip log
    with open(CLIP_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Clip log updated.")

    # Summary
    statuses = {}
    for r in rows:
        s = r["download_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print("\nSummary:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
