"""
Visualize whale tracks as a clear trajectory map.

Produces a static plot showing all whale paths over a background frame,
plus a timeline chart showing when each track is active.

Usage:
    python scripts/visualize_tracks.py outputs/track/20240527-22
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


TRACK_COLORS_MPL = [
    "#00ff00", "#00ffff", "#ff00ff", "#ffa500", "#00a5ff",
    "#ff0000", "#80ff00", "#ff8080", "#0080ff", "#8000ff",
]


def load_tracks(csv_path: Path) -> list[dict]:
    tracks = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            tracks.append({
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "class_name": row["class_name"],
                "confidence": float(row["confidence"]),
                "center_x": int(row["center_x"]),
                "center_y": int(row["center_y"]),
                "bbox_x1": float(row["bbox_x1"]),
                "bbox_y1": float(row["bbox_y1"]),
                "bbox_x2": float(row["bbox_x2"]),
                "bbox_y2": float(row["bbox_y2"]),
            })
    return tracks


def get_background_frame(track_dir: Path) -> np.ndarray | None:
    """Try to find a video file to grab a background frame from."""
    # Look for the video based on summary
    import json
    summary_path = track_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        video_stem = summary.get("video", "")
        # Search common locations
        for pattern in [
            Path("exeter") / f"{video_stem}.MP4",
            Path("exeter") / f"{video_stem}.mp4",
            Path("data/raw_clips") / f"{video_stem}.mp4",
            Path("data/raw_clips") / f"{video_stem}.MP4",
        ]:
            if pattern.exists():
                cap = cv2.VideoCapture(str(pattern))
                # Grab a frame from the middle
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize whale tracks")
    parser.add_argument("track_dir", type=Path, help="Directory with tracks.csv")
    parser.add_argument("--csv", type=str, default="tracks.csv",
                        help="CSV filename to load (default: tracks.csv)")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Only show tracks with at least N frames (default: 5)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output filenames (e.g. '_relinked')")
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    tracks = load_tracks(csv_path)
    print(f"Loaded {len(tracks)} track points")

    # Group by track ID
    by_id: dict[int, list[dict]] = {}
    for t in tracks:
        by_id.setdefault(t["track_id"], []).append(t)

    # Filter short tracks
    long_tracks = {tid: pts for tid, pts in by_id.items()
                   if len(pts) >= args.min_frames}
    short_tracks = {tid: pts for tid, pts in by_id.items()
                    if len(pts) < args.min_frames}

    print(f"  {len(long_tracks)} tracks with ≥{args.min_frames} frames "
          f"(showing these)")
    print(f"  {len(short_tracks)} short tracks filtered out "
          f"({sum(len(p) for p in short_tracks.values())} points)")

    # Try to get background frame
    bg = get_background_frame(args.track_dir)

    # --- FIGURE 1: Trajectory map ---
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    if bg is not None:
        ax.imshow(bg, alpha=0.4)
        h, w = bg.shape[:2]
    else:
        # Determine bounds from data
        all_x = [t["center_x"] for t in tracks]
        all_y = [t["center_y"] for t in tracks]
        w = max(all_x) + 100
        h = max(all_y) + 100
        ax.set_facecolor("#1a3a4a")

    for tid in sorted(long_tracks.keys()):
        pts = sorted(long_tracks[tid], key=lambda t: t["frame"])
        xs = [p["center_x"] for p in pts]
        ys = [p["center_y"] for p in pts]
        color = TRACK_COLORS_MPL[tid % len(TRACK_COLORS_MPL)]

        # Draw path
        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.8, zorder=2)

        # Start marker
        ax.scatter(xs[0], ys[0], color=color, s=120, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=3)
        # End marker
        ax.scatter(xs[-1], ys[-1], color=color, s=120, marker="s",
                   edgecolors="white", linewidths=1.5, zorder=3)

        # Label at start
        ax.annotate(f"ID {tid}", (xs[0], ys[0]),
                    textcoords="offset points", xytext=(10, -15),
                    fontsize=11, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip Y axis (image coords)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_title(f"Whale Trajectories — {args.track_dir.name}\n"
                 f"(tracks with ≥{args.min_frames} frames, "
                 f"● = start, ■ = end)", fontsize=14)

    # Legend
    patches = [mpatches.Patch(
        color=TRACK_COLORS_MPL[tid % len(TRACK_COLORS_MPL)],
        label=f"Track {tid} ({len(long_tracks[tid])} frames)")
        for tid in sorted(long_tracks.keys())]
    ax.legend(handles=patches, loc="upper right", fontsize=10,
              facecolor="black", edgecolor="gray", labelcolor="white")

    plt.tight_layout()
    map_path = args.track_dir / f"trajectory_map{args.suffix}.png"
    fig.savefig(map_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved trajectory map to {map_path}")

    # --- FIGURE 2: Timeline ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 5))
    ax2.set_facecolor("#1a1a2e")

    sorted_tids = sorted(long_tracks.keys())
    for i, tid in enumerate(sorted_tids):
        pts = long_tracks[tid]
        frames = [p["frame"] for p in pts]
        color = TRACK_COLORS_MPL[tid % len(TRACK_COLORS_MPL)]

        # Draw dots for each frame this track is present
        ax2.scatter(frames, [i] * len(frames), color=color, s=20, zorder=2)
        # Span line
        ax2.plot([min(frames), max(frames)], [i, i],
                 color=color, linewidth=3, alpha=0.4, zorder=1)

    ax2.set_yticks(range(len(sorted_tids)))
    ax2.set_yticklabels([f"Track {tid}" for tid in sorted_tids],
                        fontsize=11, color="white")
    ax2.set_xlabel("Frame number", fontsize=12, color="white")
    ax2.set_title(f"Track Timeline — {args.track_dir.name}", fontsize=14, color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    timeline_path = args.track_dir / f"track_timeline{args.suffix}.png"
    fig2.savefig(timeline_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved track timeline to {timeline_path}")


if __name__ == "__main__":
    main()
