"""
Compute camera-invariant relative metrics from whale tracks.

Instead of trying to stabilize absolute positions, computes metrics
that don't depend on camera motion:
  - Per-frame speed (displacement between consecutive detections of same track)
  - Inter-whale distance (distance between whales in the same frame)
  - Surfacing intervals (gaps in track presence)
  - Heading direction (within short windows)

Usage:
    python scripts/track_metrics.py outputs/track/20240527-22
    python scripts/track_metrics.py outputs/track/20240527-22 --csv tracks_relinked.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def compute_speed(pts: list[dict], fps: float, sample_rate: int) -> list[dict]:
    """Compute frame-to-frame speed for a single track.

    Speed is in pixels/second. Between consecutive detections only,
    so camera drift within a few frames is minimal.
    """
    pts_sorted = sorted(pts, key=lambda x: x["frame"])
    speeds = []

    for i in range(1, len(pts_sorted)):
        prev = pts_sorted[i - 1]
        curr = pts_sorted[i]

        frame_gap = curr["frame"] - prev["frame"]
        if frame_gap > sample_rate * 3:
            # Skip large gaps (likely a dive, camera may have moved a lot)
            continue

        dx = curr["center_x"] - prev["center_x"]
        dy = curr["center_y"] - prev["center_y"]
        dist_px = (dx**2 + dy**2) ** 0.5

        time_gap = frame_gap / fps
        speed_px_s = dist_px / time_gap if time_gap > 0 else 0

        speeds.append({
            "frame": curr["frame"],
            "speed_px_s": speed_px_s,
            "dist_px": dist_px,
            "frame_gap": frame_gap,
            "time_gap_s": time_gap,
            "heading_deg": np.degrees(np.arctan2(-dy, dx)) % 360,
        })

    return speeds


def compute_inter_whale_distance(tracks: list[dict]) -> list[dict]:
    """Compute distance between all whale pairs in the same frame."""
    by_frame: dict[int, list[dict]] = {}
    for t in tracks:
        by_frame.setdefault(t["frame"], []).append(t)

    distances = []
    for frame_num in sorted(by_frame.keys()):
        frame_tracks = by_frame[frame_num]
        if len(frame_tracks) < 2:
            continue

        # All unique pairs
        for i in range(len(frame_tracks)):
            for j in range(i + 1, len(frame_tracks)):
                t1 = frame_tracks[i]
                t2 = frame_tracks[j]
                dx = t1["center_x"] - t2["center_x"]
                dy = t1["center_y"] - t2["center_y"]
                dist = (dx**2 + dy**2) ** 0.5

                distances.append({
                    "frame": frame_num,
                    "track_a": t1["track_id"],
                    "track_b": t2["track_id"],
                    "distance_px": dist,
                })

    return distances


def compute_surfacing(pts: list[dict], fps: float,
                      sample_rate: int) -> list[dict]:
    """Detect surfacing bouts and dive gaps for a single track."""
    pts_sorted = sorted(pts, key=lambda x: x["frame"])
    bouts = []
    bout_start = pts_sorted[0]["frame"]
    bout_end = pts_sorted[0]["frame"]

    for i in range(1, len(pts_sorted)):
        gap = pts_sorted[i]["frame"] - pts_sorted[i - 1]["frame"]
        if gap > sample_rate * 5:
            # End of surfacing bout
            bouts.append({
                "start_frame": bout_start,
                "end_frame": bout_end,
                "duration_s": (bout_end - bout_start) / fps,
                "n_detections": sum(1 for p in pts_sorted
                                    if bout_start <= p["frame"] <= bout_end),
            })
            bout_start = pts_sorted[i]["frame"]
        bout_end = pts_sorted[i]["frame"]

    # Final bout
    bouts.append({
        "start_frame": bout_start,
        "end_frame": bout_end,
        "duration_s": (bout_end - bout_start) / fps,
        "n_detections": sum(1 for p in pts_sorted
                            if bout_start <= p["frame"] <= bout_end),
    })

    # Dive gaps between bouts
    gaps = []
    for i in range(1, len(bouts)):
        gap_start = bouts[i - 1]["end_frame"]
        gap_end = bouts[i]["start_frame"]
        gaps.append({
            "gap_start_frame": gap_start,
            "gap_end_frame": gap_end,
            "gap_duration_s": (gap_end - gap_start) / fps,
        })

    return bouts, gaps


def compute_bbox_size(pts: list[dict]) -> list[dict]:
    """Track bounding box size over time (proxy for apparent whale size / distance)."""
    sizes = []
    for p in sorted(pts, key=lambda x: x["frame"]):
        w = p["bbox_x2"] - p["bbox_x1"]
        h = p["bbox_y2"] - p["bbox_y1"]
        sizes.append({
            "frame": p["frame"],
            "width_px": w,
            "height_px": h,
            "area_px": w * h,
        })
    return sizes


def plot_metrics(track_speeds: dict, inter_distances: list[dict],
                 track_surfacing: dict, track_sizes: dict,
                 output_dir: Path, min_frames: int):
    """Generate metric plots."""
    colors = ["#00ff00", "#00ffff", "#ff00ff", "#ffa500", "#00a5ff",
              "#ff0000", "#80ff00", "#ff8080", "#0080ff", "#8000ff"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("gray")

    # 1. Speed over time
    ax1 = axes[0, 0]
    for tid in sorted(track_speeds.keys()):
        speeds = track_speeds[tid]
        if not speeds:
            continue
        frames = [s["frame"] for s in speeds]
        vals = [s["speed_px_s"] for s in speeds]
        color = colors[tid % len(colors)]
        ax1.plot(frames, vals, color=color, alpha=0.7, linewidth=1.5,
                 label=f"Track {tid}")
        # Smoothed
        if len(vals) >= 5:
            smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
            ax1.plot(frames[2:2+len(smooth)], smooth, color=color,
                     linewidth=2.5)

    ax1.set_xlabel("Frame", color="white")
    ax1.set_ylabel("Speed (px/s)", color="white")
    ax1.set_title("Frame-to-Frame Speed", color="white", fontsize=13)
    ax1.legend(fontsize=9, facecolor="black", edgecolor="gray",
               labelcolor="white")

    # 2. Inter-whale distance
    ax2 = axes[0, 1]
    if inter_distances:
        # Group by pair
        pairs: dict[tuple, list] = {}
        for d in inter_distances:
            pair = (min(d["track_a"], d["track_b"]),
                    max(d["track_a"], d["track_b"]))
            pairs.setdefault(pair, []).append(d)

        for i, (pair, dists) in enumerate(sorted(pairs.items())):
            frames = [d["frame"] for d in dists]
            vals = [d["distance_px"] for d in dists]
            if len(vals) < 3:
                continue
            color = colors[i % len(colors)]
            ax2.plot(frames, vals, color=color, alpha=0.7, linewidth=1.5,
                     label=f"T{pair[0]}↔T{pair[1]}")
            if len(vals) >= 5:
                smooth = np.convolve(vals, np.ones(5)/5, mode="valid")
                ax2.plot(frames[2:2+len(smooth)], smooth, color=color,
                         linewidth=2.5)

        ax2.legend(fontsize=9, facecolor="black", edgecolor="gray",
                   labelcolor="white")
    else:
        ax2.text(0.5, 0.5, "No simultaneous\nwhale pairs found",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=14, color="white")

    ax2.set_xlabel("Frame", color="white")
    ax2.set_ylabel("Distance (px)", color="white")
    ax2.set_title("Inter-Whale Distance", color="white", fontsize=13)

    # 3. Surfacing timeline
    ax3 = axes[1, 0]
    sorted_tids = sorted(track_surfacing.keys())
    for i, tid in enumerate(sorted_tids):
        bouts, gaps = track_surfacing[tid]
        color = colors[tid % len(colors)]
        for bout in bouts:
            ax3.barh(i, bout["end_frame"] - bout["start_frame"],
                     left=bout["start_frame"], height=0.6,
                     color=color, alpha=0.8)
            ax3.text(bout["start_frame"], i + 0.35,
                     f'{bout["duration_s"]:.1f}s',
                     fontsize=8, color="white", va="bottom")

    ax3.set_yticks(range(len(sorted_tids)))
    ax3.set_yticklabels([f"Track {tid}" for tid in sorted_tids],
                        color="white", fontsize=10)
    ax3.set_xlabel("Frame", color="white")
    ax3.set_title("Surfacing Bouts (bars) & Dive Gaps", color="white",
                  fontsize=13)

    # 4. Bounding box area (proxy for apparent size / altitude)
    ax4 = axes[1, 1]
    for tid in sorted(track_sizes.keys()):
        sizes = track_sizes[tid]
        if not sizes:
            continue
        frames = [s["frame"] for s in sizes]
        areas = [s["area_px"] for s in sizes]
        color = colors[tid % len(colors)]
        ax4.plot(frames, areas, color=color, alpha=0.5, linewidth=1)
        if len(areas) >= 5:
            smooth = np.convolve(areas, np.ones(5)/5, mode="valid")
            ax4.plot(frames[2:2+len(smooth)], smooth, color=color,
                     linewidth=2.5, label=f"Track {tid}")

    ax4.set_xlabel("Frame", color="white")
    ax4.set_ylabel("BBox area (px²)", color="white")
    ax4.set_title("Apparent Size (bbox area — proxy for distance/altitude)",
                  color="white", fontsize=13)
    ax4.legend(fontsize=9, facecolor="black", edgecolor="gray",
               labelcolor="white")

    plt.suptitle("Whale Track Metrics (camera-invariant)",
                 fontsize=16, color="white", y=1.01)
    plt.tight_layout()

    out_path = output_dir / "track_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved metrics plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute relative track metrics")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("--csv", type=str, default="tracks.csv")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (default: 30)")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Frame sample rate used in tracking (default: 3)")
    parser.add_argument("--min-frames", type=int, default=5)
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    tracks = load_tracks(csv_path)
    print(f"Loaded {len(tracks)} track points")

    # Group by track ID, filter short
    by_id: dict[int, list[dict]] = {}
    for t in tracks:
        by_id.setdefault(t["track_id"], []).append(t)

    long_tracks = {tid: pts for tid, pts in by_id.items()
                   if len(pts) >= args.min_frames}
    print(f"  {len(long_tracks)} tracks with ≥{args.min_frames} frames\n")

    # 1. Per-track speed
    print("Computing speeds...")
    track_speeds: dict[int, list] = {}
    for tid, pts in long_tracks.items():
        speeds = compute_speed(pts, args.fps, args.sample_rate)
        track_speeds[tid] = speeds
        if speeds:
            avg = np.mean([s["speed_px_s"] for s in speeds])
            mx = np.max([s["speed_px_s"] for s in speeds])
            print(f"  Track {tid}: avg {avg:.0f} px/s, max {mx:.0f} px/s")

    # 2. Inter-whale distance
    print("\nComputing inter-whale distances...")
    # Only use long tracks
    filtered = [t for t in tracks if t["track_id"] in long_tracks]
    inter_dists = compute_inter_whale_distance(filtered)
    if inter_dists:
        pairs: dict[tuple, list] = {}
        for d in inter_dists:
            pair = (min(d["track_a"], d["track_b"]),
                    max(d["track_a"], d["track_b"]))
            pairs.setdefault(pair, []).append(d["distance_px"])
        for pair, dists in sorted(pairs.items()):
            if len(dists) >= 3:
                print(f"  T{pair[0]}↔T{pair[1]}: {len(dists)} frames, "
                      f"avg {np.mean(dists):.0f}px, "
                      f"range {np.min(dists):.0f}–{np.max(dists):.0f}px")
    else:
        print("  No simultaneous whale pairs found")

    # 3. Surfacing intervals
    print("\nComputing surfacing bouts...")
    track_surfacing: dict[int, tuple] = {}
    for tid, pts in long_tracks.items():
        bouts, gaps = compute_surfacing(pts, args.fps, args.sample_rate)
        track_surfacing[tid] = (bouts, gaps)
        bout_durations = [b["duration_s"] for b in bouts]
        gap_durations = [g["gap_duration_s"] for g in gaps]
        print(f"  Track {tid}: {len(bouts)} bout(s) "
              f"({', '.join(f'{d:.1f}s' for d in bout_durations)})")
        if gap_durations:
            print(f"    Dive gaps: {', '.join(f'{d:.1f}s' for d in gap_durations)}")

    # 4. Bbox sizes
    print("\nComputing apparent sizes...")
    track_sizes: dict[int, list] = {}
    for tid, pts in long_tracks.items():
        sizes = compute_bbox_size(pts)
        track_sizes[tid] = sizes
        areas = [s["area_px"] for s in sizes]
        print(f"  Track {tid}: avg bbox area {np.mean(areas):.0f}px²")

    # Plot
    print("\nGenerating plots...")
    plot_metrics(track_speeds, inter_dists, track_surfacing, track_sizes,
                 args.track_dir, args.min_frames)

    # Save metrics as JSON
    metrics = {
        "speeds": {str(tid): {
            "avg_px_s": round(np.mean([s["speed_px_s"] for s in sp]), 1) if sp else 0,
            "max_px_s": round(np.max([s["speed_px_s"] for s in sp]), 1) if sp else 0,
            "n_measurements": len(sp),
        } for tid, sp in track_speeds.items()},
        "surfacing": {str(tid): {
            "n_bouts": len(bouts),
            "bout_durations_s": [round(b["duration_s"], 1) for b in bouts],
            "dive_gaps_s": [round(g["gap_duration_s"], 1) for g in gaps],
        } for tid, (bouts, gaps) in track_surfacing.items()},
    }
    with open(args.track_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {args.track_dir / 'metrics.json'}\n")


if __name__ == "__main__":
    main()
