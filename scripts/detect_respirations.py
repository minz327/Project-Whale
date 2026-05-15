"""
Detect whale respirations (breaths) from blow spray brightness near the blowhole.

Uses pose keypoint positions to locate the blowhole area (between rostrum_tip
and mid_saddle_patch), samples pixel brightness in that region, and detects
brightness spikes caused by blow spray.

Each brightness spike above threshold is grouped into a single breath event.
Consecutive above-threshold frames from the same blow are merged.

Usage:
    python scripts/detect_respirations.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_pose_for_blowhole(csv_path: Path) -> list[dict]:
    """Load pose data, return frames with rostrum + saddle patch for blowhole localization."""
    poses = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rx = r.get("rostrum_tip_x", "")
            ry = r.get("rostrum_tip_y", "")
            sx = r.get("mid_saddle_patch_x", "")
            sy = r.get("mid_saddle_patch_y", "")
            if rx and ry and sx and sy:
                poses.append({
                    "frame": int(r["frame"]),
                    "track_id": int(r["track_id"]),
                    "rx": float(rx), "ry": float(ry),
                    "sx": float(sx), "sy": float(sy),
                })
    return poses


def sample_blowhole_brightness(frame: np.ndarray, rx: float, ry: float,
                                sx: float, sy: float,
                                radius: int = 40) -> dict:
    """Sample pixel brightness in a region around the blowhole.

    The blowhole is approximately 30% of the way from rostrum to saddle patch.
    During a blow, this area fills with white spray.
    """
    # Blowhole center: 30% from rostrum toward saddle patch
    bx = int(rx + 0.3 * (sx - rx))
    by = int(ry + 0.3 * (sy - ry))

    h, w = frame.shape[:2]
    x1 = max(0, bx - radius)
    y1 = max(0, by - radius)
    x2 = min(w, bx + radius)
    y2 = min(h, by + radius)

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return {"brightness": 0, "white_pct": 0, "bx": bx, "by": by}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    white_mask = np.all(region > 180, axis=2)

    return {
        "brightness": float(np.mean(gray)),
        "white_pct": float(np.mean(white_mask)) * 100,
        "bx": bx,
        "by": by,
    }


def detect_breath_events(samples: list[dict], fps: float,
                         min_gap_s: float = 2.0) -> list[dict]:
    """Detect breath events from brightness signal.

    1. Compute adaptive threshold (mean + 1.5 * std of brightness)
    2. Find runs of above-threshold frames
    3. Merge runs closer than min_gap_s into single events
    4. Record peak frame per event
    """
    if not samples:
        return []

    brights = [s["brightness"] for s in samples]
    mean_b = np.mean(brights)
    std_b = np.std(brights)
    threshold = mean_b + 1.5 * std_b

    # Find above-threshold runs
    runs = []
    in_run = False
    run_start = 0
    for i, s in enumerate(samples):
        if s["brightness"] > threshold:
            if not in_run:
                run_start = i
                in_run = True
        else:
            if in_run:
                runs.append((run_start, i - 1))
                in_run = False
    if in_run:
        runs.append((run_start, len(samples) - 1))

    if not runs:
        return []

    # Merge runs that are close together (same blow event dissipating)
    min_gap_frames = int(min_gap_s * fps)
    merged_runs = [runs[0]]
    for start, end in runs[1:]:
        prev_end = merged_runs[-1][1]
        gap = samples[start]["frame"] - samples[prev_end]["frame"]
        if gap < min_gap_frames:
            merged_runs[-1] = (merged_runs[-1][0], end)
        else:
            merged_runs.append((start, end))

    # Extract breath events (peak frame per run)
    events = []
    for start, end in merged_runs:
        run_samples = samples[start:end + 1]
        peak_idx = max(range(len(run_samples)),
                       key=lambda i: run_samples[i]["brightness"])
        peak = run_samples[peak_idx]

        events.append({
            "breath_id": len(events) + 1,
            "peak_frame": peak["frame"],
            "peak_brightness": peak["brightness"],
            "peak_white_pct": peak["white_pct"],
            "start_frame": run_samples[0]["frame"],
            "end_frame": run_samples[-1]["frame"],
            "duration_frames": run_samples[-1]["frame"] - run_samples[0]["frame"],
            "blowhole_x": peak["bx"],
            "blowhole_y": peak["by"],
        })

    return events


def main():
    parser = argparse.ArgumentParser(description="Detect whale respirations from blow spray")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-radius", type=int, default=40,
                        help="Pixel radius around blowhole to sample (default: 40)")
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="Min seconds between distinct breaths (default: 2.0)")
    args = parser.parse_args()

    pose_csv = args.track_dir / "pose" / "pose_keypoints.csv"
    if not pose_csv.exists():
        print(f"Error: {pose_csv} not found")
        sys.exit(1)

    # Load pose data
    poses = load_pose_for_blowhole(pose_csv)
    print(f"Loaded {len(poses)} frames with blowhole-localizable keypoints")

    # Open video and sample brightness
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    pose_idx = 0
    samples = []

    print("Sampling blowhole brightness...")
    while pose_idx < len(poses):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == poses[pose_idx]["frame"]:
            p = poses[pose_idx]
            result = sample_blowhole_brightness(
                frame, p["rx"], p["ry"], p["sx"], p["sy"],
                radius=args.sample_radius)
            result["frame"] = p["frame"]
            result["track_id"] = p["track_id"]
            samples.append(result)
            pose_idx += 1

        frame_idx += 1

    cap.release()
    print(f"  Sampled {len(samples)} frames")

    # Detect breath events
    events = detect_breath_events(samples, fps, min_gap_s=args.min_gap)

    # Compute stats
    brights = [s["brightness"] for s in samples]
    mean_b = np.mean(brights)
    std_b = np.std(brights)
    threshold = mean_b + 1.5 * std_b

    # Compute inter-breath intervals
    ibis = []
    for i in range(1, len(events)):
        dt = (events[i]["peak_frame"] - events[i-1]["peak_frame"]) / fps
        ibis.append(dt)

    # Time span
    if samples:
        total_time_s = (samples[-1]["frame"] - samples[0]["frame"]) / fps
    else:
        total_time_s = 0

    # Save brightness signal
    signal_csv = args.track_dir / "respirations_signal.csv"
    with open(signal_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "track_id", "brightness",
                                          "white_pct", "bx", "by"])
        w.writeheader()
        w.writerows(samples)
    print(f"  Saved brightness signal to {signal_csv}")

    # Save breath events
    events_csv = args.track_dir / "respirations.csv"
    with open(events_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "breath_id", "peak_frame", "time_s",
            "peak_brightness", "peak_white_pct",
            "start_frame", "end_frame", "duration_frames",
            "blowhole_x", "blowhole_y"])
        w.writeheader()
        for e in events:
            e["time_s"] = round(e["peak_frame"] / fps, 2)
            w.writerow(e)
    print(f"  Saved breath events to {events_csv}")

    # Save summary
    summary = {
        "n_breaths": len(events),
        "total_time_s": round(total_time_s, 1),
        "breaths_per_min": round(len(events) / (total_time_s / 60), 1) if total_time_s > 0 else 0,
        "threshold": round(threshold, 1),
        "baseline_brightness": round(mean_b, 1),
        "inter_breath_intervals_s": [round(x, 1) for x in ibis],
        "mean_ibi_s": round(np.mean(ibis), 1) if ibis else 0,
        "events": events,
    }
    summary_path = args.track_dir / "respirations_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*55}")
    print(f"RESPIRATION DETECTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Threshold: brightness > {threshold:.0f} "
          f"(baseline {mean_b:.0f} ± {std_b:.0f})")
    print(f"  Breaths detected: {len(events)}")
    print(f"  Time span: {total_time_s:.1f}s")
    print(f"  Rate: {summary['breaths_per_min']} breaths/min")
    if ibis:
        print(f"  Avg interval: {np.mean(ibis):.1f}s "
              f"(range {min(ibis):.1f}–{max(ibis):.1f}s)")
    print()
    for e in events:
        t = e["peak_frame"] / fps
        print(f"  Breath {e['breath_id']}: frame {e['peak_frame']} "
              f"({t:.1f}s) — brightness {e['peak_brightness']:.0f}, "
              f"white {e['peak_white_pct']:.0f}%")


if __name__ == "__main__":
    main()
