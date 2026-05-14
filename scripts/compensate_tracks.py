"""
Compensate whale tracks for camera ego-motion using sparse optical flow.

Instead of stabilizing the video (which fails on open ocean), estimates
frame-to-frame camera motion from background features and subtracts it
from track positions. This produces ego-motion-free trajectories.

Approach:
  1. Detect Shi-Tomasi corners on each frame, masking out whale bounding boxes
  2. Track features forward using Lucas-Kanade optical flow
  3. Estimate camera translation from median of flow vectors (robust to outliers)
  4. Accumulate camera displacement across frames
  5. Subtract accumulated displacement from track center + bbox positions

Usage:
    python scripts/compensate_tracks.py outputs/track/20240527-22 videos/20240527-22.MP4
    python scripts/compensate_tracks.py outputs/track/20240527-22 videos/20240527-22.MP4 --csv tracks_relinked.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
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


def compute_ego_motion(
    video_path: Path, tracks: list[dict], sample_rate: int,
    scale: float = 0.5,
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """Estimate per-frame cumulative camera displacement using sparse optical flow.

    Detects corners on water/background (masking whale bboxes), tracks them
    with Lucas-Kanade, and takes the median flow as camera translation.

    Returns:
        ego_motion: dict mapping frame_number → (cumulative_dx, cumulative_dy)
        flow_samples: list of per-frame flow measurements for diagnostics
    """
    # Build bbox mask per frame (to exclude whale regions from flow)
    bbox_by_frame: dict[int, list[tuple]] = {}
    for t in tracks:
        bbox_by_frame.setdefault(t["frame"], []).append((
            int(t["bbox_x1"]), int(t["bbox_y1"]),
            int(t["bbox_x2"]), int(t["bbox_y2"]),
        ))

    all_frames = sorted(set(t["frame"] for t in tracks))
    if not all_frames:
        return {}, []

    min_frame, max_frame = all_frames[0], all_frames[-1]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7,
    )

    prev_gray = None
    frame_idx = 0
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    ego_motion: dict[int, tuple[float, float]] = {}
    flow_samples: list[dict] = []

    # Process at sample_rate interval aligned with track frames
    sample_offset = min_frame % sample_rate

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= min_frame and frame_idx % sample_rate == sample_offset:
            # Downscale for speed
            if scale < 1.0:
                small = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                small = frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Create mask: exclude whale bounding boxes (scaled)
                mask = np.ones_like(gray, dtype=np.uint8) * 255
                for check_frame in [frame_idx, frame_idx - sample_rate]:
                    if check_frame in bbox_by_frame:
                        for (x1, y1, x2, y2) in bbox_by_frame[check_frame]:
                            # Scale bbox coords and add 30% margin
                            sx1 = int(x1 * scale)
                            sy1 = int(y1 * scale)
                            sx2 = int(x2 * scale)
                            sy2 = int(y2 * scale)
                            margin_x = int((sx2 - sx1) * 0.3)
                            margin_y = int((sy2 - sy1) * 0.3)
                            mask[max(0, sy1 - margin_y):min(mask.shape[0], sy2 + margin_y),
                                 max(0, sx1 - margin_x):min(mask.shape[1], sx2 + margin_x)] = 0

                # Detect features on previous frame (background only)
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_gray, mask=mask, **feature_params)

                if prev_pts is not None and len(prev_pts) >= 10:
                    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_pts, None, **lk_params)

                    good = status.ravel() == 1
                    if np.sum(good) >= 5:
                        flow = next_pts[good] - prev_pts[good]
                        # Robust: median of flow vectors (ignores outlier waves)
                        dx = float(np.median(flow[:, 0, 0])) / scale
                        dy = float(np.median(flow[:, 0, 1])) / scale
                        cumulative_dx += dx
                        cumulative_dy += dy
                        flow_samples.append({
                            "frame": frame_idx,
                            "dx": round(dx, 2),
                            "dy": round(dy, 2),
                            "n_features": int(np.sum(good)),
                        })

            ego_motion[frame_idx] = (cumulative_dx, cumulative_dy)
            prev_gray = gray

        frame_idx += 1

    cap.release()
    return ego_motion, flow_samples


def compensate_tracks(
    tracks: list[dict],
    ego_motion: dict[int, tuple[float, float]],
) -> list[dict]:
    """Subtract cumulative camera displacement from track positions."""
    sorted_frames = sorted(ego_motion.keys())
    if not sorted_frames:
        return tracks

    compensated = []
    for t in tracks:
        f = t["frame"]

        # Find ego-motion for this frame (interpolate if needed)
        if f in ego_motion:
            dx, dy = ego_motion[f]
        else:
            idx = np.searchsorted(sorted_frames, f)
            if idx == 0:
                dx, dy = ego_motion[sorted_frames[0]]
            elif idx >= len(sorted_frames):
                dx, dy = ego_motion[sorted_frames[-1]]
            else:
                f0, f1 = sorted_frames[idx - 1], sorted_frames[idx]
                ratio = (f - f0) / (f1 - f0) if f1 != f0 else 0
                dx0, dy0 = ego_motion[f0]
                dx1, dy1 = ego_motion[f1]
                dx = dx0 + ratio * (dx1 - dx0)
                dy = dy0 + ratio * (dy1 - dy0)

        new_t = dict(t)
        new_t["center_x"] = round(t["center_x"] - dx)
        new_t["center_y"] = round(t["center_y"] - dy)
        new_t["bbox_x1"] = round(t["bbox_x1"] - dx, 1)
        new_t["bbox_y1"] = round(t["bbox_y1"] - dy, 1)
        new_t["bbox_x2"] = round(t["bbox_x2"] - dx, 1)
        new_t["bbox_y2"] = round(t["bbox_y2"] - dy, 1)
        compensated.append(new_t)

    return compensated


def main():
    parser = argparse.ArgumentParser(
        description="Compensate whale tracks for camera ego-motion")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--csv", type=str, default="tracks_relinked.csv")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Frame sample rate used in tracking (default: 3)")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Downscale factor for optical flow (default: 0.5)")
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        csv_path = args.track_dir / "tracks.csv"
    if not csv_path.exists():
        print(f"Error: No track CSV found in {args.track_dir}")
        sys.exit(1)

    # Load tracks
    print(f"Loading tracks from {csv_path}...")
    tracks = load_tracks(csv_path)
    print(f"  {len(tracks)} track points")

    # Compute ego-motion from optical flow
    print(f"\nEstimating camera ego-motion (sample_rate={args.sample_rate}, "
          f"scale={args.scale})...")
    ego_motion, flow_samples = compute_ego_motion(
        args.video, tracks, args.sample_rate, args.scale)

    if not ego_motion:
        print("  No ego-motion estimated. Check video path.")
        sys.exit(1)

    # Report
    last_frame = max(ego_motion.keys())
    total_dx, total_dy = ego_motion[last_frame]
    total_drift = (total_dx**2 + total_dy**2) ** 0.5
    print(f"  Processed {len(ego_motion)} frames, "
          f"{len(flow_samples)} flow measurements")
    print(f"  Total camera drift: ({total_dx:.0f}, {total_dy:.0f}) px "
          f"= {total_drift:.0f} px")

    if flow_samples:
        per_frame_drifts = [
            (s["dx"]**2 + s["dy"]**2)**0.5 for s in flow_samples]
        print(f"  Per-frame drift: avg {np.mean(per_frame_drifts):.1f} px, "
              f"max {np.max(per_frame_drifts):.1f} px")

    # Apply compensation
    print(f"\nCompensating track positions...")
    compensated = compensate_tracks(tracks, ego_motion)

    # Save compensated tracks
    out_csv = args.track_dir / "tracks_compensated.csv"
    fieldnames = list(compensated[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compensated)
    print(f"  Saved compensated tracks to {out_csv}")

    # Save ego-motion data
    ego_out = args.track_dir / "ego_motion.json"
    ego_data = {
        "params": {
            "sample_rate": args.sample_rate,
            "scale": args.scale,
            "video": str(args.video),
        },
        "total_drift_px": {"x": round(total_dx, 1), "y": round(total_dy, 1)},
        "n_frames": len(ego_motion),
        "flow_samples": flow_samples,
    }
    with open(ego_out, "w") as f:
        json.dump(ego_data, f, indent=2)
    print(f"  Saved ego-motion data to {ego_out}")

    print(f"\n{'='*55}")
    print(f"EGO-MOTION COMPENSATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Camera drift removed: {total_drift:.0f} px total")
    print(f"  Use tracks_compensated.csv for ego-motion-free analysis")
    print(f"  Run track_metrics.py with --csv tracks_compensated.csv\n")


if __name__ == "__main__":
    main()
