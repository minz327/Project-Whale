"""
Visualize SLEAP pose keypoints overlaid on video frames.

Draws skeleton connections and keypoint markers on whale detection crops.

Usage:
    python scripts/visualize_pose.py outputs/track/20240527-22 videos/20240527-22.MP4
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


KEYPOINT_NAMES = [
    "rostrum_tip",
    "mid_saddle_patch",
    "Caudal_peduncle",
    "left_pect_fin_tip",
    "right_pect_fin_tip",
    "left_caudal_fluke",
    "right_caudal_fluke",
]

# Skeleton edges (connections between keypoints)
SKELETON_EDGES = [
    ("rostrum_tip", "mid_saddle_patch"),
    ("mid_saddle_patch", "Caudal_peduncle"),
    ("mid_saddle_patch", "left_pect_fin_tip"),
    ("mid_saddle_patch", "right_pect_fin_tip"),
    ("Caudal_peduncle", "left_caudal_fluke"),
    ("Caudal_peduncle", "right_caudal_fluke"),
]

# Colors per keypoint
KP_COLORS = {
    "rostrum_tip":        (0, 255, 0),     # green
    "mid_saddle_patch":   (0, 255, 255),   # yellow
    "Caudal_peduncle":    (255, 165, 0),   # orange
    "left_pect_fin_tip":  (255, 0, 255),   # magenta
    "right_pect_fin_tip": (255, 0, 128),   # pink
    "left_caudal_fluke":  (0, 165, 255),   # light blue
    "right_caudal_fluke": (0, 128, 255),   # blue
}

TRACK_COLORS = [
    (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0),
    (0, 165, 255), (255, 0, 0), (128, 255, 0),
]


def load_tracks(csv_path: Path) -> dict[int, dict[int, dict]]:
    """Load tracks CSV, return {frame: {track_id: {bbox_x1,y1,x2,y2, class_name, confidence}}}."""
    by_frame: dict[int, dict[int, dict]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            tid = int(row["track_id"])
            by_frame.setdefault(fr, {})[tid] = {
                "x1": float(row["bbox_x1"]),
                "y1": float(row["bbox_y1"]),
                "x2": float(row["bbox_x2"]),
                "y2": float(row["bbox_y2"]),
                "class": row.get("class_name", ""),
                "conf": float(row.get("confidence", 0)),
            }
    return by_frame


def load_pose_keypoints(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            kps = {}
            for kp_name in KEYPOINT_NAMES:
                x = row.get(f"{kp_name}_x", "")
                y = row.get(f"{kp_name}_y", "")
                conf = row.get(f"{kp_name}_conf", "")
                if x and y:
                    kps[kp_name] = {
                        "x": float(x), "y": float(y),
                        "conf": float(conf) if conf else 0,
                    }
            rows.append({
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "keypoints": kps,
            })
    return rows


def draw_pose(frame: np.ndarray, keypoints: dict, track_id: int,
              point_size: int = 12, line_width: int = 3):
    """Draw skeleton and keypoints on frame."""
    color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

    # Draw skeleton edges
    for kp_a, kp_b in SKELETON_EDGES:
        if kp_a in keypoints and kp_b in keypoints:
            pt_a = (int(keypoints[kp_a]["x"]), int(keypoints[kp_a]["y"]))
            pt_b = (int(keypoints[kp_b]["x"]), int(keypoints[kp_b]["y"]))
            cv2.line(frame, pt_a, pt_b, color, line_width)

    # Draw keypoints
    for kp_name, kp_data in keypoints.items():
        pt = (int(kp_data["x"]), int(kp_data["y"]))
        kp_color = KP_COLORS.get(kp_name, color)
        cv2.circle(frame, pt, point_size, kp_color, -1)
        cv2.circle(frame, pt, point_size, (255, 255, 255), 2)

        # Label
        label = kp_name.replace("_", " ").replace("tip", "").strip()
        if len(label) > 12:
            label = label[:12]
        cv2.putText(frame, label, (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize pose keypoints on video")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max annotated frames to save (default: 30)")
    args = parser.parse_args()

    pose_csv = args.track_dir / "pose" / "pose_keypoints.csv"
    if not pose_csv.exists():
        print(f"Error: {pose_csv} not found")
        sys.exit(1)

    # Load tracks (bounding boxes) — prefer unified > relinked > raw
    tracks_csv = args.track_dir / "tracks_unified.csv"
    if not tracks_csv.exists():
        tracks_csv = args.track_dir / "tracks_relinked.csv"
    if not tracks_csv.exists():
        tracks_csv = args.track_dir / "tracks.csv"
    tracks_by_frame = {}
    if tracks_csv.exists():
        tracks_by_frame = load_tracks(tracks_csv)
        print(f"Loaded detection boxes from {tracks_csv.name}")

    # Load keypoints
    print(f"Loading keypoints from {pose_csv}...")
    all_kps = load_pose_keypoints(pose_csv)
    print(f"  {len(all_kps)} pose entries")

    # Group by frame
    by_frame: dict[int, list[dict]] = {}
    for kp in all_kps:
        by_frame.setdefault(kp["frame"], []).append(kp)

    # Pick frames to visualize (evenly spaced, prefer frames with most keypoints)
    frames_ranked = sorted(by_frame.keys(),
                           key=lambda f: -sum(len(e["keypoints"]) for e in by_frame[f]))
    selected_frames = sorted(frames_ranked[:args.max_frames])

    # Output dir
    out_dir = args.track_dir / "pose" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read video and annotate
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        sys.exit(1)

    frame_idx = 0
    target_set = set(selected_frames)
    saved = 0

    print(f"  Annotating {len(selected_frames)} frames...")

    while frame_idx <= max(selected_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set:
            annotated = frame.copy()

            # Draw detection bounding boxes
            if frame_idx in tracks_by_frame:
                for tid, det in tracks_by_frame[frame_idx].items():
                    color = TRACK_COLORS[tid % len(TRACK_COLORS)]
                    pt1 = (int(det["x1"]), int(det["y1"]))
                    pt2 = (int(det["x2"]), int(det["y2"]))
                    cv2.rectangle(annotated, pt1, pt2, color, 2)
                    label = f"{det['class']} {det['conf']:.2f}"
                    cv2.putText(annotated, label,
                                (pt1[0], pt1[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for entry in by_frame[frame_idx]:
                draw_pose(annotated, entry["keypoints"], entry["track_id"])

                # Track ID label
                if entry["keypoints"]:
                    first_kp = next(iter(entry["keypoints"].values()))
                    cv2.putText(annotated,
                                f"Track {entry['track_id']}",
                                (int(first_kp["x"]) - 20, int(first_kp["y"]) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                TRACK_COLORS[entry["track_id"] % len(TRACK_COLORS)], 2)

            # Frame label
            cv2.putText(annotated, f"Frame {frame_idx}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            out_path = out_dir / f"pose_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  Saved {saved} annotated frames to {out_dir}/")

    # Generate HTML gallery
    html_path = args.track_dir / "pose" / "pose_review.html"
    frame_files = sorted(out_dir.glob("pose_*.jpg"))

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Pose Estimation Review</title>
<style>
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 20px; }
  h1 { color: #4fc3f7; }
  .grid { display: flex; flex-wrap: wrap; gap: 10px; }
  .grid img { max-width: 720px; border-radius: 4px; cursor: pointer; }
  .grid img:hover { max-width: 100%; }
  .legend { display: flex; gap: 15px; margin: 15px 0; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 14px; }
  .dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; }
</style></head><body>
<h1>Pose Keypoints — """ + args.track_dir.name + """</h1>
<div class="legend">"""

    for kp_name, color in KP_COLORS.items():
        b, g, r = color
        html += f'<div class="legend-item"><div class="dot" style="background:rgb({r},{g},{b})"></div>{kp_name}</div>'

    html += '</div><div class="grid">'

    for fp in frame_files:
        html += f'<img src="frames/{fp.name}" loading="lazy">'

    html += "</div></body></html>"

    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Saved review page to {html_path}")


if __name__ == "__main__":
    main()
