"""
Run SLEAP pose estimation on whale detection crops.

Uses Ren's pre-trained centered_instance model to extract 7 keypoints
(rostrum, saddle patch, peduncle, flukes, pectoral fins) from
whale bounding box crops produced by the detection pipeline.

Requires the SLEAP environment (.venv_sleap) to be active.

Usage:
    # From the SLEAP venv:
    python scripts/pose_estimate.py outputs/track/20240527-22 videos/20240527-22.MP4

    # Or with specific model:
    python scripts/pose_estimate.py outputs/track/20240527-22 videos/20240527-22.MP4 \
        --model exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import sleap
except ImportError:
    print("Error: SLEAP not installed. Run from the SLEAP venv (.venv_sleap).")
    print("Setup: python -m venv .venv_sleap && .venv_sleap\\Scripts\\activate && pip install sleap[pypi]")
    sys.exit(1)


KEYPOINT_NAMES = [
    "rostrum_tip",
    "mid_saddle_patch",
    "Caudal_peduncle",
    "left_pect_fin_tip",
    "right_pect_fin_tip",
    "left_caudal_fluke",
    "right_caudal_fluke",
]

# Default model path (Ren's best model)
DEFAULT_MODEL = Path("exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231")


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


def extract_crops(video_path: Path, tracks: list[dict],
                  pad_ratio: float = 0.2) -> dict[tuple[int, int], np.ndarray]:
    """Extract padded whale crops from the video for each (frame, track_id) pair."""
    # Group by frame
    by_frame: dict[int, list[dict]] = {}
    for t in tracks:
        by_frame.setdefault(t["frame"], []).append(t)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    crops = {}
    target_frames = sorted(by_frame.keys())
    frame_idx = 0
    target_idx = 0

    while target_idx < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_frames[target_idx]:
            for t in by_frame[frame_idx]:
                # Add padding around bbox
                bw = t["bbox_x2"] - t["bbox_x1"]
                bh = t["bbox_y2"] - t["bbox_y1"]
                pad_x = bw * pad_ratio
                pad_y = bh * pad_ratio

                x1 = max(0, int(t["bbox_x1"] - pad_x))
                y1 = max(0, int(t["bbox_y1"] - pad_y))
                x2 = min(w_vid, int(t["bbox_x2"] + pad_x))
                y2 = min(h_vid, int(t["bbox_y2"] + pad_y))

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops[(frame_idx, t["track_id"])] = {
                        "image": crop,
                        "offset_x": x1,
                        "offset_y": y1,
                    }
            target_idx += 1

        frame_idx += 1

    cap.release()
    return crops


def run_pose_inference(model_path: Path, crops: dict) -> list[dict]:
    """Run SLEAP inference on cropped whale images."""
    # Load the SLEAP model
    print(f"  Loading SLEAP model from {model_path}...")
    predictor = sleap.load_model(str(model_path))

    results = []
    total = len(crops)

    for i, ((frame_num, track_id), crop_data) in enumerate(sorted(crops.items())):
        img = crop_data["image"]
        offset_x = crop_data["offset_x"]
        offset_y = crop_data["offset_y"]

        # Run prediction on the crop
        predictions = predictor.inference(img)

        if predictions and len(predictions) > 0:
            # Get the first (highest confidence) instance
            pred = predictions[0]
            keypoints = {}

            for node_idx, node_name in enumerate(KEYPOINT_NAMES):
                if node_idx < pred.shape[0]:
                    x, y = pred[node_idx]
                    if not np.isnan(x) and not np.isnan(y):
                        # Convert back to full-frame coordinates
                        keypoints[node_name] = {
                            "x": float(x + offset_x),
                            "y": float(y + offset_y),
                            "x_crop": float(x),
                            "y_crop": float(y),
                        }

            if keypoints:
                results.append({
                    "frame": frame_num,
                    "track_id": track_id,
                    "keypoints": keypoints,
                    "n_keypoints_found": len(keypoints),
                })

        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{total} crops...")

    return results


def save_results(results: list[dict], output_dir: Path):
    """Save pose estimation results."""
    # Flat CSV
    rows = []
    for r in results:
        row = {"frame": r["frame"], "track_id": r["track_id"]}
        for kp_name, kp_data in r["keypoints"].items():
            row[f"{kp_name}_x"] = kp_data["x"]
            row[f"{kp_name}_y"] = kp_data["y"]
        rows.append(row)

    if rows:
        csv_path = output_dir / "pose_keypoints.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved keypoints to {csv_path}")

    # JSON with full detail
    json_path = output_dir / "pose_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved detailed results to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SLEAP pose estimation on whale crops")
    parser.add_argument("track_dir", type=Path,
                        help="Directory with tracks.csv or tracks_relinked.csv")
    parser.add_argument("video", type=Path, help="Source video file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to SLEAP model directory")
    parser.add_argument("--csv", type=str, default="tracks_relinked.csv",
                        help="Track CSV to use (default: tracks_relinked.csv)")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Only process tracks with ≥N frames")
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        # Fall back to tracks.csv
        csv_path = args.track_dir / "tracks.csv"
    if not csv_path.exists():
        print(f"Error: No track CSV found in {args.track_dir}")
        sys.exit(1)

    # Load tracks
    print(f"Loading tracks from {csv_path}...")
    tracks = load_tracks(csv_path)

    # Filter short tracks
    by_id: dict[int, list] = {}
    for t in tracks:
        by_id.setdefault(t["track_id"], []).append(t)
    long_ids = {tid for tid, pts in by_id.items() if len(pts) >= args.min_frames}
    tracks = [t for t in tracks if t["track_id"] in long_ids]
    print(f"  {len(tracks)} points from {len(long_ids)} tracks")

    # Extract crops
    print(f"\nExtracting whale crops from {args.video}...")
    crops = extract_crops(args.video, tracks)
    print(f"  {len(crops)} crops extracted")

    # Run pose inference
    print(f"\nRunning SLEAP pose estimation...")
    results = run_pose_inference(args.model, crops)
    print(f"  {len(results)} frames with keypoints detected")

    if results:
        avg_kp = np.mean([r["n_keypoints_found"] for r in results])
        print(f"  Average keypoints per frame: {avg_kp:.1f} / {len(KEYPOINT_NAMES)}")

    # Save
    print(f"\nSaving results...")
    pose_dir = args.track_dir / "pose"
    pose_dir.mkdir(exist_ok=True)
    save_results(results, pose_dir)

    print(f"\n{'='*55}")
    print(f"POSE ESTIMATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Crops processed: {len(crops)}")
    print(f"  Keypoints found: {len(results)} frames")
    if results:
        print(f"  Avg keypoints/frame: {avg_kp:.1f}")
    print(f"  Results: {pose_dir}/")
    print()


if __name__ == "__main__":
    main()
