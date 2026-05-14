"""
Run SLEAP pose estimation on whale detections using TensorFlow directly.

Loads Ren's pre-trained centered_instance model (.h5) with TensorFlow/Keras
and runs inference on whale detection crops. No SLEAP CLI needed.

The model expects 832x832 RGB crops and outputs 208x208x7 confidence maps
(one per keypoint). We extract peak locations from each map.

Requires the SLEAP venv (.venv_sleap) with tensorflow-cpu installed.

Usage:
    python scripts/pose_estimate.py outputs/track/20240527-22 videos/20240527-22.MP4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed. Run from .venv_sleap with tensorflow-cpu.")
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

MODEL_PATH = Path("exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231/best_model.h5")
MODEL_INPUT_SIZE = 832  # pixels


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


def prepare_crop(frame: np.ndarray, bbox: dict,
                 target_size: int = MODEL_INPUT_SIZE) -> tuple[np.ndarray, dict]:
    """Extract and resize a whale crop to model input size.

    Returns the preprocessed crop and metadata for mapping back to full frame.
    """
    h, w = frame.shape[:2]

    # Use bbox center, expand to square
    cx = (bbox["bbox_x1"] + bbox["bbox_x2"]) / 2
    cy = (bbox["bbox_y1"] + bbox["bbox_y2"]) / 2
    bw = bbox["bbox_x2"] - bbox["bbox_x1"]
    bh = bbox["bbox_y2"] - bbox["bbox_y1"]

    # Make square crop with padding (1.5x the longer bbox dimension)
    side = max(bw, bh) * 1.5
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    # Resize to model input
    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, (target_size, target_size))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1 float
    preprocessed = resized_rgb.astype(np.float32) / 255.0

    meta = {
        "crop_x1": x1, "crop_y1": y1,
        "crop_x2": x2, "crop_y2": y2,
        "crop_w": crop_w, "crop_h": crop_h,
        "scale_x": crop_w / target_size,
        "scale_y": crop_h / target_size,
    }
    return preprocessed, meta


def find_peaks(confidence_maps: np.ndarray, threshold: float = 0.1) -> list[dict | None]:
    """Find peak location in each confidence map (one per keypoint).

    confidence_maps shape: (H, W, n_keypoints)
    Returns list of {x, y, confidence} or None per keypoint.
    """
    peaks = []
    for i in range(confidence_maps.shape[-1]):
        cmap = confidence_maps[:, :, i]
        max_val = np.max(cmap)
        if max_val < threshold:
            peaks.append(None)
            continue

        max_idx = np.unravel_index(np.argmax(cmap), cmap.shape)
        # Convert from confidence map coords to input image coords
        # Model outputs at 1/4 resolution (832 -> 208)
        scale = MODEL_INPUT_SIZE / cmap.shape[0]
        y_px = max_idx[0] * scale
        x_px = max_idx[1] * scale

        peaks.append({"x": float(x_px), "y": float(y_px), "confidence": float(max_val)})

    return peaks


def main():
    parser = argparse.ArgumentParser(description="Run SLEAP pose estimation on whale tracks")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--csv", type=str, default="tracks_relinked.csv")
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--peak-threshold", type=float, default=0.1,
                        help="Min confidence to accept a keypoint (default: 0.1)")
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        csv_path = args.track_dir / "tracks.csv"
    if not csv_path.exists():
        print(f"Error: No track CSV found in {args.track_dir}")
        sys.exit(1)

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(str(args.model), compile=False)
    print(f"  Input: {model.input_shape}, Output: {model.output_shape}")

    # Load tracks
    print(f"\nLoading tracks from {csv_path}...")
    tracks = load_tracks(csv_path)
    by_id = {}
    for t in tracks:
        by_id.setdefault(t["track_id"], []).append(t)
    long_ids = {tid for tid, pts in by_id.items() if len(pts) >= args.min_frames}
    tracks = [t for t in tracks if t["track_id"] in long_ids]
    print(f"  {len(tracks)} points from {len(long_ids)} tracks")

    # Group by frame for efficient video reading
    by_frame: dict[int, list[dict]] = {}
    for t in tracks:
        by_frame.setdefault(t["frame"], []).append(t)

    # Process video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    target_frames = sorted(by_frame.keys())
    frame_idx = 0
    target_idx = 0
    results = []
    processed = 0

    print(f"\nRunning pose estimation on {len(target_frames)} frames...")

    while target_idx < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_frames[target_idx]:
            for t in by_frame[frame_idx]:
                crop, meta = prepare_crop(frame, t)
                if crop is None:
                    continue

                # Run inference
                batch = np.expand_dims(crop, axis=0)
                pred = model.predict(batch, verbose=0)
                conf_maps = pred[0]  # (H, W, 7)

                # Find keypoint peaks
                peaks = find_peaks(conf_maps, args.peak_threshold)

                keypoints = {}
                for kp_name, peak in zip(KEYPOINT_NAMES, peaks):
                    if peak is None:
                        continue
                    # Map from crop coords to full-frame coords
                    full_x = peak["x"] * meta["scale_x"] + meta["crop_x1"]
                    full_y = peak["y"] * meta["scale_y"] + meta["crop_y1"]
                    keypoints[kp_name] = {
                        "x": round(full_x, 1),
                        "y": round(full_y, 1),
                        "confidence": round(peak["confidence"], 4),
                    }

                if keypoints:
                    results.append({
                        "frame": frame_idx,
                        "track_id": t["track_id"],
                        "keypoints": keypoints,
                        "n_keypoints_found": len(keypoints),
                    })

                processed += 1
                if processed % 20 == 0:
                    print(f"  Processed {processed}/{len(tracks)} crops...")

            target_idx += 1
        frame_idx += 1

    cap.release()

    # Save results
    pose_dir = args.track_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Flat CSV
    rows = []
    for r in results:
        row = {"frame": r["frame"], "track_id": r["track_id"],
               "n_keypoints": r["n_keypoints_found"]}
        for kp_name in KEYPOINT_NAMES:
            kp = r["keypoints"].get(kp_name)
            row[f"{kp_name}_x"] = kp["x"] if kp else ""
            row[f"{kp_name}_y"] = kp["y"] if kp else ""
            row[f"{kp_name}_conf"] = kp["confidence"] if kp else ""
        rows.append(row)

    if rows:
        csv_out = pose_dir / "pose_keypoints.csv"
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved keypoints to {csv_out}")

    json_out = pose_dir / "pose_results.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved detailed results to {json_out}")

    # Summary
    if results:
        avg_kp = np.mean([r["n_keypoints_found"] for r in results])
        kp_counts = {}
        for r in results:
            for kp_name in r["keypoints"]:
                kp_counts[kp_name] = kp_counts.get(kp_name, 0) + 1

        print(f"\n{'='*55}")
        print(f"POSE ESTIMATION COMPLETE")
        print(f"{'='*55}")
        print(f"  Crops processed:  {processed}")
        print(f"  With keypoints:   {len(results)}")
        print(f"  Avg keypoints:    {avg_kp:.1f} / {len(KEYPOINT_NAMES)}")
        print(f"\n  Per-keypoint detection rate:")
        for kp_name in KEYPOINT_NAMES:
            count = kp_counts.get(kp_name, 0)
            pct = count / len(results) * 100 if results else 0
            print(f"    {kp_name:25s}: {count:4d} ({pct:.0f}%)")
        print(f"\n  Results: {pose_dir}/")
    else:
        print("\n  No keypoints detected. Model may not suit this footage.")

    print()


if __name__ == "__main__":
    main()
