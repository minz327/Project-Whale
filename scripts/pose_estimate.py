"""
Run SLEAP pose estimation on whale detections using TensorFlow directly.

Uses Ren's two-stage top-down pipeline:
  1. Centroid model — finds the precise saddle patch center within a YOLO detection crop
  2. Centered instance model — predicts 7 keypoints on a crop centered on the saddle patch

This matches how SLEAP's top-down inference works: centroid detection first,
then instance-level keypoint prediction anchored on the centroid.

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


# Order must match training_config.json → model.heads.centered_instance.part_names
KEYPOINT_NAMES = [
    "rostrum_tip",
    "left_caudal_fluke",
    "right_caudal_fluke",
    "left_pect_fin_tip",
    "right_pect_fin_tip",
    "mid_saddle_patch",
    "Caudal_peduncle",
]

MODEL_PATH = Path("exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231/best_model.h5")
CENTROID_MODEL_PATH = Path("exeter/models/full_FGM_v1_250327_234344.centroid.n=231/best_model.h5")
INSTANCE_INPUT_SIZE = 832  # centered_instance model input size


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


def find_centroid(bbox: dict, frame_scale_x: float, frame_scale_y: float,
                  centroid_map: np.ndarray) -> tuple[float, float] | None:
    """Find the centroid peak closest to the YOLO bbox center from a pre-computed centroid map.

    The centroid_map is produced by running the centroid model on the full
    downscaled frame (once per frame). We find the peak nearest to each
    YOLO detection's center.
    """
    # YOLO bbox center in centroid-map coords
    cx = (bbox["bbox_x1"] + bbox["bbox_x2"]) / 2.0 * frame_scale_x
    cy = (bbox["bbox_y1"] + bbox["bbox_y2"]) / 2.0 * frame_scale_y

    # centroid_map output_stride=2, so map is (H/2, W/2)
    map_cx = cx / 2.0
    map_cy = cy / 2.0

    # Search region: within 150 map-pixels of bbox center
    search_radius = 150
    map_h, map_w = centroid_map.shape
    y1 = max(0, int(map_cy - search_radius))
    y2 = min(map_h, int(map_cy + search_radius))
    x1 = max(0, int(map_cx - search_radius))
    x2 = min(map_w, int(map_cx + search_radius))

    region = centroid_map[y1:y2, x1:x2]
    if region.size == 0:
        return None

    max_val = np.max(region)
    if max_val < 0.05:
        return None

    max_idx = np.unravel_index(np.argmax(region), region.shape)
    # Map back to full-frame coords: map_pixel * output_stride / frame_scale
    peak_x = (x1 + max_idx[1]) * 2.0 / frame_scale_x
    peak_y = (y1 + max_idx[0]) * 2.0 / frame_scale_y

    return (peak_x, peak_y)


def prepare_centered_crop(frame: np.ndarray, center_x: float, center_y: float,
                          target_size: int = INSTANCE_INPUT_SIZE) -> tuple[np.ndarray, dict]:
    """Extract a square crop centered on the centroid (saddle patch) position.

    This is what the centered_instance model expects — the whale centered on
    its anchor keypoint (mid_saddle_patch).
    """
    h, w = frame.shape[:2]
    half = target_size / 2.0

    # The crop in full-frame pixel space
    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x2 = x1 + target_size
    y2 = y1 + target_size

    # Handle boundary: pad if crop goes outside frame
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Clamp to frame bounds
    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(w, x2)
    fy2 = min(h, y2)

    crop = frame[fy1:fy2, fx1:fx2]
    if crop.size == 0:
        return None, None

    # Pad if needed
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = np.pad(crop,
                      ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode="constant")

    # Ensure exact target size (edge cases from rounding)
    if crop.shape[0] != target_size or crop.shape[1] != target_size:
        crop = cv2.resize(crop, (target_size, target_size))

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    preprocessed = crop_rgb.astype(np.float32) / 255.0

    meta = {
        "crop_x1": x1, "crop_y1": y1,  # can be negative (padding)
        "frame_x1": fx1, "frame_y1": fy1,
        "pad_left": pad_left, "pad_top": pad_top,
        "scale": 1.0,  # no resizing — 1:1 pixel mapping
    }
    return preprocessed, meta


def find_peaks(confidence_maps: np.ndarray, threshold: float = 0.1) -> list[dict | None]:
    """Find peak location in each confidence map (one per keypoint).

    confidence_maps shape: (H, W, n_keypoints)
    Returns list of {x, y, confidence} or None per keypoint.
    x, y are in the 832x832 crop pixel space (output_stride=4).
    """
    peaks = []
    output_stride = INSTANCE_INPUT_SIZE / confidence_maps.shape[0]  # should be 4
    for i in range(confidence_maps.shape[-1]):
        cmap = confidence_maps[:, :, i]
        max_val = np.max(cmap)
        if max_val < threshold:
            peaks.append(None)
            continue

        max_idx = np.unravel_index(np.argmax(cmap), cmap.shape)
        y_px = max_idx[0] * output_stride
        x_px = max_idx[1] * output_stride

        peaks.append({"x": float(x_px), "y": float(y_px), "confidence": float(max_val)})

    return peaks


def main():
    parser = argparse.ArgumentParser(description="Run SLEAP pose estimation on whale tracks")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--centroid-model", type=Path, default=CENTROID_MODEL_PATH)
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
    if not args.centroid_model.exists():
        print(f"Error: Centroid model not found: {args.centroid_model}")
        sys.exit(1)

    # Load models (two-stage top-down pipeline)
    print(f"Loading centroid model from {args.centroid_model}...")
    centroid_model = tf.keras.models.load_model(str(args.centroid_model), compile=False)
    print(f"  Input: {centroid_model.input_shape}, Output: {centroid_model.output_shape}")

    print(f"Loading centered instance model from {args.model}...")
    instance_model = tf.keras.models.load_model(str(args.model), compile=False)
    print(f"  Input: {instance_model.input_shape}, Output: {instance_model.output_shape}")

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

    # Derive centroid input dimensions from model shape or frame size
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, test_frame = cap.read()
    frame_h, frame_w = test_frame.shape[:2]

    centroid_shape = centroid_model.input_shape  # (batch, H, W, C)
    if centroid_shape[1] is not None and centroid_shape[2] is not None:
        centroid_h, centroid_w = centroid_shape[1], centroid_shape[2]
    else:
        # Fully convolutional: scale to ~0.5x, rounded to model stride (16)
        stride = 16
        centroid_h = int(round(frame_h * 0.5 / stride)) * stride
        centroid_w = int(round(frame_w * 0.5 / stride)) * stride

    frame_scale_x = centroid_w / frame_w
    frame_scale_y = centroid_h / frame_h
    print(f"  Frame: {frame_w}x{frame_h} → centroid input: {centroid_w}x{centroid_h}")
    print(f"  Scale: x={frame_scale_x:.4f}, y={frame_scale_y:.4f}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fallback_count = 0

    while target_idx < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_frames[target_idx]:
            # Stage 1: Run centroid model ONCE on the full downscaled frame
            ds_frame = cv2.resize(frame, (centroid_w, centroid_h))
            ds_rgb = cv2.cvtColor(ds_frame, cv2.COLOR_BGR2RGB)
            ds_inp = ds_rgb.astype(np.float32) / 255.0
            ds_inp = np.expand_dims(ds_inp, axis=0)
            centroid_pred = centroid_model.predict(ds_inp, verbose=0)
            centroid_map = centroid_pred[0, :, :, 0]  # (680, 1280)

            for t in by_frame[frame_idx]:
                # Find precise centroid for this detection
                centroid = find_centroid(
                    t, frame_scale_x, frame_scale_y, centroid_map,
                )
                if centroid is None:
                    fallback_count += 1
                    centroid = (
                        (t["bbox_x1"] + t["bbox_x2"]) / 2,
                        (t["bbox_y1"] + t["bbox_y2"]) / 2,
                    )

                cent_x, cent_y = centroid

                # Stage 2: Crop centered on centroid, run instance model
                crop, meta = prepare_centered_crop(frame, cent_x, cent_y)
                if crop is None:
                    continue

                batch = np.expand_dims(crop, axis=0)
                pred = instance_model.predict(batch, verbose=0)
                conf_maps = pred[0]  # (208, 208, 7)

                # Find keypoint peaks
                peaks = find_peaks(conf_maps, args.peak_threshold)

                keypoints = {}
                for kp_name, peak in zip(KEYPOINT_NAMES, peaks):
                    if peak is None:
                        continue
                    # Map from 832x832 crop coords to full-frame coords
                    # crop_x1/crop_y1 is the top-left of the 832x832 region
                    full_x = peak["x"] + meta["crop_x1"]
                    full_y = peak["y"] + meta["crop_y1"]
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
        print(f"  Centroid fallbacks: {fallback_count}/{processed}")
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
