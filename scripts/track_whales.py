"""
POC Experiment 3: Multi-object whale tracking.

Takes a video, runs YOLO-World detection per frame with cross-class NMS,
then feeds clean detections to ByteTrack for persistent ID assignment.

Outputs:
  - Annotated video with track IDs and trail lines
  - tracks.csv with per-frame track data
  - summary.json with track statistics

Usage:
    python scripts/track_whales.py exeter/20240527-22.MP4
    python scripts/track_whales.py exeter/20240527-22.MP4 --sample-rate 3 --conf 0.1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


DEFAULT_CLASSES = [
    "whale",
    "orca",
    "killer whale",
    "dolphin",
    "dorsal fin",
    "marine mammal",
]

# Track colors (up to 10 whales)
TRACK_COLORS = [
    (0, 255, 0),    # green
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
    (0, 165, 255),  # light blue
    (255, 0, 0),    # red
    (128, 255, 0),  # lime
    (255, 128, 128),# pink
    (0, 128, 255),  # sky blue
    (128, 0, 255),  # purple
]


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def cross_class_nms(detections: sv.Detections, iou_thresh: float = 0.5) -> sv.Detections:
    """Merge overlapping detections across classes, keeping highest confidence."""
    if len(detections) == 0:
        return detections

    # Sort by confidence descending
    order = np.argsort(-detections.confidence)
    keep = []

    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j in suppressed or j == i:
                continue
            if compute_iou(detections.xyxy[i], detections.xyxy[j]) >= iou_thresh:
                suppressed.add(j)

    if not keep:
        return sv.Detections.empty()

    return detections[keep]


def main():
    parser = argparse.ArgumentParser(description="Track whales with ByteTrack")
    parser.add_argument("video", type=Path, help="Path to video clip")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Process every Nth frame (default: 3, ~10fps for 30fps video)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold (default: 0.1)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for cross-class NMS (default: 0.5)")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    video_stem = args.video.stem
    output_dir = args.output_dir or Path("outputs/track") / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = args.classes or DEFAULT_CLASSES

    # Load model
    print("Loading YOLO-World model (yolov8x-worldv2)...")
    model = YOLO("yolov8x-worldv2.pt")
    model.set_classes(classes)
    print(f"  Class prompts: {classes}")

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps / args.sample_rate

    print(f"\nVideo: {args.video.name}")
    print(f"  {width}x{height} @ {fps:.1f}fps, {total_frames} frames, "
          f"{total_frames/fps:.1f}s")
    print(f"  Processing every {args.sample_rate} frames → ~{effective_fps:.1f} effective fps")
    print(f"  Confidence threshold: {args.conf}")

    # Set up tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=args.conf,
        lost_track_buffer=30,       # keep ID alive for 30 frames without detection
        minimum_matching_threshold=0.8,
        frame_rate=int(effective_fps),
    )

    # Set up video writer for output
    out_video_path = output_dir / f"{video_stem}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps,
                                (width, height))

    # Track state
    all_tracks: list[dict] = []
    track_trails: dict[int, list[tuple[int, int]]] = {}  # track_id -> list of center points
    frame_num = 0
    processed = 0

    print(f"\nRunning detection + tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % args.sample_rate != 0:
            frame_num += 1
            continue

        # Detect
        results = model(frame, conf=args.conf, verbose=False)[0]

        # Convert to supervision Detections
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int),
        )

        # Cross-class NMS
        detections = cross_class_nms(detections, args.iou_thresh)

        # Update tracker
        tracked = tracker.update_with_detections(detections)

        # Store track data
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            track_id = int(tracked.tracker_id[i])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Update trails
            if track_id not in track_trails:
                track_trails[track_id] = []
            track_trails[track_id].append((cx, cy))

            class_name = results.names[int(tracked.class_id[i])] if tracked.class_id is not None else "unknown"
            all_tracks.append({
                "frame": frame_num,
                "track_id": track_id,
                "class_name": class_name,
                "confidence": float(tracked.confidence[i]),
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "center_x": cx,
                "center_y": cy,
            })

        # Draw annotations on frame
        annotated = frame.copy()

        for i in range(len(tracked)):
            x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
            track_id = int(tracked.tracker_id[i])
            conf = float(tracked.confidence[i])
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with track ID
            label = f"ID:{track_id} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw trail
            trail = track_trails.get(track_id, [])
            if len(trail) > 1:
                for t in range(1, len(trail)):
                    alpha = t / len(trail)  # fade older points
                    thickness = max(1, int(3 * alpha))
                    pt1 = trail[t - 1]
                    pt2 = trail[t]
                    cv2.line(annotated, pt1, pt2, color, thickness)

        # Frame counter
        cv2.putText(annotated, f"Frame {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_video.write(annotated)
        processed += 1

        # Progress
        if processed % 20 == 0:
            n_active = len(tracked) if len(tracked) > 0 else 0
            print(f"  Frame {frame_num:5d}/{total_frames} — "
                  f"{n_active} active tracks, "
                  f"{len(track_trails)} total IDs seen")

        frame_num += 1

    cap.release()
    out_video.release()

    # Summary stats
    unique_tracks = set(t["track_id"] for t in all_tracks)
    frames_with_tracks = len(set(t["frame"] for t in all_tracks))

    # Per-track stats
    track_stats = {}
    for tid in sorted(unique_tracks):
        tid_data = [t for t in all_tracks if t["track_id"] == tid]
        frames_present = [t["frame"] for t in tid_data]
        track_stats[str(tid)] = {
            "frames": len(tid_data),
            "first_frame": min(frames_present),
            "last_frame": max(frames_present),
            "span": max(frames_present) - min(frames_present),
            "avg_confidence": round(np.mean([t["confidence"] for t in tid_data]), 3),
        }

    summary = {
        "video": video_stem,
        "total_frames_processed": processed,
        "sample_rate": args.sample_rate,
        "confidence_threshold": args.conf,
        "frames_with_tracks": frames_with_tracks,
        "unique_track_ids": len(unique_tracks),
        "total_track_points": len(all_tracks),
        "track_stats": track_stats,
    }

    # Save outputs
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if all_tracks:
        with open(output_dir / "tracks.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_tracks[0].keys())
            writer.writeheader()
            writer.writerows(all_tracks)

    print(f"\n{'='*55}")
    print(f"TRACKING RESULTS: {video_stem}")
    print(f"{'='*55}")
    print(f"  Frames processed:    {processed}")
    print(f"  Frames with tracks:  {frames_with_tracks}")
    print(f"  Unique track IDs:    {len(unique_tracks)}")
    print(f"  Total track points:  {len(all_tracks)}")
    print(f"\n  Per-track breakdown:")
    for tid, stats in track_stats.items():
        print(f"    Track {tid:>3s}: {stats['frames']:4d} frames, "
              f"span {stats['first_frame']}→{stats['last_frame']}, "
              f"avg conf {stats['avg_confidence']:.2f}")
    print(f"\n  Output video: {out_video_path}")
    print(f"  Track data:   {output_dir / 'tracks.csv'}")
    print()


if __name__ == "__main__":
    main()
