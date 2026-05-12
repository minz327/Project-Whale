"""
POC Experiment 2: Open-vocabulary whale detection using YOLO-World.

Uses text prompts ("whale", "orca", etc.) to detect whales without
any fine-tuning. Compares against the baseline COCO detection.

Usage:
    python scripts/detect_yoloworld.py data/raw_clips/srkw_calf_drone.mp4
    python scripts/detect_yoloworld.py data/raw_clips/srkw_calf_drone.mp4 --conf 0.1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Text prompts to try — these are the "classes" YOLO-World will look for
DEFAULT_CLASSES = [
    "whale",
    "orca",
    "killer whale",
    "dolphin",
    "dorsal fin",
    "marine mammal",
]


def extract_frames(video_path: Path, sample_rate: int = 10) -> list[tuple[int, np.ndarray]]:
    """Extract every Nth frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.1f}, Total frames: {total}, Duration: {total/fps:.1f}s")
    print(f"  Sampling every {sample_rate} frames ({total // sample_rate} frames to process)")

    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % sample_rate == 0:
            frames.append((frame_num, frame))
        frame_num += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames\n")
    return frames


def run_detection(model: YOLO, frames: list[tuple[int, np.ndarray]],
                  conf: float = 0.05) -> list[dict]:
    """Run YOLO-World detection on extracted frames."""
    all_detections = []

    for frame_num, frame in frames:
        results = model(frame, conf=conf, verbose=False)
        result = results[0]

        frame_dets = []
        for box in result.boxes:
            det = {
                "frame": frame_num,
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            }
            frame_dets.append(det)

        all_detections.extend(frame_dets)

        if frame_dets:
            classes = [f"{d['class_name']}({d['confidence']:.2f})" for d in frame_dets]
            print(f"  Frame {frame_num:5d}: {', '.join(classes)}")

    return all_detections


def save_annotated_frames(frames: list[tuple[int, np.ndarray]], detections: list[dict],
                          output_dir: Path):
    """Save frames with detection boxes drawn."""
    det_by_frame = {}
    for d in detections:
        det_by_frame.setdefault(d["frame"], []).append(d)

    # Color map per class
    colors = {
        "whale": (0, 255, 0),
        "orca": (0, 255, 255),
        "killer whale": (255, 255, 0),
        "dolphin": (255, 0, 255),
        "dorsal fin": (0, 165, 255),
        "marine mammal": (255, 128, 0),
    }

    saved = 0
    for frame_num, frame in frames:
        dets = det_by_frame.get(frame_num, [])

        # Save frames with detections + every 30th for context
        if not dets and frame_num % 300 != 0:
            continue

        annotated = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            label = f"{d['class_name']} {d['confidence']:.2f}"
            color = colors.get(d["class_name"], (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tag = "det" if dets else "nodet"
        out_path = output_dir / f"frame_{frame_num:05d}_{tag}.jpg"
        cv2.imwrite(str(out_path), annotated)
        saved += 1

    print(f"  Saved {saved} annotated frames to {output_dir}/")


def write_summary(detections: list[dict], output_dir: Path, video_name: str,
                  total_frames_processed: int, classes_used: list[str]):
    """Write detection summary."""
    class_counts = {}
    for d in detections:
        name = d["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    frames_with_dets = len(set(d["frame"] for d in detections))

    # Per-class confidence stats
    class_confs = {}
    for d in detections:
        class_confs.setdefault(d["class_name"], []).append(d["confidence"])

    conf_stats = {}
    for cls, confs in class_confs.items():
        conf_stats[cls] = {
            "count": len(confs),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3),
            "mean": round(sum(confs) / len(confs), 3),
        }

    summary = {
        "video": video_name,
        "model": "yolov8x-worldv2",
        "classes_prompted": classes_used,
        "total_frames_processed": total_frames_processed,
        "frames_with_detections": frames_with_dets,
        "total_detections": len(detections),
        "class_counts": class_counts,
        "confidence_stats": conf_stats,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if detections:
        with open(output_dir / "detections.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detections[0].keys())
            writer.writeheader()
            writer.writerows(detections)

    print(f"\n{'='*50}")
    print(f"YOLO-World RESULTS: {video_name}")
    print(f"{'='*50}")
    print(f"  Classes prompted:       {classes_used}")
    print(f"  Frames processed:       {total_frames_processed}")
    print(f"  Frames with detections: {frames_with_dets}")
    print(f"  Total detections:       {len(detections)}")
    print(f"  Per-class breakdown:")
    for cls, stats in conf_stats.items():
        print(f"    {cls:20s}: {stats['count']:4d} hits, "
              f"conf {stats['min']:.2f}–{stats['max']:.2f} (avg {stats['mean']:.2f})")
    print(f"  Results saved to:       {output_dir}/")
    print()


def main():
    parser = argparse.ArgumentParser(description="Open-vocab whale detection with YOLO-World")
    parser.add_argument("video", type=Path, help="Path to video clip")
    parser.add_argument("--sample-rate", type=int, default=15,
                        help="Extract every Nth frame (default: 15)")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Confidence threshold (default: 0.05, low to catch everything)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Custom class prompts (default: whale, orca, etc.)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    video_stem = args.video.stem
    output_dir = args.output_dir or Path("outputs/detect_world") / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = args.classes or DEFAULT_CLASSES

    # Load YOLO-World model
    print("Loading YOLO-World model (yolov8x-worldv2)...")
    model = YOLO("yolov8x-worldv2.pt")

    # Set custom classes
    print(f"  Setting class prompts: {classes}\n")
    model.set_classes(classes)

    # Extract frames
    print("Extracting frames...")
    frames = extract_frames(args.video, args.sample_rate)

    # Run detection
    print("Running open-vocabulary detection...")
    detections = run_detection(model, frames, args.conf)

    # Save annotated frames
    print("\nSaving annotated frames...")
    save_annotated_frames(frames, detections, output_dir)

    # Write summary
    write_summary(detections, output_dir, video_stem, len(frames), classes)


if __name__ == "__main__":
    main()
