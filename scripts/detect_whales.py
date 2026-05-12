"""
POC Experiment 1: Whale detection on drone clips.

Extracts frames from a video clip, runs YOLOv8 detection,
and saves annotated frames + a summary of what was detected.

Usage:
    python scripts/detect_whales.py data/raw_clips/srkw_calf_drone.mp4
    python scripts/detect_whales.py data/raw_clips/srkw_calf_drone.mp4 --sample-rate 5 --conf 0.3
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def extract_frames(video_path: Path, sample_rate: int = 10) -> list[tuple[int, np.ndarray]]:
    """Extract every Nth frame from a video. Returns list of (frame_number, frame)."""
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
                  conf: float = 0.25) -> list[dict]:
    """Run YOLO detection on extracted frames. Returns detection results."""
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
            classes = [d["class_name"] for d in frame_dets]
            print(f"  Frame {frame_num:5d}: {len(frame_dets)} detection(s) — {', '.join(classes)}")

    return all_detections


def save_annotated_frames(frames: list[tuple[int, np.ndarray]], detections: list[dict],
                          output_dir: Path, model: YOLO):
    """Save frames with detection boxes drawn on them."""
    # Group detections by frame
    det_by_frame = {}
    for d in detections:
        det_by_frame.setdefault(d["frame"], []).append(d)

    # Only save frames that have detections, plus a few without (for comparison)
    saved = 0
    for frame_num, frame in frames:
        dets = det_by_frame.get(frame_num, [])

        # Save frames with detections, or every 30th frame for context
        if not dets and frame_num % 300 != 0:
            continue

        annotated = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            label = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        tag = "det" if dets else "nodet"
        out_path = output_dir / f"frame_{frame_num:05d}_{tag}.jpg"
        cv2.imwrite(str(out_path), annotated)
        saved += 1

    print(f"  Saved {saved} annotated frames to {output_dir}/")


def write_summary(detections: list[dict], output_dir: Path, video_name: str,
                  total_frames_processed: int):
    """Write detection summary as JSON and CSV."""
    # Class frequency
    class_counts = {}
    for d in detections:
        name = d["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    frames_with_dets = len(set(d["frame"] for d in detections))

    summary = {
        "video": video_name,
        "total_frames_processed": total_frames_processed,
        "frames_with_detections": frames_with_dets,
        "total_detections": len(detections),
        "class_counts": class_counts,
        "avg_confidence": (sum(d["confidence"] for d in detections) / len(detections)
                           if detections else 0),
    }

    # JSON summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV of all detections
    if detections:
        with open(output_dir / "detections.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detections[0].keys())
            writer.writeheader()
            writer.writerows(detections)

    print(f"\n{'='*50}")
    print(f"SUMMARY: {video_name}")
    print(f"{'='*50}")
    print(f"  Frames processed:       {total_frames_processed}")
    print(f"  Frames with detections: {frames_with_dets}")
    print(f"  Total detections:       {len(detections)}")
    print(f"  Classes found:          {class_counts}")
    if detections:
        print(f"  Avg confidence:         {summary['avg_confidence']:.3f}")
    print(f"  Results saved to:       {output_dir}/")
    print()


def main():
    parser = argparse.ArgumentParser(description="POC whale detection on drone clips")
    parser.add_argument("video", type=Path, help="Path to video clip")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model to use (default: yolov8n.pt)")
    parser.add_argument("--sample-rate", type=int, default=10,
                        help="Extract every Nth frame (default: 10)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: outputs/detect/<video_name>/)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Output directory
    video_stem = args.video.stem
    output_dir = args.output_dir or Path("outputs/detect") / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"  Classes: {len(model.names)} COCO classes\n")

    # Extract frames
    print("Extracting frames...")
    frames = extract_frames(args.video, args.sample_rate)

    # Run detection
    print("Running detection...")
    detections = run_detection(model, frames, args.conf)

    # Save annotated frames
    print("\nSaving annotated frames...")
    save_annotated_frames(frames, detections, output_dir, model)

    # Write summary
    write_summary(detections, output_dir, video_stem, len(frames))


if __name__ == "__main__":
    main()
