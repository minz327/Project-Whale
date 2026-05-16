"""
Whale tracking with Norfair (appearance-aware, camera-motion-compensated).

Two-tier matching architecture:
  1. Short-term: IoU-based distance for frame-to-frame tracking
  2. Long-term ReID: ResNet18 appearance embeddings for re-identification
     after dives (up to ~15s gap)

Also integrates:
  - Camera motion compensation via Norfair's MotionEstimator
  - Frame skipping via Norfair's `period` parameter (correct Kalman dt)
  - Cross-class NMS for YOLO-World multi-label detections

Outputs:
  - Annotated video with track IDs and trail lines
  - tracks.csv with per-frame track data
  - summary.json with track statistics

Usage:
    python scripts/track_whales.py videos/20231018-40_trim.mp4
    python scripts/track_whales.py videos/20231018-40_trim.mp4 --sample-rate 3 --conf 0.1
    python scripts/track_whales.py videos/20231018-40_trim.mp4 --reid-threshold 0.4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from norfair import Detection, Tracker
from norfair.camera_motion import MotionEstimator, HomographyTransformationGetter
from norfair.filter import OptimizedKalmanFilterFactory
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


# ---------------------------------------------------------------------------
# Embedding extractor (ResNet18 feature vectors for whale appearance)
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """Extract 512-dim appearance embeddings from whale bounding box crops."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove classification head — use as feature extractor (512-dim output)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """Extract embedding from a single bounding box crop.

        Args:
            frame: Full BGR frame (H, W, 3)
            bbox_xyxy: [x1, y1, x2, y2] bounding box

        Returns:
            512-dim numpy feature vector (L2-normalized)
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox_xyxy[0]))
        y1 = max(0, int(bbox_xyxy[1]))
        x2 = min(w, int(bbox_xyxy[2]))
        y2 = min(h, int(bbox_xyxy[3]))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze().cpu().numpy()
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat


# ---------------------------------------------------------------------------
# Distance functions for Norfair
# ---------------------------------------------------------------------------

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


def iou_distance(detection: Detection, tracked_object) -> float:
    """IoU-based distance for short-term frame-to-frame matching.

    Norfair minimizes distance, so we return (1 - IoU).
    Detection.points is [[x1,y1],[x2,y2]], tracked_object.estimate is same shape.
    """
    det_box = detection.points.flatten()  # [x1, y1, x2, y2]
    est_box = tracked_object.estimate.flatten()  # [x1, y1, x2, y2]
    iou = compute_iou(det_box, est_box)
    return 1.0 - iou


def make_reid_distance(embedding_weight: float = 1.0):
    """Create a ReID distance function that compares appearance embeddings.

    This is called when a tracked object has been 'dead' (no short-term match)
    and a new uninitialized tracked object appears — Norfair checks if they
    should be merged via this function.
    """
    def reid_distance(new_tracked_obj, old_tracked_obj) -> float:
        # Collect embeddings from past detections
        new_embs = [d.embedding for d in new_tracked_obj.past_detections
                     if d.embedding is not None]
        old_embs = [d.embedding for d in old_tracked_obj.past_detections
                     if d.embedding is not None]

        if not new_embs or not old_embs:
            return 1.0  # No embeddings → max distance (no match)

        new_mean = np.mean(new_embs, axis=0)
        old_mean = np.mean(old_embs, axis=0)

        # Cosine similarity (embeddings are L2-normalized, so dot = cosine)
        similarity = float(np.dot(new_mean, old_mean))
        return 1.0 - max(0.0, similarity)  # distance = 1 - similarity

    return reid_distance


def cross_class_nms_boxes(boxes_xyxy: np.ndarray, confidences: np.ndarray,
                          class_ids: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    """Return indices to keep after cross-class NMS (highest confidence wins)."""
    if len(boxes_xyxy) == 0:
        return []

    order = np.argsort(-confidences)
    keep = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if j in suppressed or j == i:
                continue
            if compute_iou(boxes_xyxy[i], boxes_xyxy[j]) >= iou_thresh:
                suppressed.add(j)

    return keep


def main():
    parser = argparse.ArgumentParser(description="Track whales with Norfair (appearance + motion)")
    parser.add_argument("video", type=Path, help="Path to video clip")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Process every Nth frame (default: 3, ~10fps for 30fps video)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold (default: 0.1)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for cross-class NMS (default: 0.5)")
    parser.add_argument("--distance-threshold", type=float, default=0.7,
                        help="Max IoU distance for short-term matching (default: 0.7, i.e. min 0.3 IoU)")
    parser.add_argument("--hit-counter-max", type=int, default=15,
                        help="Short-term track persistence in processed frames (default: 15, ~1.5s at 10fps)")
    parser.add_argument("--reid-threshold", type=float, default=0.5,
                        help="Max ReID distance for long-term matching (default: 0.5, i.e. min 0.5 cosine sim)")
    parser.add_argument("--reid-counter-max", type=int, default=150,
                        help="Long-term ReID persistence in processed frames (default: 150, ~15s at 10fps)")
    parser.add_argument("--init-delay", type=int, default=5,
                        help="Detections needed to confirm a track (default: 5, filters noise)")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable appearance-based ReID (use only IoU)")
    parser.add_argument("--no-gmc", action="store_true",
                        help="Disable camera motion compensation")
    parser.add_argument("--min-track-conf", type=float, default=0.25,
                        help="Drop tracks with avg confidence below this (default: 0.25)")
    parser.add_argument("--min-track-frames", type=int, default=10,
                        help="Drop tracks with fewer frames than this (default: 10)")
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

    # Load YOLO-World model
    print("Loading YOLO-World model (yolov8x-worldv2)...")
    model = YOLO("yolov8x-worldv2.pt")
    model.set_classes(classes)
    print(f"  Class prompts: {classes}")

    # Load appearance embedding extractor
    if not args.no_reid:
        print("Loading ResNet18 embedding extractor...")
        embedder = EmbeddingExtractor()
        print(f"  Device: {embedder.device}")
    else:
        embedder = None

    # Open video to get metadata
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps / args.sample_rate
    cap.release()

    print(f"\nVideo: {args.video.name}")
    print(f"  {width}x{height} @ {fps:.1f}fps, {total_frames} frames, "
          f"{total_frames/fps:.1f}s")
    print(f"  Processing every {args.sample_rate} frames → ~{effective_fps:.1f} effective fps")
    print(f"  Confidence threshold: {args.conf}")

    # Initialize Norfair Tracker with two-tier matching
    tracker_kwargs = dict(
        distance_function=iou_distance,
        distance_threshold=args.distance_threshold,
        hit_counter_max=args.hit_counter_max,
        initialization_delay=args.init_delay,
        filter_factory=OptimizedKalmanFilterFactory(),
        past_detections_length=10,
    )

    if not args.no_reid:
        tracker_kwargs.update(
            reid_distance_function=make_reid_distance(),
            reid_distance_threshold=args.reid_threshold,
            reid_hit_counter_max=args.reid_counter_max,
        )

    tracker = Tracker(**tracker_kwargs)

    print(f"\n  Norfair Tracker initialized:")
    print(f"    Short-term: IoU distance, threshold={args.distance_threshold}, "
          f"hit_counter_max={args.hit_counter_max}")
    if not args.no_reid:
        print(f"    Long-term ReID: cosine distance, threshold={args.reid_threshold}, "
              f"reid_counter_max={args.reid_counter_max} (~{args.reid_counter_max/effective_fps:.0f}s)")
    else:
        print(f"    ReID: DISABLED")

    # Initialize camera motion compensation
    motion_estimator = None
    if not args.no_gmc:
        motion_estimator = MotionEstimator(
            max_points=500,
            min_distance=15,
            transformations_getter=HomographyTransformationGetter(),
        )
        print(f"    Camera motion: HomographyTransformationGetter (500 points)")
    else:
        print(f"    Camera motion: DISABLED")

    # Set up video writer for output — write as AVI/XVID (reliable),
    # then re-encode to H.264 MP4 via ffmpeg if available
    out_video_path_raw = output_dir / f"{video_stem}_tracked_raw.avi"
    out_video_path = output_dir / f"{video_stem}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_video = cv2.VideoWriter(str(out_video_path_raw), fourcc, effective_fps,
                                (width, height))

    # Re-open video for frame-by-frame processing
    cap = cv2.VideoCapture(str(args.video))

    # Track state
    all_tracks: list[dict] = []
    track_trails: dict[int, list[tuple[int, int]]] = {}
    frame_num = 0
    processed = 0

    print(f"\nRunning detection + Norfair tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % args.sample_rate != 0:
            frame_num += 1
            continue

        # Step 1: Detect with YOLO-World
        results = model.predict(
            frame,
            conf=args.conf,
            verbose=False,
        )[0]

        boxes = results.boxes
        has_detections = boxes is not None and len(boxes) > 0

        norfair_detections = []

        if has_detections:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            # Cross-class NMS (YOLO-World returns duplicates across classes)
            keep = cross_class_nms_boxes(xyxy, confs, cls_ids, args.iou_thresh)
            xyxy = xyxy[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep]

            # Build Norfair Detection objects
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                # Norfair Detection: points as [[x1,y1],[x2,y2]] (bbox corners)
                points = np.array([[x1, y1], [x2, y2]])
                scores = np.array([confs[i], confs[i]])

                # Extract appearance embedding for ReID
                embedding = None
                if embedder is not None:
                    embedding = embedder.extract(frame, xyxy[i])

                det = Detection(
                    points=points,
                    scores=scores,
                    data={
                        "class_name": results.names[int(cls_ids[i])],
                        "confidence": float(confs[i]),
                    },
                    embedding=embedding,
                )
                norfair_detections.append(det)

        # Step 2: Camera motion compensation
        coord_transform = None
        if motion_estimator is not None:
            coord_transform = motion_estimator.update(frame)

        # Step 3: Update Norfair tracker
        # period=sample_rate tells Kalman filter the correct dt
        tracked_objects = tracker.update(
            detections=norfair_detections,
            period=args.sample_rate,
            coord_transformations=coord_transform,
        )

        has_tracks = len(tracked_objects) > 0

        if has_tracks:
            for obj in tracked_objects:
                est = obj.estimate  # [[x1,y1],[x2,y2]]
                x1, y1 = est[0]
                x2, y2 = est[1]
                track_id = obj.id
                conf = obj.last_detection.data["confidence"] if obj.last_detection else 0.0
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if track_id not in track_trails:
                    track_trails[track_id] = []
                track_trails[track_id].append((cx, cy))

                class_name = obj.last_detection.data.get("class_name", "whale") if obj.last_detection else "whale"
                all_tracks.append({
                    "frame": frame_num,
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_x1": float(x1),
                    "bbox_y1": float(y1),
                    "bbox_x2": float(x2),
                    "bbox_y2": float(y2),
                    "center_x": cx,
                    "center_y": cy,
                })

        # Draw annotations on frame
        annotated = frame.copy()

        if has_tracks:
            for obj in tracked_objects:
                est = obj.estimate
                x1, y1 = int(est[0][0]), int(est[0][1])
                x2, y2 = int(est[1][0]), int(est[1][1])
                track_id = obj.id
                conf = obj.last_detection.data["confidence"] if obj.last_detection else 0.0
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                trail = track_trails.get(track_id, [])
                if len(trail) > 1:
                    for t in range(1, len(trail)):
                        alpha = t / len(trail)
                        thickness = max(1, int(3 * alpha))
                        pt1 = trail[t - 1]
                        pt2 = trail[t]
                        cv2.line(annotated, pt1, pt2, color, thickness)

        tracker_label = "Norfair"
        if not args.no_reid:
            tracker_label += "+ReID"
        if not args.no_gmc:
            tracker_label += "+GMC"
        cv2.putText(annotated, f"Frame {frame_num} [{tracker_label}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_video.write(annotated)
        processed += 1

        if processed % 20 == 0:
            n_active = len(tracked_objects) if has_tracks else 0
            print(f"  Frame {frame_num:5d}/{total_frames} — "
                  f"{n_active} active tracks, "
                  f"{len(track_trails)} total IDs seen")

        frame_num += 1

    cap.release()
    out_video.release()

    # Re-encode AVI → H.264 MP4 via ffmpeg (if available)
    import shutil as _shutil, subprocess as _sp
    if _shutil.which("ffmpeg"):
        print("\n  Re-encoding to H.264 MP4 via ffmpeg...")
        _sp.run([
            "ffmpeg", "-y", "-i", str(out_video_path_raw),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(out_video_path)
        ], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        if out_video_path.exists():
            out_video_path_raw.unlink()  # remove raw AVI
        else:
            # ffmpeg failed, keep AVI
            out_video_path = out_video_path_raw
            print("  WARNING: ffmpeg re-encode failed, keeping AVI")
    else:
        # No ffmpeg, rename AVI to final output
        out_video_path = out_video_path_raw.rename(
            out_video_path_raw.with_suffix(".avi"))
        print("  NOTE: ffmpeg not found, output is AVI format")

    # Post-processing: filter noise tracks
    unique_tracks_raw = set(t["track_id"] for t in all_tracks)
    tracks_to_drop = set()
    for tid in unique_tracks_raw:
        tid_data = [t for t in all_tracks if t["track_id"] == tid]
        avg_conf = np.mean([t["confidence"] for t in tid_data])
        if avg_conf < args.min_track_conf or len(tid_data) < args.min_track_frames:
            tracks_to_drop.add(tid)

    if tracks_to_drop:
        print(f"\n  Filtering {len(tracks_to_drop)} noise tracks "
              f"(avg_conf < {args.min_track_conf} or frames < {args.min_track_frames}): "
              f"{sorted(tracks_to_drop)}")
        all_tracks = [t for t in all_tracks if t["track_id"] not in tracks_to_drop]

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
        "tracker": "norfair",
        "reid_enabled": not args.no_reid,
        "gmc_enabled": not args.no_gmc,
        "total_frames_processed": processed,
        "sample_rate": args.sample_rate,
        "confidence_threshold": args.conf,
        "distance_threshold": args.distance_threshold,
        "hit_counter_max": args.hit_counter_max,
        "reid_threshold": args.reid_threshold if not args.no_reid else None,
        "reid_counter_max": args.reid_counter_max if not args.no_reid else None,
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

    tracker_label = "Norfair"
    if not args.no_reid:
        tracker_label += "+ReID"
    if not args.no_gmc:
        tracker_label += "+GMC"

    print(f"\n{'='*55}")
    print(f"TRACKING RESULTS: {video_stem} [{tracker_label}]")
    print(f"{'='*55}")
    print(f"  Tracker:             {tracker_label}")
    print(f"  IoU threshold:       {args.distance_threshold}")
    print(f"  hit_counter_max:     {args.hit_counter_max}")
    if not args.no_reid:
        print(f"  ReID threshold:      {args.reid_threshold}")
        print(f"  reid_counter_max:    {args.reid_counter_max} "
              f"(~{args.reid_counter_max/effective_fps:.0f}s)")
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
