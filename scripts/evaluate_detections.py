"""
Evaluate YOLO-World detection quality.

Merges overlapping detections (cross-class NMS), analyzes confidence
thresholds, and generates an HTML review grid for human TP/FP labeling.

Usage:
    python scripts/evaluate_detections.py outputs/detect_world/srkw_calf_drone
    python scripts/evaluate_detections.py outputs/detect_world/20240727-84
    python scripts/evaluate_detections.py outputs/detect_world/srkw_calf_drone --iou-thresh 0.5
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_detections(csv_path: Path) -> list[dict]:
    """Load detections from CSV, parsing bbox strings."""
    detections = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            det = {
                "frame": int(row["frame"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "confidence": float(row["confidence"]),
                "bbox": ast.literal_eval(row["bbox"]),  # [x1, y1, x2, y2]
            }
            detections.append(det)
    return detections


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
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


def cross_class_nms(detections: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Merge overlapping detections across classes within each frame.

    For overlapping boxes (IoU > threshold), keeps the highest-confidence
    detection and records which classes were merged.
    """
    # Group by frame
    by_frame: dict[int, list[dict]] = {}
    for d in detections:
        by_frame.setdefault(d["frame"], []).append(d)

    merged_all = []
    for frame_num in sorted(by_frame):
        frame_dets = sorted(by_frame[frame_num], key=lambda x: -x["confidence"])
        keep = []
        suppressed = [False] * len(frame_dets)

        for i, det_i in enumerate(frame_dets):
            if suppressed[i]:
                continue

            merged_classes = {det_i["class_name"]: det_i["confidence"]}
            for j in range(i + 1, len(frame_dets)):
                if suppressed[j]:
                    continue
                if compute_iou(det_i["bbox"], frame_dets[j]["bbox"]) >= iou_thresh:
                    suppressed[j] = True
                    merged_classes[frame_dets[j]["class_name"]] = frame_dets[j]["confidence"]

            merged_det = {
                **det_i,
                "merged_classes": merged_classes,
                "num_merged": len(merged_classes),
            }
            keep.append(merged_det)

        merged_all.extend(keep)

    return merged_all


def threshold_analysis(detections: list[dict]) -> list[dict]:
    """Show detection counts and frame coverage at different thresholds."""
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    all_frames = set(d["frame"] for d in detections)
    results = []

    for t in thresholds:
        filtered = [d for d in detections if d["confidence"] >= t]
        frames_hit = set(d["frame"] for d in filtered)
        results.append({
            "threshold": t,
            "detections": len(filtered),
            "frames_with_dets": len(frames_hit),
            "frame_coverage": len(frames_hit) / len(all_frames) if all_frames else 0,
        })

    return results


def generate_review_html(merged_dets: list[dict], output_dir: Path,
                         frame_dir: Path) -> Path:
    """Generate an HTML page for visual review of detections.

    Shows each annotated frame alongside its merged detections for
    human TP/FP assessment. Uses existing annotated JPGs.
    """
    # Find available annotated frames
    frame_files = {
        int(p.stem.split("_")[1]): p.name
        for p in frame_dir.glob("frame_*_det.jpg")
    }

    # Group merged detections by frame
    dets_by_frame: dict[int, list[dict]] = {}
    for d in merged_dets:
        dets_by_frame.setdefault(d["frame"], []).append(d)

    # Get unique frames that have both annotations and detections
    review_frames = sorted(set(frame_files.keys()) & set(dets_by_frame.keys()))

    html_path = output_dir / "review.html"

    # Confidence distribution for summary
    all_confs = [d["confidence"] for d in merged_dets]
    high_conf = [c for c in all_confs if c >= 0.3]
    mid_conf = [c for c in all_confs if 0.1 <= c < 0.3]
    low_conf = [c for c in all_confs if c < 0.1]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Detection Review — {output_dir.name}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
  h1 {{ color: #4fc3f7; }}
  .summary {{ background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
  .summary table {{ border-collapse: collapse; }}
  .summary td, .summary th {{ padding: 6px 14px; text-align: left; }}
  .summary th {{ color: #90caf9; }}
  .frame-card {{ background: #2a2a2a; margin: 15px 0; padding: 15px; border-radius: 8px;
                 display: flex; gap: 20px; align-items: flex-start; }}
  .frame-card img {{ max-width: 720px; border-radius: 4px; cursor: pointer; }}
  .frame-card img:hover {{ max-width: 100%; }}
  .det-list {{ min-width: 280px; }}
  .det-item {{ padding: 6px 10px; margin: 4px 0; border-radius: 4px; font-size: 14px; }}
  .det-item.high {{ background: #1b5e20; }}
  .det-item.mid {{ background: #e65100; }}
  .det-item.low {{ background: #b71c1c; }}
  .conf {{ font-weight: bold; font-size: 16px; }}
  .merged {{ color: #aaa; font-size: 12px; }}
  .controls {{ margin: 10px 0; }}
  .filter-btn {{ padding: 6px 14px; margin: 3px; border: none; border-radius: 4px;
                 cursor: pointer; font-size: 13px; }}
  .filter-btn.active {{ outline: 2px solid #4fc3f7; }}
  .hidden {{ display: none; }}
  .stats-bar {{ display: flex; gap: 20px; margin: 10px 0; }}
  .stat {{ text-align: center; }}
  .stat .num {{ font-size: 28px; font-weight: bold; }}
  .stat .label {{ font-size: 12px; color: #aaa; }}
</style>
</head>
<body>
<h1>Detection Review — {output_dir.name}</h1>

<div class="summary">
  <div class="stats-bar">
    <div class="stat"><div class="num">{len(merged_dets)}</div><div class="label">Merged detections</div></div>
    <div class="stat"><div class="num">{len(review_frames)}</div><div class="label">Frames to review</div></div>
    <div class="stat" style="color:#4caf50"><div class="num">{len(high_conf)}</div><div class="label">High conf (≥0.3)</div></div>
    <div class="stat" style="color:#ff9800"><div class="num">{len(mid_conf)}</div><div class="label">Mid conf (0.1–0.3)</div></div>
    <div class="stat" style="color:#f44336"><div class="num">{len(low_conf)}</div><div class="label">Low conf (<0.1)</div></div>
  </div>
</div>

<div class="controls">
  <strong>Filter by confidence:</strong>
  <button class="filter-btn active" onclick="filterFrames('all')">All</button>
  <button class="filter-btn" onclick="filterFrames('high')" style="background:#1b5e20;color:#fff">High (≥0.3)</button>
  <button class="filter-btn" onclick="filterFrames('mid')" style="background:#e65100;color:#fff">Mid (0.1–0.3)</button>
  <button class="filter-btn" onclick="filterFrames('low')" style="background:#b71c1c;color:#fff">Low (<0.1)</button>
</div>

<div id="frames">
"""

    for frame_num in review_frames:
        dets = sorted(dets_by_frame[frame_num], key=lambda x: -x["confidence"])
        img_file = frame_files[frame_num]

        # Determine max confidence tier for this frame
        max_conf = max(d["confidence"] for d in dets)
        if max_conf >= 0.3:
            tier = "high"
        elif max_conf >= 0.1:
            tier = "mid"
        else:
            tier = "low"

        det_html = ""
        for d in dets:
            if d["confidence"] >= 0.3:
                css = "high"
            elif d["confidence"] >= 0.1:
                css = "mid"
            else:
                css = "low"

            merged_info = ""
            if d.get("num_merged", 1) > 1:
                others = [f"{c}:{v:.2f}" for c, v in d["merged_classes"].items()
                          if c != d["class_name"]]
                merged_info = f'<div class="merged">merged with: {", ".join(others)}</div>'

            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            det_html += f"""<div class="det-item {css}">
  <span class="conf">{d['confidence']:.3f}</span> {d['class_name']}
  <div class="merged">bbox: {x1},{y1} → {x2},{y2} ({x2-x1}×{y2-y1}px)</div>
  {merged_info}
</div>"""

        html += f"""
<div class="frame-card" data-tier="{tier}" data-frame="{frame_num}">
  <img src="{img_file}" alt="Frame {frame_num}" loading="lazy">
  <div class="det-list">
    <strong>Frame {frame_num}</strong> — {len(dets)} detection(s)
    {det_html}
  </div>
</div>
"""

    html += """
</div>

<script>
function filterFrames(tier) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.frame-card').forEach(card => {
    if (tier === 'all' || card.dataset.tier === tier) {
      card.classList.remove('hidden');
    } else {
      card.classList.add('hidden');
    }
  });
}
</script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def print_threshold_table(thresh_results: list[dict], total_frames: int):
    """Print confidence threshold analysis."""
    print(f"\n{'='*65}")
    print("CONFIDENCE THRESHOLD ANALYSIS (after cross-class NMS)")
    print(f"{'='*65}")
    print(f"  {'Threshold':>10}  {'Detections':>11}  {'Frames hit':>11}  {'Coverage':>10}")
    print(f"  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*10}")
    for r in thresh_results:
        print(f"  {r['threshold']:>10.2f}  {r['detections']:>11d}  "
              f"{r['frames_with_dets']:>11d}  {r['frame_coverage']:>9.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World detections")
    parser.add_argument("detection_dir", type=Path,
                        help="Directory with detections.csv and annotated frames")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for cross-class NMS (default: 0.5)")
    args = parser.parse_args()

    det_dir = args.detection_dir
    csv_path = det_dir / "detections.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load raw detections
    print(f"Loading detections from {csv_path}...")
    raw_dets = load_detections(csv_path)
    total_frames = len(set(d["frame"] for d in raw_dets))
    print(f"  {len(raw_dets)} raw detections across {total_frames} frames")

    # Per-class breakdown (raw)
    class_counts: dict[str, int] = {}
    for d in raw_dets:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

    print(f"\n  Raw per-class counts:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:20s}: {cnt}")

    # Cross-class NMS
    print(f"\n  Applying cross-class NMS (IoU threshold={args.iou_thresh})...")
    merged = cross_class_nms(raw_dets, args.iou_thresh)
    print(f"  {len(raw_dets)} raw → {len(merged)} merged detections "
          f"({len(raw_dets) - len(merged)} suppressed)")

    # How many had multiple classes merged
    multi_merged = [d for d in merged if d.get("num_merged", 1) > 1]
    print(f"  {len(multi_merged)} detections had overlapping classes merged")

    # Threshold analysis on merged detections
    thresh_results = threshold_analysis(merged)
    print_threshold_table(thresh_results, total_frames)

    # Save merged detections CSV
    merged_csv = det_dir / "detections_merged.csv"
    fieldnames = ["frame", "class_name", "confidence", "bbox", "num_merged", "merged_classes"]
    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for d in merged:
            row = {**d, "merged_classes": str(d.get("merged_classes", {}))}
            writer.writerow(row)
    print(f"  Saved merged detections to {merged_csv}")

    # Generate HTML review page
    print(f"\n  Generating HTML review page...")
    html_path = generate_review_html(merged, det_dir, det_dir)
    print(f"  Saved review page to {html_path}")
    print(f"  Open in browser to visually assess TP/FP detections.\n")


if __name__ == "__main__":
    main()
