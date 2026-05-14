"""
Re-link fragmented whale tracks using spatial proximity + appearance similarity.

Post-processes tracks.csv from track_whales.py to merge track fragments
that likely belong to the same whale (e.g., across a short dive).

Strategy:
  1. Spatial: If a new track starts near where an old track ended within
     a time window, they're candidates for merging.
  2. Appearance: Crop whale bounding boxes from the video, extract visual
     features with ResNet18, compare embeddings between candidate pairs.
  3. Merge tracks that pass both filters.

Usage:
    python scripts/relink_tracks.py outputs/track/20240527-22 exeter/20240527-22.MP4
    python scripts/relink_tracks.py outputs/track/20240527-22 exeter/20240527-22.MP4 --max-gap 200 --max-dist 500
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


def get_track_endpoints(tracks: list[dict]) -> dict:
    """Get first/last position and frame for each track."""
    by_id: dict[int, list[dict]] = {}
    for t in tracks:
        by_id.setdefault(t["track_id"], []).append(t)

    endpoints = {}
    for tid, pts in by_id.items():
        pts_sorted = sorted(pts, key=lambda x: x["frame"])
        endpoints[tid] = {
            "first_frame": pts_sorted[0]["frame"],
            "last_frame": pts_sorted[-1]["frame"],
            "first_pos": (pts_sorted[0]["center_x"], pts_sorted[0]["center_y"]),
            "last_pos": (pts_sorted[-1]["center_x"], pts_sorted[-1]["center_y"]),
            "first_bbox": (pts_sorted[0]["bbox_x1"], pts_sorted[0]["bbox_y1"],
                           pts_sorted[0]["bbox_x2"], pts_sorted[0]["bbox_y2"]),
            "last_bbox": (pts_sorted[-1]["bbox_x1"], pts_sorted[-1]["bbox_y1"],
                          pts_sorted[-1]["bbox_x2"], pts_sorted[-1]["bbox_y2"]),
            "num_points": len(pts),
        }
    return endpoints


def find_relink_candidates(endpoints: dict, max_gap_frames: int,
                           max_dist_px: float, min_frames: int) -> list[tuple[int, int, float, float]]:
    """Find pairs of tracks that could be the same whale.

    Returns list of (old_track_id, new_track_id, time_gap, spatial_dist).
    """
    candidates = []
    tids = sorted(endpoints.keys())

    for i, tid_a in enumerate(tids):
        ep_a = endpoints[tid_a]
        if ep_a["num_points"] < min_frames:
            continue

        for tid_b in tids[i + 1:]:
            ep_b = endpoints[tid_b]
            if ep_b["num_points"] < min_frames:
                continue

            # Check if B starts after A ends
            gap = ep_b["first_frame"] - ep_a["last_frame"]
            if gap <= 0 or gap > max_gap_frames:
                # Also check reverse (A starts after B ends)
                gap_rev = ep_a["first_frame"] - ep_b["last_frame"]
                if gap_rev <= 0 or gap_rev > max_gap_frames:
                    continue
                # Swap: B ended first, A started later
                tid_a, tid_b = tid_b, tid_a
                ep_a, ep_b = ep_b, ep_a
                gap = gap_rev

            # Spatial distance between A's last position and B's first position
            dx = ep_a["last_pos"][0] - ep_b["first_pos"][0]
            dy = ep_a["last_pos"][1] - ep_b["first_pos"][1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist <= max_dist_px:
                candidates.append((tid_a, tid_b, gap, dist))

    # Sort by distance (closest first)
    candidates.sort(key=lambda x: x[3])
    return candidates


def extract_track_crops(video_path: Path, tracks: list[dict],
                        track_ids: set[int], max_crops: int = 10) -> dict[int, list[np.ndarray]]:
    """Extract whale bounding box crops from the video for given track IDs."""
    # Collect which frames to grab for each track
    by_id: dict[int, list[dict]] = {}
    for t in tracks:
        if t["track_id"] in track_ids:
            by_id.setdefault(t["track_id"], []).append(t)

    # Select evenly spaced frames per track (up to max_crops)
    frames_needed: dict[int, list[dict]] = {}  # frame_num -> list of track points
    for tid, pts in by_id.items():
        pts_sorted = sorted(pts, key=lambda x: x["frame"])
        # Pick highest confidence frames
        pts_by_conf = sorted(pts_sorted, key=lambda x: -x["confidence"])[:max_crops]
        for p in pts_by_conf:
            frames_needed.setdefault(p["frame"], []).append(p)

    # Read video and extract crops
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: Cannot open video {video_path}")
        return {}

    crops: dict[int, list[np.ndarray]] = {tid: [] for tid in track_ids}
    target_frames = sorted(frames_needed.keys())
    frame_idx = 0
    current_target = 0

    while current_target < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_frames[current_target]:
            h, w = frame.shape[:2]
            for pt in frames_needed[frame_idx]:
                x1 = max(0, int(pt["bbox_x1"]))
                y1 = max(0, int(pt["bbox_y1"]))
                x2 = min(w, int(pt["bbox_x2"]))
                y2 = min(h, int(pt["bbox_y2"]))
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crops[pt["track_id"]].append(crop)
            current_target += 1

        frame_idx += 1

    cap.release()
    return crops


def compute_embeddings(crops: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    """Compute average visual embedding per track using ResNet18."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove classification head — use as feature extractor
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    with torch.no_grad():
        for tid, crop_list in crops.items():
            if not crop_list:
                continue
            feats = []
            for crop in crop_list:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = transform(crop_rgb).unsqueeze(0).to(device)
                feat = model(tensor).squeeze().cpu().numpy()
                feats.append(feat)
            # Average embedding across all crops for this track
            embeddings[tid] = np.mean(feats, axis=0)

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_pose_features(pose_csv: Path, track_ids: set[int]) -> dict[int, dict]:
    """Compute average body length from pose keypoints per track.

    Body length (rostrum→caudal distance) is consistent for the same whale
    at similar drone altitude, making it more discriminative than generic
    ResNet18 features for same-whale matching.
    """
    if not pose_csv.exists():
        return {}

    by_track: dict[int, list[float]] = {tid: [] for tid in track_ids}

    with open(pose_csv, newline="") as f:
        for row in csv.DictReader(f):
            tid = int(row["track_id"])
            if tid not in track_ids:
                continue
            rx = row.get("rostrum_tip_x", "")
            ry = row.get("rostrum_tip_y", "")
            cx = row.get("Caudal_peduncle_x", "")
            cy = row.get("Caudal_peduncle_y", "")
            if not all([rx, ry, cx, cy]):
                continue
            dx = float(rx) - float(cx)
            dy = float(ry) - float(cy)
            by_track[tid].append((dx**2 + dy**2) ** 0.5)

    features = {}
    for tid, lengths in by_track.items():
        if not lengths:
            continue
        features[tid] = {
            "mean_body_length": float(np.mean(lengths)),
            "n_pose_frames": len(lengths),
        }
    return features


def pose_body_length_ratio(feat_a: dict, feat_b: dict) -> float:
    """Score body-length similarity between two tracks.

    Same whale at similar altitude should have consistent apparent body length.
    Returns ratio in [0, 1] — closer to 1 means more similar.
    """
    len_a = feat_a["mean_body_length"]
    len_b = feat_b["mean_body_length"]
    if min(len_a, len_b) < 1:
        return 0.5
    return min(len_a, len_b) / max(len_a, len_b)


def merge_tracks(tracks: list[dict], merge_map: dict[int, int]) -> list[dict]:
    """Apply merge mapping to tracks, reassigning IDs."""
    merged = []
    for t in tracks:
        new_t = dict(t)
        old_id = t["track_id"]
        # Follow merge chain
        while old_id in merge_map:
            old_id = merge_map[old_id]
        new_t["track_id"] = old_id
        merged.append(new_t)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Re-link fragmented whale tracks")
    parser.add_argument("track_dir", type=Path,
                        help="Directory with tracks.csv and summary.json")
    parser.add_argument("video", type=Path, help="Path to the source video")
    parser.add_argument("--max-gap", type=int, default=200,
                        help="Max frame gap between tracks to consider merging (default: 200)")
    parser.add_argument("--max-dist", type=float, default=600,
                        help="Max pixel distance between track end/start (default: 600)")
    parser.add_argument("--min-similarity", type=float, default=0.7,
                        help="Minimum cosine similarity for appearance match (default: 0.7)")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Ignore tracks shorter than N frames (default: 5)")
    args = parser.parse_args()

    csv_path = args.track_dir / "tracks.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load tracks
    print(f"Loading tracks from {csv_path}...")
    tracks = load_tracks(csv_path)
    endpoints = get_track_endpoints(tracks)
    print(f"  {len(tracks)} points, {len(endpoints)} unique tracks")

    # Find spatial candidates
    print(f"\nFinding re-link candidates (max gap={args.max_gap} frames, "
          f"max dist={args.max_dist}px)...")
    candidates = find_relink_candidates(endpoints, args.max_gap, args.max_dist,
                                        args.min_frames)
    print(f"  {len(candidates)} spatial candidates found:")
    for old_id, new_id, gap, dist in candidates:
        print(f"    Track {old_id} → Track {new_id}: "
              f"gap={gap} frames, dist={dist:.0f}px")

    if not candidates:
        print("\n  No candidates to merge. Try increasing --max-gap or --max-dist.")
        return

    # Extract crops for candidate tracks
    candidate_tids = set()
    for old_id, new_id, _, _ in candidates:
        candidate_tids.add(old_id)
        candidate_tids.add(new_id)

    print(f"\nExtracting appearance crops for {len(candidate_tids)} tracks...")
    crops = extract_track_crops(args.video, tracks, candidate_tids)
    for tid, crop_list in crops.items():
        print(f"    Track {tid}: {len(crop_list)} crops")

    # Compute embeddings
    print(f"\nComputing visual embeddings (ResNet18)...")
    embeddings = compute_embeddings(crops)
    print(f"  Computed embeddings for {len(embeddings)} tracks")

    # Load pose features if available
    pose_csv = args.track_dir / "pose" / "pose_keypoints.csv"
    pose_features = load_pose_features(pose_csv, candidate_tids)
    if pose_features:
        print(f"  Loaded pose features for {len(pose_features)} tracks")

    # Score candidates with appearance similarity
    print(f"\nScoring candidates (min similarity={args.min_similarity})...")
    merge_map: dict[int, int] = {}  # new_id -> old_id (merge into old)
    already_merged: set[int] = set()

    for old_id, new_id, gap, dist in candidates:
        if new_id in already_merged or old_id in already_merged:
            continue
        if old_id not in embeddings or new_id not in embeddings:
            print(f"    Track {old_id} → {new_id}: SKIP (no embeddings)")
            continue

        sim = cosine_similarity(embeddings[old_id], embeddings[new_id])

        # Check pose body-length consistency if available
        pose_info = ""
        if old_id in pose_features and new_id in pose_features:
            bl_ratio = pose_body_length_ratio(
                pose_features[old_id], pose_features[new_id])
            pose_info = f", body-length ratio={bl_ratio:.3f}"
            if bl_ratio < 0.7:
                print(f"    Track {old_id} → {new_id}: similarity={sim:.3f}"
                      f"{pose_info} → REJECT (body length mismatch)")
                continue

        status = "MERGE" if sim >= args.min_similarity else "REJECT"
        print(f"    Track {old_id} → {new_id}: similarity={sim:.3f}"
              f"{pose_info}, gap={gap}, dist={dist:.0f}px → {status}")

        if sim >= args.min_similarity:
            merge_map[new_id] = old_id
            already_merged.add(new_id)

    if not merge_map:
        print("\n  No tracks passed appearance matching. "
              "Try lowering --min-similarity.")
        return

    # Apply merges
    print(f"\nMerging {len(merge_map)} track pairs...")
    for new_id, old_id in merge_map.items():
        print(f"    Track {new_id} → Track {old_id}")

    merged_tracks = merge_tracks(tracks, merge_map)

    # Save results
    out_csv = args.track_dir / "tracks_relinked.csv"
    fieldnames = list(merged_tracks[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_tracks)

    # Summary
    new_endpoints = get_track_endpoints(merged_tracks)
    long_tracks = {tid: ep for tid, ep in new_endpoints.items()
                   if ep["num_points"] >= args.min_frames}

    print(f"\n{'='*55}")
    print(f"RE-LINK RESULTS")
    print(f"{'='*55}")
    print(f"  Before: {len(endpoints)} tracks")
    print(f"  Merged: {len(merge_map)} pairs")
    print(f"  After:  {len(new_endpoints)} tracks "
          f"({len(long_tracks)} with ≥{args.min_frames} frames)")
    print(f"\n  Re-linked tracks (≥{args.min_frames} frames):")
    for tid in sorted(long_tracks.keys()):
        ep = long_tracks[tid]
        print(f"    Track {tid:>3d}: {ep['num_points']:4d} frames, "
              f"span {ep['first_frame']}→{ep['last_frame']}")
    print(f"\n  Saved to {out_csv}")

    # Save merge summary
    merge_summary = {
        "merge_map": {str(k): v for k, v in merge_map.items()},
        "params": {
            "max_gap_frames": args.max_gap,
            "max_dist_px": args.max_dist,
            "min_similarity": args.min_similarity,
            "min_frames": args.min_frames,
        },
        "before_track_count": len(endpoints),
        "after_track_count": len(new_endpoints),
    }
    with open(args.track_dir / "relink_summary.json", "w") as f:
        json.dump(merge_summary, f, indent=2)
    print(f"  Merge details saved to {args.track_dir / 'relink_summary.json'}\n")


if __name__ == "__main__":
    main()
