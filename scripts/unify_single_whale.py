"""
Unify tracks for a single-whale clip.

For clips known to contain only one whale, merges all significant tracks
into a single ID and drops noise fragments.

Usage:
    python scripts/unify_single_whale.py outputs/track/20231018-40_trim
    python scripts/unify_single_whale.py outputs/track/20231018-40_trim --min-frames 5
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Unify tracks for single-whale clip")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("--csv", type=str, default="tracks_relinked.csv",
                        help="Input CSV (default: tracks_relinked.csv)")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Drop tracks shorter than N frames (default: 5)")
    parser.add_argument("--whale-id", type=int, default=1,
                        help="Unified whale track ID (default: 1)")
    args = parser.parse_args()

    csv_path = args.track_dir / args.csv
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Load tracks
    tracks = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            tracks.append(row)

    # Group by track ID
    by_id = {}
    for t in tracks:
        tid = int(t["track_id"])
        by_id.setdefault(tid, []).append(t)

    print(f"Input: {len(tracks)} points, {len(by_id)} tracks")

    # Show current state
    for tid in sorted(by_id.keys()):
        pts = by_id[tid]
        frames = [int(p["frame"]) for p in pts]
        print(f"  Track {tid}: {len(pts)} frames, range {min(frames)}-{max(frames)}")

    # Filter out noise tracks
    significant = {tid: pts for tid, pts in by_id.items() if len(pts) >= args.min_frames}
    dropped = {tid: pts for tid, pts in by_id.items() if len(pts) < args.min_frames}

    print(f"\nKeeping {len(significant)} tracks (>= {args.min_frames} frames)")
    print(f"Dropping {len(dropped)} noise tracks ({sum(len(p) for p in dropped.values())} points)")

    # Merge all significant tracks into one whale ID
    unified = []
    merge_map = {}
    for tid in sorted(significant.keys()):
        merge_map[tid] = args.whale_id
        for t in significant[tid]:
            new_t = dict(t)
            new_t["track_id"] = str(args.whale_id)
            unified.append(new_t)

    # Sort by frame
    unified.sort(key=lambda x: int(x["frame"]))

    print(f"\nUnified: {len(unified)} points as Track {args.whale_id}")
    frames = [int(t["frame"]) for t in unified]
    print(f"  Frame range: {min(frames)}-{max(frames)}")

    # Detect gaps
    sorted_frames = sorted(set(frames))
    if len(sorted_frames) > 1:
        diffs = [sorted_frames[i+1] - sorted_frames[i] for i in range(len(sorted_frames)-1)]
        gaps = [(sorted_frames[i], sorted_frames[i+1], d) for i, d in enumerate(diffs) if d > 30]
        if gaps:
            print(f"  Surfacing gaps (>30 frame breaks):")
            for start, end, gap in gaps:
                print(f"    Frame {start} → {end} (gap={gap} frames)")

    # Save
    out_path = args.track_dir / "tracks_unified.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unified)
    print(f"\nSaved to {out_path}")

    # Save summary
    summary = {
        "input_csv": args.csv,
        "input_tracks": len(by_id),
        "merged_track_ids": list(significant.keys()),
        "dropped_track_ids": list(dropped.keys()),
        "unified_whale_id": args.whale_id,
        "total_points": len(unified),
        "frame_range": [min(frames), max(frames)],
    }
    summary_path = args.track_dir / "unify_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
