"""
Render a tracked video with detection boxes, minimap, and pose skeleton overlay.

Combines all pipeline outputs into a single annotated video, similar to Ren's DORSAP output:
  - Bounding boxes with track ID and confidence
  - Corner minimap showing full trajectory with current position (racing-game style)
  - 7-keypoint skeleton with edges
  - Frame counter and track info HUD

Usage:
    python scripts/render_tracked_video.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4
    python scripts/render_tracked_video.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --csv tracks_unified.csv --sample-rate 3
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


TRACK_COLORS = [
    (0, 255, 0),     # green
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 165, 0),   # orange
    (0, 165, 255),   # light blue
    (255, 0, 0),     # red
    (128, 255, 0),   # lime
    (255, 128, 128), # pink
    (0, 128, 255),   # sky blue
    (128, 0, 255),   # purple
]

KEYPOINT_NAMES = [
    "rostrum_tip",
    "left_caudal_fluke",
    "right_caudal_fluke",
    "left_pect_fin_tip",
    "right_pect_fin_tip",
    "mid_saddle_patch",
    "Caudal_peduncle",
]

SKELETON_EDGES = [
    ("rostrum_tip", "mid_saddle_patch"),
    ("mid_saddle_patch", "Caudal_peduncle"),
    ("mid_saddle_patch", "left_pect_fin_tip"),
    ("mid_saddle_patch", "right_pect_fin_tip"),
    ("Caudal_peduncle", "left_caudal_fluke"),
    ("Caudal_peduncle", "right_caudal_fluke"),
]

KP_COLORS = {
    "rostrum_tip":        (0, 255, 0),
    "mid_saddle_patch":   (0, 255, 255),
    "Caudal_peduncle":    (255, 165, 0),
    "left_pect_fin_tip":  (255, 0, 255),
    "right_pect_fin_tip": (255, 0, 128),
    "left_caudal_fluke":  (0, 165, 255),
    "right_caudal_fluke": (0, 128, 255),
}

KP_SHORT_NAMES = {
    "rostrum_tip":        "rostrum",
    "mid_saddle_patch":   "saddle",
    "Caudal_peduncle":    "peduncle",
    "left_pect_fin_tip":  "L pec",
    "right_pect_fin_tip": "R pec",
    "left_caudal_fluke":  "L fluke",
    "right_caudal_fluke": "R fluke",
}


def load_tracks(csv_path: Path) -> dict[int, list[dict]]:
    """Load tracks CSV, return {frame: [track_dicts]}."""
    by_frame: dict[int, list[dict]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            by_frame.setdefault(fr, []).append({
                "track_id": int(row["track_id"]),
                "confidence": float(row["confidence"]),
                "center_x": int(row["center_x"]),
                "center_y": int(row["center_y"]),
                "x1": float(row["bbox_x1"]),
                "y1": float(row["bbox_y1"]),
                "x2": float(row["bbox_x2"]),
                "y2": float(row["bbox_y2"]),
            })
    return by_frame


def load_pose(csv_path: Path) -> dict[int, dict[int, dict]]:
    """Load pose CSV, return {frame: {track_id: {kp_name: {x, y, conf}}}}."""
    by_frame: dict[int, dict[int, dict]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fr = int(row["frame"])
            tid = int(row["track_id"])
            kps = {}
            for kp in KEYPOINT_NAMES:
                x = row.get(f"{kp}_x", "")
                y = row.get(f"{kp}_y", "")
                conf = row.get(f"{kp}_conf", "")
                if x and y:
                    kps[kp] = {"x": float(x), "y": float(y),
                               "conf": float(conf) if conf else 0}
            by_frame.setdefault(fr, {})[tid] = kps
    return by_frame


def draw_skeleton(frame: np.ndarray, keypoints: dict, track_id: int,
                  point_size: int = 10, line_width: int = 3, label: bool = True):
    """Draw skeleton edges and keypoint markers."""
    color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

    # Edges
    for kp_a, kp_b in SKELETON_EDGES:
        if kp_a in keypoints and kp_b in keypoints:
            pt_a = (int(keypoints[kp_a]["x"]), int(keypoints[kp_a]["y"]))
            pt_b = (int(keypoints[kp_b]["x"]), int(keypoints[kp_b]["y"]))
            cv2.line(frame, pt_a, pt_b, color, line_width, cv2.LINE_AA)

    # Keypoints
    for kp_name, kp_data in keypoints.items():
        pt = (int(kp_data["x"]), int(kp_data["y"]))
        kp_color = KP_COLORS.get(kp_name, color)
        cv2.circle(frame, pt, point_size, kp_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, point_size + 1, (255, 255, 255), 2, cv2.LINE_AA)

        if label:
            short = KP_SHORT_NAMES.get(kp_name, kp_name[:8])
            conf = kp_data.get("conf", 0)
            txt = f"{short} {conf:.2f}"
            cv2.putText(frame, txt, (pt[0] + 14, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)


class Minimap:
    """Racing-game style minimap showing full trajectory with current position."""

    def __init__(self, trajectory_tracks: dict, size: int = 300, margin: int = 20,
                 padding: int = 25):
        """Build minimap from trajectory data.

        Args:
            trajectory_tracks: {frame: [track_dicts]} — should use compensated
                               coordinates when available for camera-drift-free paths.
        """
        self.size = size
        self.margin = margin
        self.padding = padding

        # Build full trajectory per track: {tid: [(frame, cx, cy), ...]}
        self.trajectories: dict[int, list[tuple[int, int, int]]] = {}
        for fr, tracks in trajectory_tracks.items():
            for t in tracks:
                tid = t["track_id"]
                self.trajectories.setdefault(tid, []).append(
                    (fr, t["center_x"], t["center_y"]))
        for tid in self.trajectories:
            self.trajectories[tid].sort(key=lambda x: x[0])

        # Compute bounding box across ALL track points for stable mapping
        all_x = [p[1] for pts in self.trajectories.values() for p in pts]
        all_y = [p[2] for pts in self.trajectories.values() for p in pts]
        if not all_x:
            self.scale = 1.0
            self.off_x = self.off_y = 0
            return

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        range_x = max(max_x - min_x, 1)
        range_y = max(max_y - min_y, 1)
        draw_area = size - 2 * padding
        self.scale = min(draw_area / range_x, draw_area / range_y)
        # Center the trajectory in the minimap
        self.off_x = padding + (draw_area - range_x * self.scale) / 2 - min_x * self.scale
        self.off_y = padding + (draw_area - range_y * self.scale) / 2 - min_y * self.scale

    def _map(self, x: int, y: int) -> tuple[int, int]:
        return (int(x * self.scale + self.off_x),
                int(y * self.scale + self.off_y))

    def draw(self, frame: np.ndarray, current_frame: int,
             active_track_ids: set[int]):
        """Draw minimap on top-left corner of frame."""
        fh, fw = frame.shape[:2]
        s = self.size
        m = self.margin

        # Minimap region — top-left, below HUD bar
        rx1 = m
        ry1 = m + 85  # offset below HUD
        rx2 = m + s
        ry2 = ry1 + s

        # Dark semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 100, 120), 2,
                      cv2.LINE_AA)

        for tid, pts in self.trajectories.items():
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            dim = tuple(max(30, v // 3) for v in color)  # dimmed color

            # Find index of current position
            current_idx = None
            for i, (fr, cx, cy) in enumerate(pts):
                if fr <= current_frame:
                    current_idx = i

            # Draw full path — dim for future, bright for traveled
            mapped = [(self._map(cx, cy)) for _, cx, cy in pts]
            for i in range(1, len(mapped)):
                px1 = rx1 + mapped[i - 1][0]
                py1 = ry1 + mapped[i - 1][1]
                px2 = rx1 + mapped[i][0]
                py2 = ry1 + mapped[i][1]
                if current_idx is not None and i <= current_idx:
                    cv2.line(frame, (px1, py1), (px2, py2), color, 2,
                             cv2.LINE_AA)
                else:
                    cv2.line(frame, (px1, py1), (px2, py2), dim, 1,
                             cv2.LINE_AA)

            # Start marker (small circle)
            sx, sy = rx1 + mapped[0][0], ry1 + mapped[0][1]
            cv2.circle(frame, (sx, sy), 5, (255, 255, 255), 1, cv2.LINE_AA)

            # End marker (small square)
            ex, ey = rx1 + mapped[-1][0], ry1 + mapped[-1][1]
            cv2.rectangle(frame, (ex - 4, ey - 4), (ex + 4, ey + 4),
                          (255, 255, 255), 1, cv2.LINE_AA)

            # Current position — bright pulsing dot
            if current_idx is not None:
                cx_m = rx1 + mapped[current_idx][0]
                cy_m = ry1 + mapped[current_idx][1]
                cv2.circle(frame, (cx_m, cy_m), 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx_m, cy_m), 9, (255, 255, 255), 2,
                           cv2.LINE_AA)

        # Label
        cv2.putText(frame, "TRACK MAP", (rx1 + 8, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1,
                    cv2.LINE_AA)

        # Progress indicator
        if self.trajectories:
            all_frames_sorted = sorted(
                set(fr for pts in self.trajectories.values() for fr, _, _ in pts))
            if len(all_frames_sorted) > 1:
                first_f = all_frames_sorted[0]
                last_f = all_frames_sorted[-1]
                progress = max(0, min(1, (current_frame - first_f) /
                                      max(1, last_f - first_f)))
                bar_y = ry2 - 10
                bar_x1 = rx1 + 8
                bar_x2 = rx2 - 8
                bar_w = bar_x2 - bar_x1
                cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 4),
                              (60, 60, 70), -1)
                cv2.rectangle(frame, (bar_x1, bar_y),
                              (bar_x1 + int(bar_w * progress), bar_y + 4),
                              (0, 255, 180), -1)


def draw_hud(frame: np.ndarray, frame_num: int, fps: float,
             n_tracks: int, n_keypoints: int, sample_rate: int):
    """Draw heads-up display with frame info."""
    h, w = frame.shape[:2]
    time_s = frame_num / fps if fps > 0 else 0

    lines = [
        f"Frame {frame_num}  |  {time_s:.1f}s",
        f"Tracks: {n_tracks}  |  Keypoints: {n_keypoints}/7",
    ]

    # Background bar
    bar_h = 35 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (15, 28 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)


class MetricsPanel:
    """Live metrics overlay: body calibration, speed, surfacing, respiration."""

    def __init__(self, comp_tracks: dict, all_frames: list[int],
                 fps: float, sample_rate: int, body_length_m: float = 7.0,
                 resp_csv: Path | None = None):
        self.fps = fps
        self.sample_rate = sample_rate
        self.body_length_m = body_length_m
        self.dt = sample_rate / fps if fps > 0 else 0.1

        # Build per-track ordered position lists from compensated coords
        # {tid: [(frame, cx, cy), ...]}
        self._positions: dict[int, list[tuple[int, float, float]]] = {}
        for fr, tracks in comp_tracks.items():
            for t in tracks:
                tid = t["track_id"]
                self._positions.setdefault(tid, []).append(
                    (fr, float(t["center_x"]), float(t["center_y"])))
        for tid in self._positions:
            self._positions[tid].sort()

        # Calibrate body length from bounding boxes (use raw tracks for pixel size)
        self._bbox_lengths: dict[int, list[float]] = {}
        for fr, tracks in comp_tracks.items():
            for t in tracks:
                tid = t["track_id"]
                bw = abs(t["x2"] - t["x1"])
                bh = abs(t["y2"] - t["y1"])
                diag = (bw**2 + bh**2) ** 0.5
                self._bbox_lengths.setdefault(tid, []).append(diag)

        # Body length in pixels (median bbox diagonal as proxy)
        all_diags = [d for ds in self._bbox_lengths.values() for d in ds]
        self.body_length_px = float(np.median(all_diags)) if all_diags else 500.0
        self.px_per_m = self.body_length_px / body_length_m

        # Pre-compute per-frame speed (compensated, in m/s and BL/s)
        self.frame_speed: dict[int, dict[int, float]] = {}  # {frame: {tid: speed_m_s}}
        for tid, pts in self._positions.items():
            for i in range(1, len(pts)):
                fr_prev, x_prev, y_prev = pts[i - 1]
                fr_curr, x_curr, y_curr = pts[i]
                d_frames = fr_curr - fr_prev
                if d_frames <= 0:
                    continue
                dist_px = ((x_curr - x_prev)**2 + (y_curr - y_prev)**2) ** 0.5
                dt = d_frames / fps if fps > 0 else 0.1
                speed_ms = (dist_px / self.px_per_m) / dt
                self.frame_speed.setdefault(fr_curr, {})[tid] = speed_ms

        # Pre-compute surfacing state per frame
        # A frame with a detection = surfacing, gap = diving
        self._track_frames: dict[int, set[int]] = {}
        for tid, pts in self._positions.items():
            self._track_frames[tid] = {fr for fr, _, _ in pts}

        # Detect surfacing bouts per track
        self._bouts: dict[int, list[tuple[int, int]]] = {}
        gap_threshold = int(fps * 1.0)  # >1s gap = dive
        for tid, pts in self._positions.items():
            frames_sorted = sorted(self._track_frames[tid])
            if not frames_sorted:
                continue
            bouts = []
            bout_start = frames_sorted[0]
            prev = frames_sorted[0]
            for fr in frames_sorted[1:]:
                if fr - prev > gap_threshold:
                    bouts.append((bout_start, prev))
                    bout_start = fr
                prev = fr
            bouts.append((bout_start, prev))
            self._bouts[tid] = bouts

        # Load blow-spray-based respiration events if available
        self._breaths: list[dict] = []  # [{peak_frame, start_frame, end_frame, ...}]
        self._has_resp = False
        if resp_csv is not None and resp_csv.exists():
            with open(resp_csv, newline="") as f:
                for row in csv.DictReader(f):
                    self._breaths.append({
                        "peak_frame": int(row["peak_frame"]),
                        "start_frame": int(row["start_frame"]),
                        "end_frame": int(row["end_frame"]),
                        "breath_id": int(row["breath_id"]),
                    })
            self._has_resp = True

    def get_state(self, frame_num: int) -> dict:
        """Get all metrics for a given frame."""
        state = {
            "body_length_m": self.body_length_m,
            "body_length_px": self.body_length_px,
            "speeds": {},       # {tid: speed_m_s}
            "surfacing": {},    # {tid: True/False}
            "bout_index": {},   # {tid: (current_bout, total_bouts)}
            "resp_rate": {},    # {tid: breaths_per_min}
            "is_breathing": False,  # True during a blow event
            "breath_count": 0,
        }

        for tid in self._positions:
            # Speed
            speed = self.frame_speed.get(frame_num, {}).get(tid, None)
            if speed is not None:
                state["speeds"][tid] = speed

            # Surfacing state
            is_surfacing = frame_num in self._track_frames.get(tid, set())
            state["surfacing"][tid] = is_surfacing

            # Current bout and total bouts up to now
            bouts = self._bouts.get(tid, [])
            total_bouts = 0
            current_bout = 0
            for i, (bs, be) in enumerate(bouts):
                if bs <= frame_num:
                    total_bouts = i + 1
                    if bs <= frame_num <= be:
                        current_bout = i + 1
            state["bout_index"][tid] = (current_bout, total_bouts)

            # Respiration — use blow-spray detection if available
            if self._has_resp:
                breaths_so_far = sum(1 for b in self._breaths
                                     if b["peak_frame"] <= frame_num)
                state["breath_count"] = breaths_so_far

                # Check if currently in a blow event
                for b in self._breaths:
                    if b["start_frame"] <= frame_num <= b["end_frame"]:
                        state["is_breathing"] = True
                        break

                # Rate from actual breaths
                first_frame = self._positions[tid][0][0]
                elapsed_s = (frame_num - first_frame) / self.fps if self.fps > 0 else 1
                if elapsed_s > 5 and breaths_so_far > 0:
                    state["resp_rate"][tid] = breaths_so_far / (elapsed_s / 60.0)
            else:
                # Fallback: bout-based (less accurate)
                first_frame = self._positions[tid][0][0]
                elapsed_s = (frame_num - first_frame) / self.fps if self.fps > 0 else 1
                if elapsed_s > 5:
                    state["resp_rate"][tid] = total_bouts / (elapsed_s / 60.0)

        return state

    def draw(self, frame: np.ndarray, frame_num: int):
        """Draw metrics panel on the right side of the frame."""
        fh, fw = frame.shape[:2]
        state = self.get_state(frame_num)

        # Panel dimensions
        panel_w = 520
        panel_h = 280
        margin = 20
        px1 = fw - panel_w - margin
        py1 = margin + 85  # below HUD bar
        px2 = fw - margin
        py2 = py1 + panel_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 120), 2,
                      cv2.LINE_AA)

        # Title
        cv2.putText(frame, "METRICS", (px1 + 12, py1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 200), 1,
                    cv2.LINE_AA)

        y = py1 + 55
        line_h = 38
        label_x = px1 + 16
        val_x = px1 + 220

        # 1) Body size calibration
        cv2.putText(frame, "Body length:", (label_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, f"{state['body_length_m']:.1f}m  ({state['body_length_px']:.0f}px)",
                    (val_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        y += line_h

        # Per-track live metrics
        for tid in sorted(self._positions.keys()):
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]

            # 2) Surfacing state
            is_surf = state["surfacing"].get(tid, False)
            surf_label = "SURFACE" if is_surf else "DIVING"
            surf_color = (0, 255, 180) if is_surf else (100, 100, 255)
            bout_cur, bout_total = state["bout_index"].get(tid, (0, 0))

            cv2.putText(frame, "State:", (label_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, surf_label, (val_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, surf_color, 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"  bout {bout_cur}/{bout_total}",
                        (val_x + 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 160), 1,
                        cv2.LINE_AA)
            y += line_h

            # 3) Compensated speed
            speed = state["speeds"].get(tid, None)
            cv2.putText(frame, "Speed:", (label_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1,
                        cv2.LINE_AA)
            if speed is not None:
                bl_s = speed / self.body_length_m  # body lengths per second
                cv2.putText(frame, f"{speed:.1f} m/s  ({bl_s:.2f} BL/s)",
                            (val_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, "---", (val_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1,
                            cv2.LINE_AA)
            y += line_h

            # 4) Respiration rate
            resp = state["resp_rate"].get(tid, None)
            cv2.putText(frame, "Respiration:", (label_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1,
                        cv2.LINE_AA)
            if resp is not None:
                breath_txt = f"{resp:.1f} br/min"
                if self._has_resp:
                    breath_txt += f"  ({state['breath_count']} breaths)"
                cv2.putText(frame, breath_txt,
                            (val_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, "calculating...", (val_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1,
                            cv2.LINE_AA)
            y += line_h

        # Breath flash indicator
        if state["is_breathing"]:
            cv2.putText(frame, "BREATH", (px1 + panel_w - 130, py1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2,
                        cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Render tracked video with bbox + trail + pose overlay")
    parser.add_argument("track_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--csv", type=str, default=None,
                        help="Track CSV (auto-detects: unified > relinked > tracks)")
    parser.add_argument("--sample-rate", type=int, default=3,
                        help="Frame sample rate used during tracking (default: 3)")
    parser.add_argument("--minimap-size", type=int, default=500,
                        help="Minimap size in pixels (default: 500)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide keypoint labels")
    parser.add_argument("--body-length", type=float, default=7.0,
                        help="Estimated whale body length in meters (default: 7.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video filename (default: auto)")
    args = parser.parse_args()

    # Find track CSV
    if args.csv:
        tracks_csv = args.track_dir / args.csv
    else:
        for name in ["tracks_unified.csv", "tracks_relinked.csv", "tracks.csv"]:
            tracks_csv = args.track_dir / name
            if tracks_csv.exists():
                break
    if not tracks_csv.exists():
        print(f"Error: No track CSV found in {args.track_dir}")
        sys.exit(1)

    # Find pose CSV
    pose_csv = args.track_dir / "pose" / "pose_keypoints.csv"
    has_pose = pose_csv.exists()

    print(f"Track CSV:  {tracks_csv.name}")
    print(f"Pose CSV:   {'pose_keypoints.csv' if has_pose else 'NOT FOUND (bbox-only mode)'}")

    # Load data
    tracks_by_frame = load_tracks(tracks_csv)
    pose_by_frame = load_pose(pose_csv) if has_pose else {}

    all_frames = sorted(set(tracks_by_frame.keys()) | set(pose_by_frame.keys()))
    print(f"Frames with data: {len(all_frames)} (range {all_frames[0]}-{all_frames[-1]})")

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps / args.sample_rate

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Output FPS: {effective_fps:.1f} (sample_rate={args.sample_rate})")

    # Output path
    if args.output:
        out_path = args.track_dir / args.output
    else:
        stem = args.video.stem
        out_path = args.track_dir / f"{stem}_tracked_pose.mp4"

    out_path_raw = out_path.with_name(out_path.stem + "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(out_path_raw), fourcc, effective_fps, (width, height))

    # Build minimap from compensated coordinates (camera-drift-free)
    comp_csv = args.track_dir / "tracks_compensated.csv"
    if comp_csv.exists():
        minimap_tracks = load_tracks(comp_csv)
        print(f"Minimap: using compensated coordinates ({comp_csv.name})")
    else:
        minimap_tracks = tracks_by_frame
        print("Minimap: using raw coordinates (no compensated CSV found)")
    minimap = Minimap(minimap_tracks, size=args.minimap_size)
    print(f"  {len(minimap.trajectories)} track(s)")

    # Build metrics panel from compensated coordinates
    resp_csv = args.track_dir / "respirations.csv"
    metrics_panel = MetricsPanel(
        comp_tracks=minimap_tracks,  # already compensated if available
        all_frames=all_frames,
        fps=fps,
        sample_rate=args.sample_rate,
        body_length_m=args.body_length,
        resp_csv=resp_csv,
    )
    print(f"Metrics: body={metrics_panel.body_length_px:.0f}px "
          f"≈ {args.body_length}m, {metrics_panel.px_per_m:.1f} px/m")
    if metrics_panel._has_resp:
        print(f"  Respiration: {len(metrics_panel._breaths)} breath events loaded (blow-spray detection)")
    else:
        print("  Respiration: using surfacing-bout fallback (run detect_respirations.py for better accuracy)")

    frame_idx = 0
    written = 0
    target_set = set(all_frames)

    print(f"\nRendering annotated video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx not in target_set:
            frame_idx += 1
            continue

        annotated = frame.copy()

        # Get tracks for this frame
        frame_tracks = tracks_by_frame.get(frame_idx, [])
        frame_pose = pose_by_frame.get(frame_idx, {})

        n_kps = 0

        for t in frame_tracks:
            tid = t["track_id"]
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label
            label = f"Whale {tid}  {t['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Pose skeleton
            if tid in frame_pose:
                kps = frame_pose[tid]
                n_kps = len(kps)
                draw_skeleton(annotated, kps, tid,
                              label=not args.no_labels)

        # Minimap
        active_ids = {t["track_id"] for t in frame_tracks}
        minimap.draw(annotated, frame_idx, active_ids)

        # Metrics panel
        metrics_panel.draw(annotated, frame_idx)

        # HUD
        draw_hud(annotated, frame_idx, fps, len(frame_tracks), n_kps,
                 args.sample_rate)

        out.write(annotated)
        written += 1

        if written % 50 == 0:
            print(f"  Written {written}/{len(all_frames)} frames...")

        frame_idx += 1

    cap.release()
    out.release()

    # Re-encode AVI → H.264 MP4 via ffmpeg
    import shutil as _shutil, subprocess as _sp
    if _shutil.which("ffmpeg"):
        print("\n  Re-encoding to H.264 MP4 via ffmpeg...")
        _sp.run([
            "ffmpeg", "-y", "-i", str(out_path_raw),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(out_path)
        ], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        if out_path.exists():
            out_path_raw.unlink()
        else:
            out_path = out_path_raw
    else:
        out_path = out_path_raw.rename(out_path_raw.with_suffix(".avi"))

    print(f"\n{'='*55}")
    print(f"TRACKED + POSE VIDEO COMPLETE")
    print(f"{'='*55}")
    print(f"  Frames written: {written}")
    print(f"  Output: {out_path}")
    print(f"  Duration: {written/effective_fps:.1f}s @ {effective_fps:.1f}fps")


if __name__ == "__main__":
    main()
