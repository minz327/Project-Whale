# Single Whale Analysis — 20231018-40_trim

> Clip: `20231018-40_trim.mp4` | Resolution: 3840×2160 (4K) | 30fps | ~86s  
> Species: Southern Resident Killer Whale (SRKW) | Aerial drone footage  
> Date processed: 2026-05-14

---

## Summary

This is the reference clip for the ProjectWhale POC — a single orca tracked from aerial drone footage through an end-to-end pipeline: detection → tracking → re-linking → unification → ego-motion compensation → metrics → pose estimation → annotated video.

**Key results:**
- 469 frames with whale detected across 86 seconds of footage
- Single whale successfully unified from 13 fragmented track IDs into 1
- 7-keypoint pose skeleton predicted on every detected frame (avg 6.5/7 keypoints)
- 11 surfacing bouts detected with 0.8–4.0s dive gaps
- Camera drift of 6,134 pixels removed via optical flow compensation
- Final output: annotated video with bounding box, skeleton, minimap, and live metrics

---

## Pipeline Steps (Reproducible)

### Prerequisites

Two Python virtual environments:

| Venv | Purpose | Activation |
|------|---------|------------|
| `.venv` | Detection, tracking, metrics, visualization | `.\.venv\Scripts\Activate.ps1` |
| `.venv_sleap` | Pose estimation (TensorFlow + SLEAP models) | `.\.venv_sleap\Scripts\Activate.ps1` |

### Step 1 — Detection (YOLO-World)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/detect_yoloworld.py videos/20231018-40_trim.mp4 --sample-rate 10 --conf 0.05
```

**What it does:** Runs YOLO-World (yolov8x-worldv2) with open-vocabulary text prompts ("whale", "orca", "killer whale", "dolphin", "dorsal fin", "marine mammal"). No fine-tuning needed — zero-shot detection.

**Output:** `outputs/detect_world/20231018-40_trim/`

| File | Contents |
|------|----------|
| `detections.csv` | Per-frame bounding boxes with class + confidence |
| `detections_merged.csv` | After cross-class NMS (removes duplicate labels on same whale) |
| `review.html` | Visual grid for human QA |
| `summary.json` | Detection statistics |
| `frame_*_det.jpg` | Annotated frame images |

**Results:**
- 259 frames processed, 231 with detections (89%)
- 491 raw detections → reduced by cross-class NMS
- Detections split across classes: whale (256), marine mammal (213), dorsal fin (21), killer whale (1)
- Mean confidence: whale 0.30, marine mammal 0.37

**QA:** Open `review.html` in a browser to visually verify detections.

---

### Step 2 — Evaluation / Review

```powershell
python scripts/evaluate_detections.py outputs/detect_world/20231018-40_trim
```

**What it does:** Applies cross-class NMS to merge overlapping detections from different class labels (e.g., "whale" + "marine mammal" on the same animal) and generates an HTML review page.

**Output:** `review.html` with detection grid for manual inspection.

---

### Step 3 — Tracking (Norfair + ReID + GMC)

```powershell
python scripts/track_whales.py videos/20231018-40_trim.mp4 --sample-rate 3 --conf 0.1 --output-dir outputs/track/20231018-40_trim
```

**What it does:** Runs YOLO-World detection on every 3rd frame, then feeds results to **Norfair** for multi-object tracking with two-tier matching:
- **Short-term (IoU)**: distance_threshold=0.7, hit_counter_max=15 frames
- **Long-term (ReID)**: ResNet18 embeddings, cosine similarity threshold=0.5, reid_counter_max=150 (~15s)
- **Camera motion compensation**: HomographyTransformationGetter (500 keypoints)
- **Post-processing noise filter**: drops tracks with avg_conf < 0.25 or < 10 frames

**Parameters:**
- `--sample-rate 3` → processes ~10fps (every 3rd frame of 30fps video)
- `--conf 0.1` → low threshold to catch dim/partial surfacings
- `--min-track-conf 0.25` → filter noise tracks (default)
- `--min-track-frames 10` → filter very short tracks (default)

**Output:** `outputs/track/20231018-40_trim/`

| File | Contents |
|------|----------|
| `tracks.csv` | Per-frame track ID, bounding box, center, confidence |
| `20231018-40_trim_tracked.mp4` | Video with bounding boxes and trail lines (H.264) |
| `summary.json` | Track statistics |

**Results:**
- 862 frames processed, ~800 with tracks (93%)
- 5 raw IDs → **2 after noise filter** (was 13 IDs with old ByteTrack)
- Track 1: 695 frames, span 3→2421 (~80s of 86s video) — main whale
- Track 21: 131 frames, concurrent detection overlapping Track 1

**Improvement over ByteTrack:**
| Metric | ByteTrack (old) | Norfair (current) |
|--------|-----------------|-------------------|
| Track IDs | 13 | 2 |
| Longest track | 168 frames | 695 frames |
| Coverage | 56% | 93% |

---

### Step 4 — Re-linking (Appearance Matching)

```powershell
python scripts/relink_tracks.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --min-similarity 0.4 --max-dist 2000 --max-gap 300
```

**What it does:** Merges fragmented track IDs by comparing spatial proximity and visual appearance (ResNet18 embeddings) between track endpoints. Tracks that end near where another starts, and look similar, are merged.

**Parameters (relaxed for single-whale clip):**
- `--min-similarity 0.4` → lower cosine similarity threshold (default 0.7)
- `--max-dist 2000` → allow larger pixel gaps (camera drift makes whale appear to jump)
- `--max-gap 300` → merge across up to 10 seconds of dive gap

**Output:**

| File | Contents |
|------|----------|
| `tracks_relinked.csv` | Merged track data |
| `relink_summary.json` | Merge map and parameters |

**Results:**
- 4 merge operations: Tracks 4,5,6,9 → Track 3; Track 11 stayed separate
- 13 tracks → 9 tracks (3 with ≥5 frames)

---

### Step 5 — Track Unification (Single Whale)

```powershell
python scripts/unify_single_whale.py outputs/track/20231018-40_trim
```

**What it does:** For clips known to contain one whale, merges all significant tracks (≥5 frames) into a single ID and drops noise fragments. This is an **optional** step — only used for single-whale clips.

**Output:**

| File | Contents |
|------|----------|
| `tracks_unified.csv` | Single whale as Track 1, 469 points |
| `unify_summary.json` | Merge details |

**Results:**
- 3 significant tracks (2, 3, 11) merged into Track 1
- 6 noise fragments dropped (14 points from 1–4 frame tracks)
- Final: **1 whale, 469 frames, range 63–2151**
- 5 surfacing gaps detected: 39, 66, 57, 108, 120 frames

---

### Step 6 — Ego-Motion Compensation

```powershell
python scripts/compensate_tracks.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --csv tracks_unified.csv
```

**What it does:** Estimates camera motion from background features using sparse optical flow (Shi-Tomasi corners + Lucas-Kanade), then subtracts accumulated camera drift from track positions. Produces trajectories that reflect the whale's actual movement, not the drone's.

**Why needed:** The drone moves constantly — 6,134 pixels of total camera drift across the clip. Without compensation, the whale's path looks chaotic even though it's swimming straight.

**Output:**

| File | Contents |
|------|----------|
| `tracks_compensated.csv` | Drift-free positions (same columns, corrected center_x/center_y) |
| `ego_motion.json` | Per-frame camera displacement vectors |

**Results:**
- 697 frames with flow measurements, 689 valid samples
- Total camera drift removed: 1,620px horizontal + 5,916px vertical = 6,134px

**Important:** Use `tracks_compensated.csv` for any trajectory, speed, or spatial analysis. Raw coordinates are only valid for on-frame overlays (bboxes, keypoints).

---

### Step 7 — Metrics

```powershell
python scripts/track_metrics.py outputs/track/20231018-40_trim --csv tracks_unified.csv
```

**What it does:** Computes body-length-calibrated metrics from track data.

**Output:**

| File | Contents |
|------|----------|
| `metrics.json` | All computed metrics |
| `track_metrics.png` | Speed and surfacing plots |

**Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| Body length | 562px ≈ 7.0m | Bbox diagonal, assumed adult SRKW |
| Scale | 80.3 px/m | Derived from body length |
| Avg speed | 160 px/s (0.29 BL/s) | Raw pixel speed (includes camera noise) |
| Surfacing bouts | 11 | Based on detection presence |
| Bout durations | 0.9s – 13.2s | Continuous detection runs |
| Dive gaps | 0.8s – 4.0s | All short — whale mostly at surface |

**Limitations:**
- Speed uses raw pixel coords (camera-motion contaminated)
- Surfacing = "detection present" — conflates missed detections with dives
- Body length is assumed, not measured from altitude

---

### Step 8 — Pose Estimation (SLEAP)

```powershell
.\.venv_sleap\Scripts\Activate.ps1
python scripts/pose_estimate.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --csv tracks_unified.csv
```

**What it does:** Two-stage top-down pose estimation using Ren's pre-trained SLEAP models:

1. **Centroid model** — runs on full frame downscaled to 1360×2560, finds the saddle patch center via confidence map peak
2. **Centered instance model** — crops 832×832 around the centroid, predicts 7 keypoint confidence maps (208×208×7), extracts peak locations

**Models used:**
- Centroid: `exeter/models/full_FGM_v1_250327_234344.centroid.n=231/best_model.h5`
- Instance: `exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231/best_model.h5`

**7 Keypoints (skeleton):**

```
rostrum_tip ── mid_saddle_patch ── Caudal_peduncle ── left_caudal_fluke
                    │                     │
              left_pect_fin_tip    right_caudal_fluke
              right_pect_fin_tip
```

1. `rostrum_tip` — nose/beak
2. `mid_saddle_patch` — distinctive grey patch behind dorsal fin (anchor point)
3. `Caudal_peduncle` — tail base
4. `left_caudal_fluke` / `right_caudal_fluke` — tail tips
5. `left_pect_fin_tip` / `right_pect_fin_tip` — pectoral fin tips

**Output:** `outputs/track/20231018-40_trim/pose/`

| File | Contents |
|------|----------|
| `pose_keypoints.csv` | x, y, confidence per keypoint per frame |
| `pose_results.json` | Full structured results |

**Results:**

| Keypoint | Detection rate | Avg confidence |
|----------|---------------|----------------|
| rostrum_tip | 99% (466/469) | 0.719 |
| mid_saddle_patch | 100% (467/469) | 0.784 |
| right_caudal_fluke | 97% (453/469) | 0.356 |
| Caudal_peduncle | 98% (459/469) | 0.192 |
| right_pect_fin_tip | 100% (468/469) | 0.295 |
| left_pect_fin_tip | 85% (400/469) | 0.168 |
| left_caudal_fluke | 68% (320/469) | 0.153 |

- Average keypoints per frame: **6.5 / 7**
- 60% of frames have all 7 keypoints
- 0 centroid fallbacks (SLEAP centroid model found saddle patch in every frame)

---

### Step 9 — Pose Visualization

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/visualize_pose.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4
```

**Output:** `outputs/track/20231018-40_trim/pose/`

| File | Contents |
|------|----------|
| `pose_review.html` | HTML page with 30 annotated frames for QA |
| `frames/` | 30 JPGs with skeleton + keypoint labels drawn on whale |

---

### Step 10 — Trajectory Visualization

```powershell
python scripts/visualize_tracks.py outputs/track/20231018-40_trim --csv tracks_unified.csv --suffix _unified
python scripts/visualize_tracks.py outputs/track/20231018-40_trim --csv tracks_compensated.csv --suffix _compensated
```

**Output:**

| File | Contents |
|------|----------|
| `trajectory_map_unified.png` | Raw pixel trajectory (shows camera drift — NOT useful) |
| `trajectory_map_compensated.png` | Drift-free trajectory (real whale path) |
| `track_timeline_unified.png` | Frame presence timeline |

---

### Step 11 — Final Annotated Video

```powershell
python scripts/render_tracked_video.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --sample-rate 3
```

**What it does:** Renders a single video combining all pipeline outputs:

- **Bounding box** with "Whale 1" label and confidence
- **7-keypoint skeleton** with colored joints and edge connections
- **Minimap** (top-left) — racing-game-style track map using compensated coordinates, showing full trajectory with current position dot and progress bar
- **Metrics panel** (top-right) — live body calibration, surfacing state (SURFACE/DIVING), compensated speed (m/s and BL/s), respiration rate (breaths/min)
- **HUD** — frame number, timestamp, track count, keypoint count

**Output:** `outputs/track/20231018-40_trim/20231018-40_trim_tracked_pose.mp4`
- 469 frames, 46.9s @ 10fps

---

## Output File Tree

```
outputs/
├── detect_world/20231018-40_trim/
│   ├── detections.csv
│   ├── detections_merged.csv
│   ├── review.html
│   ├── summary.json
│   └── frame_*_det.jpg (259 detection images)
│
└── track/20231018-40_trim/
    ├── tracks.csv                          (raw: 13 tracks)
    ├── tracks_relinked.csv                 (merged: 9 tracks)
    ├── tracks_unified.csv                  (final: 1 whale, 469 points)
    ├── tracks_compensated.csv              (drift-free positions)
    ├── 20231018-40_trim_tracked.mp4        (bbox-only video)
    ├── 20231018-40_trim_tracked_pose.mp4   (full annotated video)
    ├── summary.json
    ├── relink_summary.json
    ├── unify_summary.json
    ├── ego_motion.json
    ├── metrics.json
    ├── track_metrics.png
    ├── trajectory_map_unified.png
    ├── trajectory_map_compensated.png
    ├── track_timeline_unified.png
    ├── track_timeline_compensated.png
    └── pose/
        ├── pose_keypoints.csv
        ├── pose_results.json
        ├── pose_review.html
        ├── frames/ (30 annotated JPGs)
        └── debug_frames/ (centroid debug images)
```

---

## Scripts Reference

| # | Script | Venv | Input | Output |
|---|--------|------|-------|--------|
| 1 | `detect_yoloworld.py` | .venv | video | detections CSV + review HTML |
| 2 | `evaluate_detections.py` | .venv | detection dir | merged detections + review |
| 3 | `track_whales.py` | .venv | video | tracks.csv + tracked video |
| 4 | `relink_tracks.py` | .venv | track dir + video | tracks_relinked.csv |
| 5 | `unify_single_whale.py` | .venv | track dir | tracks_unified.csv |
| 6 | `compensate_tracks.py` | .venv | track dir + video | tracks_compensated.csv |
| 7 | `track_metrics.py` | .venv | track dir | metrics.json + plot |
| 8 | `pose_estimate.py` | .venv_sleap | track dir + video | pose_keypoints.csv |
| 9 | `visualize_pose.py` | .venv | track dir + video | pose_review.html + frames |
| 10 | `visualize_tracks.py` | .venv | track dir | trajectory maps + timelines |
| 11 | `render_tracked_video.py` | .venv | track dir + video | full annotated video |

---

## Quick Reproduce (Full Pipeline)

```powershell
# Step 1-7: Detection through metrics (.venv)
.\.venv\Scripts\Activate.ps1
python scripts/detect_yoloworld.py videos/20231018-40_trim.mp4 --sample-rate 10 --conf 0.05
python scripts/evaluate_detections.py outputs/detect_world/20231018-40_trim
python scripts/track_whales.py videos/20231018-40_trim.mp4 --sample-rate 3 --conf 0.1
python scripts/relink_tracks.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --min-similarity 0.4 --max-dist 2000 --max-gap 300
python scripts/unify_single_whale.py outputs/track/20231018-40_trim
python scripts/compensate_tracks.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --csv tracks_unified.csv
python scripts/track_metrics.py outputs/track/20231018-40_trim --csv tracks_unified.csv

# Step 8: Pose estimation (.venv_sleap)
.\.venv_sleap\Scripts\Activate.ps1
python scripts/pose_estimate.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --csv tracks_unified.csv

# Step 9-11: Visualization (.venv)
.\.venv\Scripts\Activate.ps1
python scripts/visualize_pose.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4
python scripts/visualize_tracks.py outputs/track/20231018-40_trim --csv tracks_compensated.csv --suffix _compensated
python scripts/render_tracked_video.py outputs/track/20231018-40_trim videos/20231018-40_trim.mp4 --sample-rate 3
```

---

## Known Limitations

| Area | Issue | Impact |
|------|-------|--------|
| Surfacing detection | Uses detection presence as proxy | Missed detections count as dives |
| Speed | Raw pixel speed includes camera motion | Use compensated coords for analysis |
| Body length | Assumed 7.0m (adult SRKW) | Unknown if juvenile — need SRT for altitude |
| Left-side keypoints | Lower detection (68–85%) and confidence (0.15–0.17) | Body orientation occludes left side |
| Track unification | Assumes single whale | Must skip this step for multi-whale clips |
| Respiration rate | Counts surfacing bouts / time | Not validated against ground truth |

---

## What This Proves (POC Answers)

1. **Detection works zero-shot** — YOLO-World finds orcas with no training data
2. **Tracking works but fragments** — ByteTrack maintains IDs during surfacing, breaks across dives
3. **Re-linking + unification solves single-whale ID** — 13 fragments → 1 whale
4. **Ego-motion compensation is essential** — 6,134px of camera drift makes raw trajectories useless
5. **Pose estimation works** — Ren's SLEAP model predicts 6.5/7 keypoints on average
6. **Behavioural metrics are measurable** — surfacing bouts, speed, respiration rate all computable from the pipeline
