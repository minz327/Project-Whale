# ProjectWhale — Progress & Continuation Guide

> Last updated: 2026-05-15
>
> **Current Focus:** Multi-whale tracking validated. Pipeline works end-to-end on both single-whale and multi-whale clips.
> Next: simplify `relink_tracks.py` (largely redundant now), run remaining clips.

---

## Goal

Build a minimal end-to-end POC pipeline for whale detection, tracking, and behaviour analysis from aerial drone footage. See [poc-goal.md](poc-goal.md) for full scope and research questions. Detailed findings in [poc-findings.md](poc-findings.md).

---

## Pipeline Overview

```
Video (.MP4)
  │
  ├─ 1. track_whales.py        → outputs/track/{clip}/tracks.csv + tracked video
  │      (Norfair + YOLO-World detection + ReID + camera motion compensation)
  │      Replaces old detect→track→relink pipeline in a single step.
  │
  ├─ 2. compensate_tracks.py   → tracks_compensated.csv (optical flow ego-motion removal)
  ├─ 3. pose_estimate.py       → pose/pose_keypoints.csv (SLEAP, needs .venv_sleap)
  ├─ 4. detect_respirations.py → respirations.csv (blowhole brightness)
  ├─ 5. track_metrics.py       → metrics.json + track_metrics.png
  ├─ 6. visualize_tracks.py    → trajectory_map.png + track_timeline.png
  └─ 7. render_tracked_video.py → {clip}_tracked_pose.mp4 (full annotated video)

  Optional / legacy:
  ├─ detect_yoloworld.py       → standalone detection (replaced by track_whales.py)
  ├─ evaluate_detections.py    → detection QA review HTML
  ├─ relink_tracks.py          → post-hoc track merging (largely redundant with Norfair ReID)
  └─ unify_single_whale.py     → force-merge all tracks for known single-whale clips
```

---

## Environment Setup

| Environment | Purpose | Activation |
|---|---|---|
| `.venv` | Detection, tracking, metrics, visualization | `.\.venv\Scripts\Activate.ps1` |
| `.venv_sleap` | Pose estimation (TensorFlow + SLEAP) | `.\.venv_sleap\Scripts\Activate.ps1` |

**Note:** `requirements.txt` only lists `yt-dlp`. ML dependencies (ultralytics, opencv, torch, tensorflow, etc.) are installed directly in the venvs but not pinned.

---

## Scripts Reference

| Script | Venv | Purpose |
|---|---|---|
| `detect_whales.py` | `.venv` | Baseline COCO YOLOv8 detection (Exp 1 — failed, 0 whale detections) |
| `detect_yoloworld.py` | `.venv` | Open-vocab YOLO-World detection (Exp 2 — works, 80%+ accuracy) |
| `download_clips.py` | `.venv` | Downloads clips from `data/clip_log.csv` via yt-dlp |
| `evaluate_detections.py` | `.venv` | Cross-class NMS + HTML review grid for human QA |
| `track_whales.py` | `.venv` | YOLO-World + **Norfair** tracking (IoU + ReID + camera motion comp.) |
| `relink_tracks.py` | `.venv` | Merges fragmented tracks via spatial + ResNet18 (largely redundant with Norfair ReID) |
| `unify_single_whale.py` | `.venv` | For single-whale clips: merges all significant tracks into one ID, drops noise |
| `compensate_tracks.py` | `.venv` | Optical flow ego-motion compensation (subtracts camera drift from tracks) |
| `track_metrics.py` | `.venv` | Relative metrics with body-length calibration + pose-derived heading |
| `visualize_tracks.py` | `.venv` | Static trajectory map + timeline chart |
| `pose_estimate.py` | `.venv_sleap` | SLEAP pose estimation using Ren's pre-trained model (7 keypoints) |
| `visualize_pose.py` | `.venv` | Draws skeletons + keypoints on frames, generates HTML review |

---

## Clip Processing Status

| Video Clip | Whales | track | compensate | pose | respirations | metrics | visualize | render |
|---|---|---|---|---|---|---|---|---|
| **20231018-40_trim** | 1 | ✅ (2 IDs) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **20240727-84** ⭐ | 3 | ✅ (3 IDs) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **20240527-22** | ? | old ByteTrack | — | ✅ | — | ✅ | ✅ | — |
| srkw_calf_drone | 2 | — | — | — | — | — | — | — |
| 20231018-48_trim | ? | — | — | — | — | — | — | — |
| 20240127-7_trim | ? | — | — | — | — | — | — | — |

⭐ = **First multi-whale clip** — full pipeline validated with Norfair tracker.

**Note:** Clips marked "old ByteTrack" should be re-run with `track_whales.py` (now uses Norfair).

Videos are in `videos/` folder.

---

## Key Findings So Far

1. **Detection works zero-shot** — YOLO-World with text prompts ("whale", "orca", etc.) at conf ≥ 0.1 achieves 80%+ accuracy. Standard COCO YOLOv8 completely fails.
2. **Norfair tracker solves ID fragmentation** — Two-tier matching (IoU short-term + ResNet18 ReID long-term) with camera motion compensation reduced single-whale IDs from 13 → 2. Multi-whale (3 whales) tracked as 3 stable IDs across full 70s video.
3. **ByteTrack is superseded** — IoU-only matching with no appearance features. Norfair replaces it entirely.
4. **ReID bridges dive gaps** — ResNet18 cosine similarity (threshold 0.5) re-identifies whales after 15+ second dives. `relink_tracks.py` now finds 0 candidates to merge.
5. **Post-processing noise filter works** — `--min-track-conf 0.25` and `--min-track-frames 10` drop false positive tracks (avg conf 0.17-0.22) while keeping real whale tracks (conf 0.26-0.50).
6. **Absolute trajectories are unusable** — Camera motion dominates. Pivoted to relative/pairwise metrics (inter-whale distance, heading, speed) which are camera-motion-invariant.
7. **Pose estimation runs** — Two-stage top-down pipeline: centroid model finds saddle patch center, then centered-instance model predicts 7 keypoints. Achieves 6.2-6.5/7 keypoints avg.
8. **Video encoding** — Must use XVID + ffmpeg H.264 re-encode. OpenCV's `mp4v` codec produces unplayable files.

---

## Tracking Results Comparison

### Single-whale clip (20231018-40_trim, 86s, 1 whale)

| Tracker | Track IDs | Longest track | Coverage |
|---------|-----------|---------------|----------|
| ByteTrack (buffer=30) | 13 | 168 frames | 56% |
| ByteTrack (buffer=150) | 9 | 219 frames | 61% |
| **Norfair+ReID+GMC** | **5 → 2** (after noise filter) | **695 frames** | **93%** |

### Multi-whale clip (20240727-84, 70s, 3 whales)

| Tracker | Track IDs | Avg frames/track | Coverage |
|---------|-----------|-------------------|----------|
| **Norfair+ReID+GMC** | **6 → 3** (after noise filter) | **556 frames** | **99%** |

---

## Top Blockers

1. **Camera motion / no telemetry** — Without DJI SRT files, can't geo-reference trajectories
2. **Processing speed** — 4K frames are slow on CPU
3. **Respiration detection needs tuning** — Current brightness-based approach only found 1 breath on multi-whale clip

---

## Output Locations

```
outputs/
├── detect/srkw_calf_drone/         — COCO baseline (failed experiment)
├── detect_world/
│   ├── 20240527-22/                 — YOLO-World detections + review.html
│   ├── 20240727-84/                 — YOLO-World detections + review.html
│   └── srkw_calf_drone/            — YOLO-World detections + review.html
├── track/20231018-40_trim/           ⭐ FOCUS CLIP (single whale)
│   ├── 20231018-40_trim_tracked.mp4  — annotated tracking video (bbox only)
│   ├── tracks.csv                    — raw ByteTrack output (13 tracks)
│   ├── tracks_relinked.csv           — post-relink (9 tracks)
│   ├── tracks_unified.csv            — single whale ID (1 track, 469 points)
│   ├── tracks_compensated.csv        — ego-motion compensated
│   ├── ego_motion.json               — camera drift data
│   ├── metrics.json + track_metrics.png
│   ├── trajectory_map_unified.png + trajectory_map_compensated.png
│   ├── track_timeline_unified.png + track_timeline_compensated.png
│   └── pose/
│       ├── pose_keypoints.csv        — 7 keypoints × 469 frames
│       ├── pose_results.json
│       ├── pose_review.html          — visual QA page
│       ├── frames/                   — 30 annotated frame JPGs
│       └── debug_frames/             — centroid debug crops
└── track/20240527-22/
    ├── 20240527-22_tracked.mp4      — annotated tracking video
    ├── tracks.csv / tracks_relinked.csv
    ├── metrics.json + track_metrics.png
    ├── trajectory_map*.png + track_timeline*.png
    └── pose/                        — 7 keypoints, pose_review.html
```

---

## Next Steps

### Current Focus — 20231018-40_trim (Single Whale Reference Clip)

Goal: Match Ren's DORSAP quality for this clip. Full pipeline is now complete.

**Done so far:**
- Detection → tracking → relinking (13→9 tracks) → unification (→1 whale, 469 frames)
- Ego-motion compensation (6134px camera drift removed)
- Metrics: 11 surfacing bouts, body-length calibrated (562px ≈ 7.0m)
- Pose: 6.5/7 avg keypoints, rostrum 99%, saddle_patch 100%, right_caudal_fluke 97%
- Visualizations: trajectory maps (raw + compensated), timeline, metrics, pose review

**Quality to improve:**
1. **Low-confidence keypoints** — left_caudal_fluke (68% detection, 0.15 avg conf), left_pect_fin (85%, 0.17 conf). These may be occluded by body orientation.
2. **Pose-derived heading** — rostrum→caudal axis works but 0% heading from pose in metrics (needs fix in track_metrics.py to use unified track)
3. **Tracked video with pose overlay** — current `_tracked.mp4` has bounding boxes only; should overlay skeleton keypoints
4. **Review `pose_review.html`** — human QA on keypoint placement accuracy

### After Single-Whale Quality is Confirmed

5. Run pipeline on **20240727-84** (multi-whale, detection already done)
6. Run pipeline on **20231018-48_trim** and **20240127-7_trim**
7. Compare pose quality across clips

### Pipeline Improvements

8. **Pin dependencies** — Export `pip freeze` from both venvs to proper requirements files
9. **Obtain DJI SRT telemetry files** — Enables geo-referencing and absolute trajectory analysis
10. **Fine-tune YOLO on whale boxes** (~100 annotated frames) to improve detection confidence
11. **Build unified pipeline script** — Single command: video → all outputs

---

## Exeter / Ren's Work

Ren's DORSAP project from Exeter is stored in `exeter/`. Key assets used:
- **Centroid model**: `exeter/models/full_FGM_v1_250327_234344.centroid.n=231/best_model.h5` — finds saddle patch center on full frame (requires 1360×2560 input due to U-Net skip connections)
- **Centered-instance model**: `exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231/best_model.h5` — 7-keypoint prediction on 832×832 crop centered on centroid → 208×208×7 confidence maps
- **Skeleton definition**: `exeter/basic_aerial_skeleton.json`
- Two-stage pipeline: centroid on full frame → crop centered on saddle patch → instance model

---

## Common Commands

```powershell
# Activate main venv
.\.venv\Scripts\Activate.ps1

# Activate SLEAP venv
.\.venv_sleap\Scripts\Activate.ps1

# Full pipeline for a new clip (replace CLIP_ID and VIDEO_PATH)
python scripts/detect_yoloworld.py VIDEO_PATH --sample-rate 30 --conf 0.05
python scripts/evaluate_detections.py outputs/detect_world/CLIP_ID
python scripts/track_whales.py VIDEO_PATH --sample-rate 3 --conf 0.1
python scripts/relink_tracks.py outputs/track/CLIP_ID
python scripts/track_metrics.py outputs/track/CLIP_ID
python scripts/visualize_tracks.py outputs/track/CLIP_ID --csv tracks_relinked.csv --suffix _relinked

# Switch to SLEAP venv for pose
.\.venv_sleap\Scripts\Activate.ps1
python scripts/pose_estimate.py outputs/track/CLIP_ID VIDEO_PATH

# Switch back for pose visualization
.\.venv\Scripts\Activate.ps1
python scripts/visualize_pose.py outputs/track/CLIP_ID VIDEO_PATH
```

### Single-Whale Clip Pipeline (e.g., 20231018-40_trim)

```powershell
# After relink, unify all tracks into single whale ID:
python scripts/unify_single_whale.py outputs/track/CLIP_ID

# Then compensate + metrics + visualize using unified CSV:
python scripts/compensate_tracks.py outputs/track/CLIP_ID VIDEO_PATH --csv tracks_unified.csv
python scripts/track_metrics.py outputs/track/CLIP_ID --csv tracks_unified.csv
python scripts/visualize_tracks.py outputs/track/CLIP_ID --csv tracks_unified.csv --suffix _unified
python scripts/visualize_tracks.py outputs/track/CLIP_ID --csv tracks_compensated.csv --suffix _compensated

# Pose on unified track:
.\.venv_sleap\Scripts\Activate.ps1
python scripts/pose_estimate.py outputs/track/CLIP_ID VIDEO_PATH --csv tracks_unified.csv

.\.venv\Scripts\Activate.ps1
python scripts/visualize_pose.py outputs/track/CLIP_ID VIDEO_PATH
```
