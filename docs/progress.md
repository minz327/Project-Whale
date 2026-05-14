# ProjectWhale — Progress & Continuation Guide

> Last updated: 2026-05-13

---

## Goal

Build a minimal end-to-end POC pipeline for whale detection, tracking, and behaviour analysis from aerial drone footage. See [poc-goal.md](poc-goal.md) for full scope and research questions. Detailed findings in [poc-findings.md](poc-findings.md).

---

## Pipeline Overview

```
Video (.MP4)
  │
  ├─ 1. detect_yoloworld.py   → outputs/detect_world/{clip}/
  ├─ 2. evaluate_detections.py → review.html (human QA)
  ├─ 3. track_whales.py        → outputs/track/{clip}/tracks.csv + tracked video
  ├─ 4. relink_tracks.py       → tracks_relinked.csv (fix ID breaks)
  ├─ 5. track_metrics.py       → metrics.json + track_metrics.png
  ├─ 6. visualize_tracks.py    → trajectory_map.png + track_timeline.png
  └─ 7. pose_estimate.py       → pose/pose_keypoints.csv + pose_results.json
       └─ visualize_pose.py    → pose/pose_review.html + annotated frames
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
| `track_whales.py` | `.venv` | YOLO-World + ByteTrack multi-object tracking |
| `relink_tracks.py` | `.venv` | Merges fragmented tracks via spatial + ResNet18 appearance embeddings |
| `track_metrics.py` | `.venv` | Camera-invariant relative metrics (speed, inter-whale distance, surfacing) |
| `visualize_tracks.py` | `.venv` | Static trajectory map + timeline chart |
| `pose_estimate.py` | `.venv_sleap` | SLEAP pose estimation using Ren's pre-trained model (7 keypoints) |
| `visualize_pose.py` | `.venv` | Draws skeletons + keypoints on frames, generates HTML review |

---

## Clip Processing Status

| Video Clip | detect_world | evaluate | track | relink | metrics | visualize | pose |
|---|---|---|---|---|---|---|---|
| **20240527-22.MP4** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **20240727-84.MP4** | ✅ | ✅ | — | — | — | — | — |
| srkw_calf_drone | ✅ | ✅ | — | — | — | — | — |
| 20231018-40_trim.mp4 | — | — | — | — | — | — | — |
| 20231018-48_trim.mp4 | — | — | — | — | — | — | — |
| 20240127-7_trim.mp4 | — | — | — | — | — | — | — |

Only **20240527-22** has gone through the full pipeline. Videos are in `videos/` folder.

---

## Key Findings So Far

1. **Detection works zero-shot** — YOLO-World with text prompts ("whale", "orca", etc.) at conf ≥ 0.05 achieves 80%+ accuracy. Standard COCO YOLOv8 completely fails.
2. **Tracking works during surfacing** — ByteTrack maintains IDs for up to 12s of continuous surfacing, but IDs fragment across dives.
3. **Re-linking helps** — ResNet18 appearance embedding re-linking reduced 13 tracks → 9 on test clip.
4. **Absolute trajectories are unusable** — Camera motion dominates. Pivoted to relative/pairwise metrics (inter-whale distance, heading, speed) which are camera-motion-invariant.
5. **Stabilization failed** — Attempted and scrapped; produced bad artifacts.
6. **Pose estimation runs** — Two-stage top-down pipeline: centroid model finds saddle patch center on full frame (1360×2560), then centered-instance model predicts 7 keypoints on 832×832 crop. Achieves 6.5/7 keypoints avg with consistent quality across all frames.

---

## Top Blockers

1. **Camera motion / no telemetry** — Without DJI SRT files, can't geo-reference trajectories
2. **ID fragmentation across dives** — Re-linking helps but isn't perfect
3. **Low detection confidence** — Many detections in 0.05–0.30 range
4. **Processing speed** — 4K frames are slow on CPU

---

## Output Locations

```
outputs/
├── detect/srkw_calf_drone/         — COCO baseline (failed experiment)
├── detect_world/
│   ├── 20240527-22/                 — YOLO-World detections + review.html
│   ├── 20240727-84/                 — YOLO-World detections + review.html
│   └── srkw_calf_drone/            — YOLO-World detections + review.html
└── track/20240527-22/
    ├── 20240527-22_tracked.mp4      — annotated tracking video
    ├── tracks.csv                   — raw track data
    ├── tracks_relinked.csv          — post-relink track data
    ├── summary.json / relink_summary.json
    ├── metrics.json + track_metrics.png
    ├── trajectory_map*.png + track_timeline*.png
    └── pose/
        ├── pose_keypoints.csv       — keypoint coordinates + confidence
        ├── pose_results.json
        ├── pose_review.html         — visual QA page
        └── frames/                  — 30 annotated frame JPGs
```

---

## Next Steps

### Immediate — Complete Pipeline on More Clips

1. Run tracking → relinking → metrics → visualization → pose on **20240727-84**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python scripts/track_whales.py videos/20240727-84.MP4 --sample-rate 3 --conf 0.1
   python scripts/relink_tracks.py outputs/track/20240727-84
   python scripts/track_metrics.py outputs/track/20240727-84
   python scripts/visualize_tracks.py outputs/track/20240727-84 --csv tracks_relinked.csv --suffix _relinked

   .\.venv_sleap\Scripts\Activate.ps1
   python scripts/pose_estimate.py outputs/track/20240727-84 videos/20240727-84.MP4

   .\.venv\Scripts\Activate.ps1
   python scripts/visualize_pose.py outputs/track/20240727-84 videos/20240727-84.MP4
   ```

2. Run full pipeline on the **3 unprocessed trimmed clips** (20231018-40, 20231018-48, 20240127-7)

### Pose Estimation Quality

3. **Review pose results** — Open `outputs/track/20240527-22/pose/pose_review.html` and check keypoint placement accuracy. Decide if SLEAP model needs fine-tuning for aerial footage or if filtering to high-confidence keypoints (rostrum + fluke) is sufficient.

### Pipeline Improvements

4. **Pin dependencies** — Export `pip freeze` from both venvs to proper requirements files
5. **Obtain DJI SRT telemetry files** — Enables geo-referencing and absolute trajectory analysis
6. **Fine-tune YOLO on whale boxes** (~100 annotated frames) to improve detection confidence
7. **Combine pose + tracking** — Use rostrum-to-fluke axis as heading direction for behaviour metrics

### Stretch

8. **Build unified pipeline script** — Single command: video → all outputs
9. **Cross-clip re-identification** — Match individuals across different clips
10. **Focus on measurable behaviours** — Surfacing intervals, proximity patterns, synchronized movements

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
