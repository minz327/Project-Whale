# ProjectWhale — Progress & Continuation Guide

> Last updated: 2026-05-20
>
> **Current Focus:** Quality improvement — fix detection false positives, pose multi-whale assignment, and keypoint stability.
> Focus clips: **20231018-40_trim** (1 whale) and **20240727-84** (3 whales). Other clips wait until quality is validated.

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
| **20231018-40_trim** ⭐ | 1 | ✅ (2 IDs) | ✅ | ✅ (6.5/7¹) | ✅ | ✅ | ✅ | ✅ |
| **20240727-84** ⭐ | 3 | ✅ (3 IDs) | ✅ | ✅ (6.2/7²) | ✅ | ✅ | ✅ | ✅ |
| **20240527-22** | 1 | ✅ (1 ID) | ✅ | ✅ (3.5/7) | — | ✅ | ✅ | ✅ |
| **20231018-48_trim** | 13 | ✅ (13 IDs) | ✅ | ✅ (5.7/7) | — | ✅ | ✅ | ✅ |
| **20240127-7_trim** | 2 | ✅ (2 IDs) | ✅ | ✅ (4.2/7) | — | ✅ | ✅ | ✅ |
| srkw_calf_drone | 2 | — | — | — | — | — | — | — |

⭐ = **Quality focus clips** — all other clips wait until these meet quality bar.

¹ Keypoints are unstable (jitter frame-to-frame) — needs temporal smoothing.
² Keypoints jump between whales — needs Hungarian centroid assignment.

**Note:** All clips have been run through the full Norfair pipeline, but pipeline quality needs improvement before results are usable (see Known Quality Issues below).

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

## Known Quality Issues (Active — 2026-05-20)

### Issue 1: Detection False Positives — `track_whales.py`

**Problem:** Water splashes and waves are detected as whales (visible as small bounding boxes in the tracked video). The detection confidence threshold (`--conf 0.1`) is too low and there is no bbox size filtering.

**Root cause:** YOLO-World at conf=0.1 picks up any vaguely whale-shaped water feature. No min/max bbox area filter exists. The post-tracking noise filter (`--min-track-conf 0.25`) only removes noisy tracks after tracking, it doesn't prevent false detections from entering the tracker.

**Fix plan:**
- [ ] Add `--min-bbox-area` filter (default ~2000px²) to reject tiny splash/wave boxes before tracking
- [ ] Add `--max-bbox-area` filter (default ~500000px²) to reject oversized false positives
- [ ] Raise default `--conf` from 0.1 → 0.15
- [ ] Test on both focus clips

### Issue 2: Pose Multi-Whale Assignment — `pose_estimate.py`

**Problem:** On the multi-whale clip (20240727-84), keypoints jump between whales. The pose output is unusable for multi-whale analysis.

**Root cause:** The centroid model runs ONCE per frame, producing a single heatmap. Each track then greedily searches for the highest peak within 150 pixels of its bbox center. When whales are close together (0.8–1.2 body lengths apart), multiple tracks find the **same** centroid peak → same crop → identical keypoints assigned to different whales.

**Fix plan:**
- [ ] Replace greedy peak search with **Hungarian (one-to-one) assignment**:
  1. Find ALL local maxima in centroid heatmap
  2. Build cost matrix: distance from each peak to each track's bbox center
  3. Use `scipy.optimize.linear_sum_assignment` for one-to-one matching
  4. Tracks without a matched peak fall back to bbox center
- [ ] Test on 20240727-84 — verify keypoints stay on their assigned whale

### Issue 3: Pose Keypoint Instability — `pose_estimate.py`

**Problem:** Even on the single-whale clip (20231018-40_trim), keypoints jitter frame-to-frame. The skeleton overlay in the rendered video looks unstable.

**Root cause:** Each frame is processed independently. The centroid peak position fluctuates slightly each frame, shifting the 832×832 crop, which cascades into all keypoint positions. No temporal smoothing is applied.

**Fix plan:**
- [ ] Add **exponential moving average (EMA) smoothing** per track:
  - Centroid position: alpha ~0.5 (moderate smoothing)
  - Keypoint positions: alpha ~0.3, confidence-weighted (low-confidence → more smoothing)
- [ ] Only smooth within same track_id; reset on new track
- [ ] Test on both focus clips

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

### Phase A: Detection Filtering — `track_whales.py`
- [ ] Add bbox area filters + raise conf threshold
- [ ] Test on 20231018-40_trim and 20240727-84

### Phase B: Pose Multi-Whale Fix — `pose_estimate.py`
- [ ] Implement Hungarian centroid assignment (depends on Phase A)
- [ ] Test on 20240727-84

### Phase C: Pose Temporal Smoothing — `pose_estimate.py`
- [ ] Add EMA smoothing for centroid + keypoints (depends on Phase B)
- [ ] Test on both focus clips

### Phase D: Verification
- [ ] Re-run full pipeline on both focus clips (track → compensate → pose → render)
- [ ] Visual QA of rendered videos
- [ ] Commit + push

### After Quality is Validated
- [ ] Re-run remaining clips with improved pipeline
- [ ] Pin dependencies (export pip freeze)
- [ ] Obtain DJI SRT telemetry files from Darren
- [ ] Fine-tune YOLO on whale boxes (~100 annotated frames)
- [ ] Build unified pipeline script (single command: video → all outputs)

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

# Full pipeline for a clip (Norfair tracker)
python scripts/track_whales.py VIDEO_PATH --output-dir outputs/track/CLIP_ID
python scripts/relink_tracks.py outputs/track/CLIP_ID VIDEO_PATH
python scripts/compensate_tracks.py outputs/track/CLIP_ID VIDEO_PATH --csv tracks.csv
python scripts/track_metrics.py outputs/track/CLIP_ID --csv tracks_compensated.csv
python scripts/visualize_tracks.py outputs/track/CLIP_ID --csv tracks_compensated.csv

# Pose estimation (requires .venv_sleap)
.\.venv_sleap\Scripts\Activate.ps1
python scripts/pose_estimate.py outputs/track/CLIP_ID VIDEO_PATH --csv tracks.csv

# Render annotated video (requires .venv)
.\.venv\Scripts\Activate.ps1
python scripts/render_tracked_video.py outputs/track/CLIP_ID VIDEO_PATH
```
