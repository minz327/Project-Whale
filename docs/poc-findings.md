# POC Findings Report

## Overview

End-to-end prototype testing whale detection, tracking, and behaviour measurement from drone footage. The outcome is learning — not a production model.

## Pipeline Built

```
Video → Frame extraction → YOLO-World detection → Cross-class NMS → Norfair (IoU + ReID + GMC) → Noise filter → Relative metrics
```

### Scripts

| Script | Purpose |
|--------|---------|
| `detect_whales.py` | Baseline COCO YOLOv8n detection (failed) |
| `detect_yoloworld.py` | Standalone open-vocabulary detection (legacy, replaced by track_whales.py) |
| `evaluate_detections.py` | Cross-class NMS + HTML review grid |
| `track_whales.py` | **Primary pipeline**: YOLO-World detection + Norfair tracking (ReID + GMC) |
| `relink_tracks.py` | Post-hoc track merging (largely redundant with Norfair ReID) |
| `compensate_tracks.py` | Ego-motion compensation via optical flow |
| `pose_estimate.py` | SLEAP 7-keypoint pose estimation (needs `.venv_sleap`) |
| `detect_respirations.py` | Breath detection from blowhole brightness |
| `track_metrics.py` | Body-length calibrated metrics |
| `visualize_tracks.py` | Trajectory map + timeline plots |
| `render_tracked_video.py` | Full annotated video (box + skeleton + minimap + HUD) |
| `download_clips.py` | Batch clip downloader |

### Clips Tested

| Clip | Type | Resolution | Duration |
|------|------|-----------|----------|
| `srkw_calf_drone.mp4` | Boat-level, L90 + calf L128 | 1080p | ~2.5 min |
| `20240727-84.MP4` | Aerial drone | 4K | ~35s |
| `20240527-22.MP4` | Aerial drone | 4K | 32s |

---

## Research Question Answers

### 1. Is whale detection feasible? **Yes, with caveats.**

- YOLO-World (open-vocabulary, zero fine-tuning) detects whales with **80%+ accuracy** at high confidence
- Works on both boat-level and aerial drone footage
- Confidence is moderate (mean ~0.57 on aerial clips) — fine-tuning on whale data would improve edge cases
- Standard COCO models (YOLOv8n) completely fail — **0 whale detections**
- Cross-class NMS is essential: ~40% of raw detections are duplicate labels ("whale" + "marine mammal" + "dorsal fin") on the same animal

### 2. Is within-video tracking feasible? **Yes.**

- **Norfair tracker** with two-tier matching solves the ID fragmentation problem:
  - Short-term: IoU distance matching (threshold 0.7, hit_counter_max 15)
  - Long-term ReID: ResNet18 cosine similarity (threshold 0.5, ~15s buffer)
  - Camera motion compensation: Norfair `MotionEstimator` with homography
- Single-whale clip: **13 IDs → 2** (1 main whale + 1 concurrent detection)
- Multi-whale clip (3 whales): **6 IDs → 3** stable tracks spanning full 70s video
- ReID bridges dive gaps of 15+ seconds
- Post-processing noise filter (`--min-track-conf 0.25`) removes false positive tracks
- **Previous approach (ByteTrack)** was IoU-only, no appearance features — superseded
- `relink_tracks.py` post-hoc merging is now largely redundant (Norfair finds 0 candidates to merge)

### 3. What annotation work is needed? **Minimal for detection, moderate for tracking.**

- Detection works out of the box — no labeled whale data was needed
- Track re-linking needs human judgment to validate merges
- Ground truth bounding boxes would enable proper precision/recall metrics

### 4. What behaviours are measurable? **Surfacing patterns and proximity.**

| Metric | Reliability | Notes |
|--------|------------|-------|
| Surfacing bouts & dive gaps | **High** | Directly measurable, camera-invariant |
| Inter-whale distance | **High** | Per-frame measurement, no stabilization needed |
| Apparent size (bbox area) | **Medium** | Proxy for depth/altitude changes |
| Frame-to-frame speed | **Low** | Contaminated by camera motion |
| Absolute trajectories | **Unusable** | Requires drone telemetry (SRT files) |

### 5. What are the real blockers?

1. **Camera motion** — No drone telemetry means no reliable absolute positions. Ego-motion compensation via optical flow removes drift for relative analysis, but geo-referencing requires DJI SRT files.
2. **Respiration detection** — Current brightness-based approach needs tuning per clip (only found 1 breath on multi-whale clip).
3. **4K processing speed** — Very slow on CPU; needs GPU or frame downscaling.
4. **Video encoding** — OpenCV's `mp4v` codec produces unplayable files. Must use XVID + ffmpeg H.264 re-encode.

---

## Ren's Prior Work (Exeter / DORSAP)

Ren (Lawrence Cutler, Univ. of Exeter) built **DORSAP** — Dorsal Saddle Patch Automated Recognition & Positioning — for orca pose estimation from aerial drone footage.

### What was done

- Trained **SLEAP pose estimation models** on 231 labeled frames of SRKW from aerial drone video (2700×5120)
- 7-keypoint skeleton: rostrum tip, saddle patch, caudal peduncle, left/right flukes, left/right pectoral fins, left/right eye patches
- Two models (centroid + centered instance), both with excellent convergence (val loss ~3.5e-05)
- Ant tracking as a prototype before applying to whales

### Relationship to this POC

| Pipeline step | This POC | Ren's work |
|---------------|----------|------------|
| Detection | YOLO-World (bounding boxes) | SLEAP centroid model |
| Tracking | ByteTrack + appearance ReID | Not implemented |
| Body detail | Not done | 7-keypoint pose estimation |
| ID features | Generic ResNet crops | Saddle patch + eye patch keypoints |

### What Ren's work can help with

- **Individual whale ID** — Saddle patch shape is unique per orca; Ren's skeleton locates it
- **Behaviour classification** — Keypoints enable body bend angle, fluke position, pectoral fin spread
- **Training recipes** — U-Net with 231 frames proven to converge on aerial whale footage

### Limitations

- Models expect **top-down aerial footage only** — this is fine since Darren's drone data is aerial
- Only 231 labeled frames — good for POC, not production
- Python pipeline files are inaccessible (OneDrive sync errors)
- SLEAP `.h5` checkpoints may need version compatibility work

---

## Recommendations for Next Phase

1. **Ask Darren for DJI SRT files** — unlocks absolute trajectories and solves camera motion entirely
2. **Run remaining clips** through Norfair pipeline — `20240527-22`, `srkw_calf_drone`, `20231018-48_trim`, `20240127-7_trim`
3. **Tune respiration detection** — brightness-based approach needs per-clip or adaptive thresholding
4. **Simplify `relink_tracks.py`** — largely redundant now that Norfair handles ReID during tracking
5. **Fine-tune YOLO on whale bounding boxes** — small annotation effort (~100 frames), big confidence improvement
6. **Focus analysis on surfacing patterns + inter-whale distance** — these work reliably today
7. **Combine pipelines**: YOLO detection → Norfair → SLEAP keypoints (two-stage architecture is proven)
