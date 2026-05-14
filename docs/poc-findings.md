# POC Findings Report

## Overview

End-to-end prototype testing whale detection, tracking, and behaviour measurement from drone footage. The outcome is learning — not a production model.

## Pipeline Built

```
Video → Frame extraction → YOLO-World detection → Cross-class NMS → ByteTrack → Track re-linking (appearance) → Relative metrics
```

### Scripts

| Script | Purpose |
|--------|---------|
| `detect_whales.py` | Baseline COCO YOLOv8n detection |
| `detect_yoloworld.py` | Open-vocabulary whale detection (YOLO-World) |
| `evaluate_detections.py` | Cross-class NMS + HTML review grid |
| `track_whales.py` | ByteTrack multi-object tracking + annotated video |
| `relink_tracks.py` | Merge fragmented tracks (spatial + appearance ReID) |
| `visualize_tracks.py` | Trajectory map + timeline plots |
| `track_metrics.py` | Camera-invariant relative metrics |
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

### 2. Is within-video tracking feasible? **Partially.**

- ByteTrack maintains whale IDs during continuous surfacing (up to **12 seconds** tested)
- IDs break across dives — a whale that submerges gets a new ID when it resurfaces
- Appearance-based re-linking (ResNet18 embeddings) successfully merges fragments (0.86–0.92 cosine similarity), reducing **13 tracks to 9** in the test clip
- **Unsolved**: Long dives + drone movement cause large spatial gaps that can't be reliably re-linked

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

1. **Camera motion** — No drone telemetry means no reliable absolute positions. Feature-based stabilization fails on open ocean (insufficient stable features). This is the #1 blocker.
2. **ID fragmentation across dives** — Solvable with better appearance models but fundamentally hard.
3. **Low detection confidence** — Fine-tuning on whale data would help.
4. **4K processing speed** — Very slow on CPU; needs GPU or frame downscaling.

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
2. **Fine-tune a YOLO model on whale bounding boxes** — small annotation effort (~100 frames), big confidence improvement
3. **Focus analysis on surfacing patterns + inter-whale distance** — these work today
4. **Combine pipelines**: YOLO detection → ByteTrack → SLEAP keypoints (two-stage architecture)
5. **Run on more clips** — test `noaa_uav_overview.mp4` + get additional clips from Darren covering different scenarios (glare, multiple whales, social interaction)
