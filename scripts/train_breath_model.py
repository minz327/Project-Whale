"""
Train a breath (respiration) detection model from blow-spray features.

Uses brightness signal near the blowhole + temporal context features to
train a Random Forest classifier. More robust than a simple threshold:
  - Learns from temporal patterns (brightness rises/falls across frames)
  - Considers multiple features (brightness, white%, gradients, pose)
  - Can be retrained as more clips are processed

Workflow:
  1. Extract features from respirations_signal.csv + pose data
  2. Generate labels from respirations.csv (heuristic detections)
  3. Train Random Forest with temporal context window
  4. Save model for inference on new clips

Training:
    python scripts/train_breath_model.py outputs/track/20231018-40_trim --save-model models/breath_detector.pkl

Inference on a new clip:
    python scripts/train_breath_model.py outputs/track/NEW_CLIP --predict --model models/breath_detector.pkl
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


WINDOW_SIZE = 5  # frames of context on each side


def load_signal(csv_path: Path) -> list[dict]:
    """Load brightness signal CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame": int(r["frame"]),
                "brightness": float(r["brightness"]),
                "white_pct": float(r["white_pct"]),
            })
    return rows


def load_breath_events(csv_path: Path) -> list[dict]:
    """Load detected breath events for labeling."""
    events = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            events.append({
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "peak_frame": int(r["peak_frame"]),
            })
    return events


def load_pose_features(csv_path: Path) -> dict[int, dict]:
    """Load pose keypoints and compute per-frame features."""
    features = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            rx = r.get("rostrum_tip_x", "")
            ry = r.get("rostrum_tip_y", "")
            sx = r.get("mid_saddle_patch_x", "")
            sy = r.get("mid_saddle_patch_y", "")
            cx = r.get("Caudal_peduncle_x", "")
            cy = r.get("Caudal_peduncle_y", "")

            feat = {}
            if rx and ry and sx and sy:
                feat["rs_dist"] = ((float(rx)-float(sx))**2 +
                                   (float(ry)-float(sy))**2) ** 0.5
            if rx and ry and sx and sy and cx and cy:
                # Body arch angle at saddle patch
                v1 = np.array([float(rx)-float(sx), float(ry)-float(sy)])
                v2 = np.array([float(cx)-float(sx), float(cy)-float(sy)])
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                feat["arch_angle"] = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
            features[fr] = feat
    return features


def build_features(signal: list[dict], pose_features: dict[int, dict],
                   window: int = WINDOW_SIZE) -> tuple[np.ndarray, list[int]]:
    """Build feature matrix with temporal context.

    For each frame, creates features from:
      - Current frame: brightness, white_pct
      - Temporal window: brightness/white values for ±window frames
      - Derived: brightness gradient, rolling mean, rolling std, max in window
      - Pose: arch angle, rostrum-saddle distance
    """
    n = len(signal)
    brights = np.array([s["brightness"] for s in signal])
    whites = np.array([s["white_pct"] for s in signal])
    frames = [s["frame"] for s in signal]

    feature_rows = []

    for i in range(n):
        feat = []

        # Current frame features
        feat.append(brights[i])         # brightness
        feat.append(whites[i])          # white_pct

        # Temporal context window
        w_start = max(0, i - window)
        w_end = min(n, i + window + 1)
        w_bright = brights[w_start:w_end]
        w_white = whites[w_start:w_end]

        # Window statistics
        feat.append(np.mean(w_bright))   # rolling mean brightness
        feat.append(np.std(w_bright))    # rolling std brightness
        feat.append(np.max(w_bright))    # max brightness in window
        feat.append(np.mean(w_white))    # rolling mean white%
        feat.append(np.max(w_white))     # max white% in window

        # Gradients (change from previous frames)
        if i > 0:
            feat.append(brights[i] - brights[i-1])    # brightness delta
            feat.append(whites[i] - whites[i-1])       # white delta
        else:
            feat.append(0.0)
            feat.append(0.0)

        if i > 1:
            feat.append(brights[i] - brights[i-2])    # 2-frame delta
        else:
            feat.append(0.0)

        # Forward gradient (is brightness about to rise/fall?)
        if i < n - 1:
            feat.append(brights[i+1] - brights[i])
        else:
            feat.append(0.0)

        # Brightness relative to baseline
        baseline = np.median(brights)
        feat.append(brights[i] - baseline)             # above baseline
        feat.append(brights[i] / (baseline + 1e-6))    # ratio to baseline

        # Position in brightness peak (am I on a rising or falling edge?)
        if i > 0 and i < n - 1:
            feat.append(float(brights[i] > brights[i-1] and brights[i] > brights[i+1]))  # local peak
            feat.append(float(brights[i] > brights[i-1]))  # rising
        else:
            feat.append(0.0)
            feat.append(0.0)

        # Pose features
        pf = pose_features.get(frames[i], {})
        feat.append(pf.get("arch_angle", 170.0))
        feat.append(pf.get("rs_dist", 180.0))

        feature_rows.append(feat)

    feature_names = [
        "brightness", "white_pct",
        "roll_mean_bright", "roll_std_bright", "roll_max_bright",
        "roll_mean_white", "roll_max_white",
        "bright_delta_1", "white_delta_1", "bright_delta_2",
        "bright_fwd_delta",
        "above_baseline", "ratio_baseline",
        "is_local_peak", "is_rising",
        "arch_angle", "rs_dist",
    ]

    return np.array(feature_rows), frames, feature_names


def generate_labels(frames: list[int], events: list[dict],
                    expand: int = 3) -> np.ndarray:
    """Generate binary labels: 1 = breath frame, 0 = normal.

    Expands breath events by ±expand frames to capture the full blow event.
    """
    labels = np.zeros(len(frames), dtype=int)
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    for event in events:
        start = event["start_frame"]
        end = event["end_frame"]
        # Expand slightly to catch onset/offset
        for f in frames:
            if start - expand * 3 <= f <= end + expand * 3:
                if f in frame_to_idx:
                    labels[frame_to_idx[f]] = 1

    return labels


def train_model(X: np.ndarray, y: np.ndarray,
                feature_names: list[str]) -> tuple:
    """Train a breath detection model."""
    # Handle class imbalance with class weights
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"  Training samples: {len(y)} ({n_pos} breath, {n_neg} normal)")
    print(f"  Positive ratio: {n_pos/len(y)*100:.1f}%")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest with class weight balancing
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    # Cross-validation (stratified)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="f1")
    print(f"  5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training performance
    y_pred = rf.predict(X_scaled)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, pos_label=1,
                                                   average="binary",
                                                   zero_division=0)
    print(f"  Training — Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    # Feature importance
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Top features:")
    for i in range(min(8, len(feature_names))):
        idx = sorted_idx[i]
        print(f"    {feature_names[idx]}: {importances[idx]:.3f}")

    return rf, scaler


def predict_breaths(rf, scaler, X: np.ndarray, frames: list[int],
                    fps: float, min_gap_s: float = 2.0) -> list[dict]:
    """Run model inference and extract breath events."""
    X_scaled = scaler.transform(X)
    probs = rf.predict_proba(X_scaled)[:, 1]

    # Find runs of high-probability frames
    threshold = 0.5
    min_gap_frames = int(min_gap_s * fps)

    # Group consecutive positive predictions into events
    events = []
    in_event = False
    event_start = 0
    event_frames = []

    for i in range(len(frames)):
        if probs[i] >= threshold:
            if not in_event:
                event_start = i
                event_frames = []
                in_event = True
            event_frames.append((i, frames[i], probs[i]))
        else:
            if in_event:
                events.append(event_frames)
                in_event = False
                event_frames = []
    if in_event:
        events.append(event_frames)

    # Merge events that are close together
    merged = [events[0]] if events else []
    for ev in events[1:]:
        prev_end_frame = merged[-1][-1][1]
        curr_start_frame = ev[0][1]
        if curr_start_frame - prev_end_frame < min_gap_frames:
            merged[-1].extend(ev)
        else:
            merged.append(ev)

    # Extract breath event info
    breath_events = []
    for ev in merged:
        peak_idx = max(range(len(ev)), key=lambda i: ev[i][2])
        breath_events.append({
            "breath_id": len(breath_events) + 1,
            "peak_frame": ev[peak_idx][1],
            "peak_prob": float(ev[peak_idx][2]),
            "start_frame": ev[0][1],
            "end_frame": ev[-1][1],
            "duration_frames": ev[-1][1] - ev[0][1],
            "time_s": round(ev[peak_idx][1] / fps, 2),
        })

    return breath_events


def main():
    parser = argparse.ArgumentParser(description="Train/run breath detection model")
    parser.add_argument("track_dir", type=Path,
                        help="Track directory with respirations_signal.csv")
    parser.add_argument("--save-model", type=Path, default=None,
                        help="Save trained model to this path")
    parser.add_argument("--model", type=Path, default=None,
                        help="Load model for inference (prediction mode)")
    parser.add_argument("--predict", action="store_true",
                        help="Run in prediction mode (requires --model)")
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="Min seconds between breaths (default: 2.0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS (default: 30.0)")
    args = parser.parse_args()

    signal_csv = args.track_dir / "respirations_signal.csv"
    if not signal_csv.exists():
        print(f"Error: {signal_csv} not found")
        print("Run detect_respirations.py first to generate the brightness signal.")
        sys.exit(1)

    pose_csv = args.track_dir / "pose" / "pose_keypoints.csv"

    # Load data
    print(f"Loading signal from {signal_csv}...")
    signal = load_signal(signal_csv)
    print(f"  {len(signal)} frames")

    # Load pose features
    pose_features = {}
    if pose_csv.exists():
        pose_features = load_pose_features(pose_csv)
        print(f"  Loaded pose features for {len(pose_features)} frames")

    # Build feature matrix
    print(f"Building features (window={WINDOW_SIZE})...")
    X, frames, feature_names = build_features(signal, pose_features)
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    if args.predict and args.model:
        # ---- INFERENCE MODE ----
        print(f"\nLoading model from {args.model}...")
        with open(args.model, "rb") as f:
            saved = pickle.load(f)
        rf = saved["model"]
        scaler = saved["scaler"]

        print("Running breath prediction...")
        events = predict_breaths(rf, scaler, X, frames, args.fps, args.min_gap)

        # Save predictions
        out_csv = args.track_dir / "respirations_model.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "breath_id", "peak_frame", "time_s", "peak_prob",
                "start_frame", "end_frame", "duration_frames"])
            w.writeheader()
            w.writerows(events)

        total_time = (frames[-1] - frames[0]) / args.fps
        rate = len(events) / (total_time / 60) if total_time > 0 else 0

        print(f"\n{'='*55}")
        print(f"MODEL PREDICTION RESULTS")
        print(f"{'='*55}")
        print(f"  Breaths detected: {len(events)}")
        print(f"  Rate: {rate:.1f} breaths/min")
        print(f"  Saved to {out_csv}")
        for e in events:
            print(f"  Breath {e['breath_id']}: frame {e['peak_frame']} "
                  f"({e['time_s']:.1f}s), prob={e['peak_prob']:.2f}")

    else:
        # ---- TRAINING MODE ----
        events_csv = args.track_dir / "respirations.csv"
        if not events_csv.exists():
            print(f"Error: {events_csv} not found")
            print("Run detect_respirations.py first to generate breath labels.")
            sys.exit(1)

        events = load_breath_events(events_csv)
        print(f"  Loaded {len(events)} breath events as labels")

        # Generate labels
        y = generate_labels(frames, events)
        print(f"  Labels: {np.sum(y==1)} breath frames, {np.sum(y==0)} normal frames")

        # Train model
        print(f"\nTraining Random Forest...")
        rf, scaler = train_model(X, y, feature_names)

        # Run inference on training data to verify
        print(f"\nVerifying on training data...")
        detected = predict_breaths(rf, scaler, X, frames, args.fps, args.min_gap)
        total_time = (frames[-1] - frames[0]) / args.fps
        rate = len(detected) / (total_time / 60) if total_time > 0 else 0

        print(f"  Breaths detected: {len(detected)} (vs {len(events)} labeled)")
        print(f"  Rate: {rate:.1f} breaths/min")
        for e in detected:
            print(f"    Breath {e['breath_id']}: frame {e['peak_frame']} "
                  f"({e['time_s']:.1f}s), prob={e['peak_prob']:.2f}")

        # Save model
        save_path = args.save_model or Path("models/breath_detector.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "model": rf,
                "scaler": scaler,
                "feature_names": feature_names,
                "window_size": WINDOW_SIZE,
                "training_clips": [str(args.track_dir)],
                "n_breath_events": len(events),
                "n_training_frames": len(frames),
            }, f)
        print(f"\n  Model saved to {save_path}")

        # Save training summary
        summary = {
            "model_path": str(save_path),
            "training_clip": str(args.track_dir),
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "n_training_frames": len(frames),
            "n_breath_events": len(events),
            "n_breath_frames": int(np.sum(y == 1)),
            "detected_breaths": len(detected),
            "detected_rate_bpm": round(rate, 1),
            "cv_f1": "see console output",
        }
        summary_path = args.track_dir / "breath_model_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*55}")
        print(f"BREATH MODEL TRAINING COMPLETE")
        print(f"{'='*55}")
        print(f"  Model: {save_path}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Training data: {len(frames)} frames, {len(events)} breath events")
        print(f"  Use --predict --model {save_path} to run on new clips")


if __name__ == "__main__":
    main()
