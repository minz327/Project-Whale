# SLEAP Pose Estimation Environment

Separate environment for running Ren's SLEAP pose estimation models.
Kept isolated from the main YOLO/PyTorch venv to avoid dependency conflicts.

## Setup

```powershell
# Create separate venv
python -m venv .venv_sleap

# Activate it
.\.venv_sleap\Scripts\Activate.ps1

# Install SLEAP (PyPI version, no conda needed)
pip install sleap[pypi]

# Verify
python -c "import sleap; print(sleap.__version__)"
```

## Usage

```powershell
# Activate the SLEAP env (not the main .venv)
.\.venv_sleap\Scripts\Activate.ps1

# Run pose estimation on tracked whales
python scripts/pose_estimate.py outputs/track/20240527-22 videos/20240527-22.MP4
```

## What it does

1. Loads whale tracks from the tracking pipeline (tracks.csv / tracks_relinked.csv)
2. Extracts bounding box crops from the source video
3. Runs Ren's SLEAP centered_instance model to find 7 keypoints per whale:
   - `rostrum_tip` — nose
   - `mid_saddle_patch` — distinctive white marking (used for individual ID)
   - `Caudal_peduncle` — tail base
   - `left/right_pect_fin_tip` — pectoral fins
   - `left/right_caudal_fluke` — tail flukes
4. Saves keypoint coordinates (in full-frame pixel space) to CSV + JSON

## Model

Uses Ren's best model: `exeter/models/full_FGM_v1_250328_113346.centered_instance.n=231/`
- Trained on 231 labeled aerial drone frames of SRKW
- Centered instance approach (crops around saddle patch anchor)
- U-Net backbone, RGB input
