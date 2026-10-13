# Automotive Sealant Bead Inspection & Quality Verification

Industrial Quality Control (QC) System for Automotive Sheet Metal Sealant Bead Inspection.
Combines **Deterministic CAD-Path Tolerance Corridors** (distance transforms) with **Deep Learning Classification (YOLOv8)**.

---

## Architecture Overview

### 1. Deterministic Industrial Tolerance Corridor (`bead_engine.py`)
- **Presence Verification:** Evaluates bead coverage percentage against nominal path ($>85\%$ required for PASS).
- **Placement & Overflow Detection:** Evaluates paste adherence to nominal stamped grooves using a configurable tolerance margin corridor ($\pm 18\text{ px}$). Paste wandering outside is flagged as misplaced.
- **Continuity & Gap Detection:** Identifies gaps along the bead trajectory caused by nozzle clogs, air bubbles, or pressure drops.
- **3-Zone Spatial Inspection:**
  - **Zone 1:** Mounting Bracket Cup (Circular Bead)
  - **Zone 2:** Upper Flange (Curved Rim Bead)
  - **Zone 3:** Center Aperture Perimeter (Closed-loop Contour Bead)

### 2. Deep Learning Classifier (`train_yolo.py`)
- Fine-tuned **Ultralytics YOLOv8** model trained on photorealistic synthetic defect variations:
  - `PASS_CORRECT`: Full bead coverage with lighting, glare, and sensor noise variations.
  - `FAIL_MISSING`: 0% paste (bare metal) or missing critical zone beads.
  - `FAIL_GAP_DEFECT`: Discontinuous bead with localized breaks.
  - `FAIL_MISPLACED`: Shifted / warped beads wandering outside the groove.
- Exports `best.pt` with top-1 accuracy metrics and confusion matrices.

### 3. Interactive Web Dashboard (`app.py`)
- Built with **Streamlit**.
- Includes live test samples, custom image upload, interactive tolerance sliders, visual HUD overlay (Green/Red/Orange), and downloadable inspection JSON reports.

---

## Quickstart

### 1. Launch the Web Dashboard
```bash
streamlit run app.py
```

### 2. Run CLI Batch Inspection
```bash
python test_inspector.py
```

### 3. Retrain the YOLOv8 Classifier
```bash
# 1. Generate synthetic dataset
python generate_dataset.py

# 2. Train YOLOv8 model
python train_yolo.py
```

---

## Color Codes in Visual HUD
- 🟢 **Green Overlay:** Verified bead conforming to nominal tolerance corridor.
- 🔴 **Red Outline:** Missing bead trajectory / detected gap defect.
- 🟠 **Orange Overlay:** Out-of-bounds paste wandering outside tolerance corridor.
- 🔵 **Cyan Outline:** Nominal tolerance zone boundaries.
