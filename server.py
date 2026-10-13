"""
Industrial Automotive Vision Server (FastAPI).
Serves the Cognex/Keyence-style Industrial Machine Vision HMI dashboard
and provides sub-20ms inspection endpoints.
"""

import os
import time
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from bead_engine import BeadInspector

app = FastAPI(title="Industrial Automotive Vision System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, 'test_samples')

# Initialize inspection engine
inspector = BeadInspector(base_dir=BASE_DIR)

# Load YOLO model if available
yolo_model = None
yolo_candidates = [
    os.path.join(BASE_DIR, 'best.pt'),
    os.path.join(BASE_DIR, 'runs', 'car_bead_classifier', 'weights', 'best.pt')
]
for p in yolo_candidates:
    if os.path.exists(p):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(p)
            print(f"[AI] Loaded trained model from {p}")
            break
        except Exception as e:
            print(f"[AI] Warning loading YOLO: {e}")


def b64_encode_img(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')


@app.get("/api/samples")
def get_samples():
    """Return pre-configured industrial test parts."""
    if not os.path.exists(TEST_SAMPLES_DIR):
        return []
    samples = []
    for f in sorted(os.listdir(TEST_SAMPLES_DIR)):
        if f.endswith('.jpg') and not f.startswith('result_'):
            # Prettify name
            clean_name = f.replace('.jpg', '').replace('_', ' ')
            samples.append({
                "filename": f,
                "label": clean_name
            })
    return samples


@app.post("/api/inspect")
async def inspect_endpoint(
    sample_name: str = Form(None),
    tolerance_px: int = Form(18),
    min_coverage: float = Form(0.85),
    max_misplaced: float = Form(0.15),
    file: UploadFile = File(None)
):
    t_start = time.perf_counter()

    # Dynamic tolerance config
    inspector.tolerance_px = tolerance_px
    inspector.min_coverage = min_coverage
    inspector.max_misplaced_ratio = max_misplaced
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_px * 2 + 1, tolerance_px * 2 + 1))
    inspector.corridor_all = cv2.dilate(inspector.golden_all, kernel)
    inspector.corridor_cup = cv2.dilate(inspector.golden_cup, kernel)
    inspector.corridor_flange = cv2.dilate(inspector.golden_flange, kernel)
    inspector.corridor_cutout = cv2.dilate(inspector.golden_cutout, kernel)

    img_bgr = None
    source_label = "Custom Upload"

    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        source_label = file.filename
    elif sample_name:
        path = os.path.join(TEST_SAMPLES_DIR, sample_name)
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            source_label = sample_name

    if img_bgr is None:
        golden_path = os.path.join(BASE_DIR, 'WhatsApp Image 2026-10-13 at 19.11.31.jpeg')
        img_bgr = cv2.imread(golden_path)
        source_label = "01_PASS_Golden_Nominal.jpg"

    # Run deterministic inspection
    res, annotated_bgr = inspector.inspect(img_bgr)

    # Run AI inference if available
    yolo_result = None
    if yolo_model is not None:
        try:
            preds = yolo_model.predict(img_bgr, verbose=False)[0]
            top1_idx = int(preds.probs.top1)
            probs = preds.probs.data.cpu().numpy()
            yolo_result = {
                "label": preds.names[top1_idx],
                "confidence": round(float(probs[top1_idx]) * 100, 1),
                "distribution": {preds.names[i]: round(float(probs[i]) * 100, 1) for i in range(len(preds.names))}
            }
        except Exception as e:
            yolo_result = {"error": str(e)}

    cycle_time_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return {
        "status": "success",
        "source": source_label,
        "cycle_time_ms": cycle_time_ms,
        "results": res,
        "ai": yolo_result,
        "raw_img": b64_encode_img(img_bgr),
        "annotated_img": b64_encode_img(annotated_bgr)
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = os.path.join(BASE_DIR, 'templates', 'index.html')
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Industrial Vision System loading...</h1>"


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
