"""
server.py
─────────
FastAPI Backend Server for the Multi-Task Perception Cockpit.
Serves live inference for joint semantic segmentation & monocular depth estimation,
metric distance calculations in real-world meters, 3D point cloud generation,
and static frontend HUD assets.
"""

import os
import sys
import io
import time
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.segformer_encoder import SegFormerEncoder
from models.segformer_decoder import SegFormerDecoder
from models.decoder.progressive_depth_decoder import ProgressiveDepthDecoder
from models.mtl_model import MTLModel

# ─────────────────────────────────────────────────────────────────────────────
# Global Configuration & Model Initialization
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 19
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "mtl", "best_mtl_model.pth")
PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")

# Cityscapes 19 Classes
CITYSCAPES_CLASSES = [
    {"id": 0,  "name": "Road",          "category": "Flat",         "color": [128, 64, 128]},
    {"id": 1,  "name": "Sidewalk",      "category": "Flat",         "color": [244, 35, 232]},
    {"id": 2,  "name": "Building",      "category": "Construction", "color": [70, 70, 70]},
    {"id": 3,  "name": "Wall",          "category": "Construction", "color": [102, 102, 156]},
    {"id": 4,  "name": "Fence",         "category": "Construction", "color": [190, 153, 153]},
    {"id": 5,  "name": "Pole",          "category": "Object",       "color": [153, 153, 153]},
    {"id": 6,  "name": "Traffic Light", "category": "Object",       "color": [250, 170, 30]},
    {"id": 7,  "name": "Traffic Sign",  "category": "Object",       "color": [220, 220, 0]},
    {"id": 8,  "name": "Vegetation",    "category": "Nature",       "color": [107, 142, 35]},
    {"id": 9,  "name": "Terrain",       "category": "Nature",       "color": [152, 251, 152]},
    {"id": 10, "name": "Sky",           "category": "Sky",          "color": [70, 130, 180]},
    {"id": 11, "name": "Person",        "category": "Human",        "color": [220, 20, 60]},
    {"id": 12, "name": "Rider",         "category": "Human",        "color": [255, 0, 0]},
    {"id": 13, "name": "Car",           "category": "Vehicle",      "color": [0, 0, 142]},
    {"id": 14, "name": "Truck",         "category": "Vehicle",      "color": [0, 0, 70]},
    {"id": 15, "name": "Bus",           "category": "Vehicle",      "color": [0, 60, 100]},
    {"id": 16, "name": "Train",         "category": "Vehicle",      "color": [0, 80, 100]},
    {"id": 17, "name": "Motorcycle",    "category": "Vehicle",      "color": [0, 0, 230]},
    {"id": 18, "name": "Bicycle",       "category": "Vehicle",      "color": [119, 11, 32]}
]

COLOR_PALETTE = np.array([c["color"] for c in CITYSCAPES_CLASSES], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Global model holder
model: Optional[MTLModel] = None

def load_model():
    global model
    print(f"[INIT] Loading MTL Model onto {DEVICE.upper()}...")
    encoder       = SegFormerEncoder()
    seg_decoder   = SegFormerDecoder(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.1)
    depth_decoder = ProgressiveDepthDecoder(embed_dim=256)
    model = MTLModel(encoder, seg_decoder, depth_decoder, NUM_CLASSES).to(DEVICE)

    if os.path.exists(CKPT_PATH):
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INIT] Loaded best MTL checkpoint from: {CKPT_PATH}")
    else:
        print(f"[WARN] Checkpoint not found at: {CKPT_PATH}. Model initialized with default weights.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INIT] Model ready. Total Parameters: {total_params:.2f}M | Device: {DEVICE}")

# Initialize FastAPI App
app = FastAPI(title="Autonomous Driving Multi-Task Perception Cockpit", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions: Metric Depth, Overlays, and 3D Point Cloud
# ─────────────────────────────────────────────────────────────────────────────
def inverse_depth_to_meters(inv_depth: np.ndarray, d_min: float = 1.0, d_max: float = 80.0) -> np.ndarray:
    """
    Calibrates continuous inverse depth in [0, 1] into physical distance in meters [1.0m, 80.0m].
    Formula: Z = 1.0 / (inv_depth * (1/d_min - 1/d_max) + 1/d_max)
    """
    alpha = (1.0 / d_min) - (1.0 / d_max)
    beta  = 1.0 / d_max
    meters = 1.0 / (inv_depth * alpha + beta + 1e-8)
    return np.clip(meters, d_min, d_max)

def array_to_base64_png(arr: np.ndarray) -> str:
    """Converts RGB numpy image (H, W, 3) uint8 to base64 data URI."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    b64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def generate_point_cloud_3d(rgb_img: np.ndarray, depth_meters: np.ndarray, class_mask: np.ndarray, step: int = 6):
    """
    Back-projects 2D pixels (u, v) with metric depth Z into 3D camera coordinates (X, Y, Z).
    Downsampled by `step` for 60 FPS WebGL browser rendering.
    """
    H, W = depth_meters.shape
    fx = 0.58 * W
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    # Subsample grid
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ys_flat = ys.flatten()
    xs_flat = xs.flatten()

    z = depth_meters[ys_flat, xs_flat]
    valid = z < 75.0  # filter sky / extreme background points for clean mesh

    x_val = xs_flat[valid]
    y_val = ys_flat[valid]
    z_val = z[valid]

    X = (x_val - cx) * z_val / fx
    Y = -(y_val - cy) * z_val / fy  # inverted for Three.js coordinates
    Z = -z_val                       # forward in Three.js is -Z

    r = rgb_img[y_val, x_val, 0] / 255.0
    g = rgb_img[y_val, x_val, 1] / 255.0
    b = rgb_img[y_val, x_val, 2] / 255.0
    c_ids = class_mask[y_val, x_val]

    # Return structured interleaved array
    points = []
    for i in range(len(X)):
        points.extend([
            round(float(X[i]), 3),
            round(float(Y[i]), 3),
            round(float(Z[i]), 3),
            round(float(r[i]), 3),
            round(float(g[i]), 3),
            round(float(b[i]), 3),
            int(c_ids[i]),
            round(float(z_val[i]), 2)  # depth in meters
        ])
    return points

def create_hazard_map(rgb_img: np.ndarray, depth_meters: np.ndarray, class_mask: np.ndarray) -> np.ndarray:
    """
    Creates an automotive hazard map:
    - Obstacles < 5m: Flashing Red Alert
    - Obstacles 5-15m: Yellow Warning
    - Safe Road > 15m: Soft Green
    Dynamic object classes: Person(11), Rider(12), Car(13), Truck(14), Bus(15), Motorcycle(17), Bicycle(18)
    """
    hazard = rgb_img.copy().astype(np.float32)
    dynamic_classes = {11, 12, 13, 14, 15, 17, 18}
    is_dynamic = np.isin(class_mask, list(dynamic_classes))

    danger_mask = is_dynamic & (depth_meters < 5.0)
    warning_mask = is_dynamic & (depth_meters >= 5.0) & (depth_meters < 15.0)

    hazard[danger_mask] = hazard[danger_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
    hazard[warning_mask] = hazard[warning_mask] * 0.4 + np.array([255, 200, 0]) * 0.6
    return np.clip(hazard, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/api/status")
async def get_status():
    global model
    total_params = sum(p.numel() for p in model.parameters()) / 1e6 if model else 34.82
    return {
        "device": DEVICE.upper(),
        "is_cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU",
        "total_parameters_M": round(total_params, 2),
        "backbone": "SegFormer MiT-B2 (Shared)",
        "tasks": ["19-Class Semantic Segmentation", "Monocular Inverse Depth"],
        "checkpoint": os.path.basename(CKPT_PATH)
    }

@app.get("/api/presets")
async def get_presets():
    presets = [
        {
            "id": "scene1_urban",
            "title": "Dense Urban Street",
            "city": "Frankfurt am Main",
            "description": "High density of vehicles, building facades, and roadway infrastructure.",
            "filename": "scene1_urban.png",
            "thumb": "/api/preset/scene1_urban"
        },
        {
            "id": "scene2_pedestrian",
            "title": "Pedestrian Crosswalk",
            "city": "Frankfurt am Main",
            "description": "Pedestrians, cyclists, traffic signals, and sidewalk curbs.",
            "filename": "scene2_pedestrian.png",
            "thumb": "/api/preset/scene2_pedestrian"
        },
        {
            "id": "scene3_highway",
            "title": "Wide Arterial Avenue",
            "city": "Lindau",
            "description": "High-speed multi-lane roadway with distant horizon depth and poles.",
            "filename": "scene3_highway.png",
            "thumb": "/api/preset/scene3_highway"
        },
        {
            "id": "scene4_intersection",
            "title": "Urban Intersection",
            "city": "Munster",
            "description": "Multi-direction intersection with bicycles, cars, and complex fences.",
            "filename": "scene4_intersection.png",
            "thumb": "/api/preset/scene4_intersection"
        }
    ]
    return presets

@app.get("/api/preset/{preset_id}")
async def get_preset_image(preset_id: str):
    file_path = os.path.join(PRESET_DIR, f"{preset_id}.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Preset image not found")
    return FileResponse(file_path, media_type="image/png")

@app.post("/api/predict")
async def predict_multi_task(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
    resolution: str = Form("fast")  # 'fast' = 512x1024, 'native' = 1024x2048
):
    global model
    if model is None:
        load_model()

    # 1. Load image from upload or preset
    if file is not None and hasattr(file, "filename") and file.filename:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    elif preset_id:
        preset_path = os.path.join(PRESET_DIR, f"{preset_id}.png")
        if not os.path.exists(preset_path):
            raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
        pil_img = Image.open(preset_path).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Must provide an uploaded image or a preset_id")

    orig_w, orig_h = pil_img.size

    # Set processing resolution (Fast mode: 512x1024 for instant CPU / HD mode: 1024x2048)
    if resolution == "native" or DEVICE == "cuda":
        target_h, target_w = 1024, 2048
    else:
        target_h, target_w = 512, 1024

    img_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # 2. Normalize and prepare PyTorch tensor
    norm_img = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # 3. Model Inference with latency measurement
    t_start = time.perf_counter()
    with torch.no_grad():
        seg_logits, depth_sigmoid = model(tensor)
    t_infer = (time.perf_counter() - t_start) * 1000.0  # milliseconds

    # 4. Post-process Segmentation
    pred_classes = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    seg_color = COLOR_PALETTE[pred_classes]

    # 5. Post-process Depth
    inv_depth = depth_sigmoid.squeeze().cpu().numpy()
    depth_meters = inverse_depth_to_meters(inv_depth, d_min=1.0, d_max=80.0)

    # Colormap Depth (Turbo colormap: near = red/orange, mid = green/cyan, far = dark blue)
    depth_norm_vis = np.clip(1.0 - (depth_meters - 1.0) / 79.0, 0.0, 1.0)
    depth_colormap = cv2.applyColorMap((depth_norm_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    # 6. Create Blended Overlays & Hazard Heatmap
    rgb_uint8 = (img_np * 255).astype(np.uint8)
    blend_seg = (rgb_uint8 * 0.45 + seg_color * 0.55).astype(np.uint8)
    hazard_map = create_hazard_map(rgb_uint8, depth_meters, pred_classes)

    # 7. Generate 3D Point Cloud for WebGL
    point_cloud_data = generate_point_cloud_3d(rgb_uint8, depth_meters, pred_classes, step=8 if target_w == 1024 else 12)

    # 8. Compute Scene Statistics
    unique_classes, counts = np.unique(pred_classes, return_counts=True)
    total_pixels = pred_classes.size
    class_distribution = []
    for cls_id, cnt in zip(unique_classes, counts):
        meta = CITYSCAPES_CLASSES[cls_id]
        class_distribution.append({
            "id": int(cls_id),
            "name": meta["name"],
            "category": meta["category"],
            "color": f"rgb({meta['color'][0]},{meta['color'][1]},{meta['color'][2]})",
            "percentage": round(float(cnt / total_pixels * 100.0), 2)
        })
    class_distribution.sort(key=lambda x: x["percentage"], reverse=True)

    # Depth statistics in meters
    depth_stats = {
        "min_distance_m": round(float(np.min(depth_meters)), 2),
        "max_distance_m": round(float(np.max(depth_meters)), 2),
        "median_distance_m": round(float(np.median(depth_meters)), 2),
        "road_distance_m": round(float(np.median(depth_meters[pred_classes == 0])) if 0 in unique_classes else 0.0, 2),
        "obstacles_under_10m": int(np.sum((depth_meters < 10.0) & np.isin(pred_classes, [11, 12, 13, 14, 15, 17, 18])))
    }

    # 9. Downsampled Depth & Class Matrix for Client-Side Cursor Hover Probing (128x256 grid)
    probe_h, probe_w = 128, 256
    probe_classes = cv2.resize(pred_classes, (probe_w, probe_h), interpolation=cv2.INTER_NEAREST)
    probe_depth = cv2.resize(depth_meters, (probe_w, probe_h), interpolation=cv2.INTER_LINEAR)

    return {
        "success": True,
        "inference_time_ms": round(t_infer, 1),
        "fps": round(1000.0 / max(t_infer, 1.0), 1),
        "resolution": f"{target_w}x{target_h}",
        "device": DEVICE.upper(),
        "images": {
            "rgb": array_to_base64_png(rgb_uint8),
            "segmentation": array_to_base64_png(seg_color),
            "blend_seg": array_to_base64_png(blend_seg),
            "depth": array_to_base64_png(depth_colormap),
            "hazard": array_to_base64_png(hazard_map)
        },
        "stats": {
            "classes": class_distribution,
            "depth": depth_stats
        },
        "probe": {
            "grid_w": probe_w,
            "grid_h": probe_h,
            "classes": probe_classes.tolist(),
            "depth": np.round(probe_depth, 2).tolist()
        },
        "point_cloud": point_cloud_data,
        "classes_meta": {c["id"]: c["name"] for c in CITYSCAPES_CLASSES}
    }

# Mount static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"  Autonomous Driving Multi-Task Perception Dashboard   ")
    print(f"  Live at: http://localhost:{port}                    ")
    print(f"=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
