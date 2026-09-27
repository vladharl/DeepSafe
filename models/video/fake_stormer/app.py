"""
FakeSTormer Video Deepfake Detection Service

A FastAPI service that wraps the FakeSTormer model for video deepfake detection, following upstream
(https://github.com/10Ring/FakeSTormer @ c5b5f751): RetinaFace-R50 crops of the largest face with 25% padding at
256x256, 15 px margin crop, 224x224, ImageNet normalisation, clips of 4 consecutive face frames shaped
[K, 3, 4, 224, 224], P(fake clip) = sigmoid(out[0]['cls']), averaged over clips.
"""

import os
import sys
import base64
import tempfile
import torch
import logging
import time
import threading
import gc
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, Optional, List

# The upstream repo is cloned to model_code/ next to app.py (/app/model_code in the image)
app_dir = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get(
    "FAKESTORMER_MODEL_CODE_DIR", os.path.join(app_dir, "model_code")
)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FakeSTormer Video Deepfake Detection Service",
    description="Detects deepfakes in videos using FakeSTormer spatio-temporal transformer model.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "fake_stormer_service"
DEVICE = torch.device(os.environ.get("MODEL_DEVICE", "cpu"))

PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "900"))
# C23 (compressed video) checkpoint with its matching config; the weights dir is mounted at /app/weights
MODEL_PATH = os.environ.get(
    "FAKESTORMER_MODEL_PATH",
    os.path.join(
        app_dir,
        "weights",
        "TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth",
    ),
)
CONFIG_PATH = os.environ.get(
    "FAKESTORMER_CONFIG_PATH",
    os.path.join(CODE_DIR, "configs/temporal/FakeSFormer_base_c23.yaml"),
)
FRAMES_PER_VIDEO = int(os.environ.get("FRAMES_PER_VIDEO", "32"))
CLIP_LEN = 4  # the time embedding and temp_loc head are sized for 4 frames
FACE_PADDING, FACE_SIZE, TEST_MARGIN, IMAGE_SIZE = (
    0.25,
    256,
    15,
    224,
)  # images_crop.py; fakesformer_sbi.py test split
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

model_instance: Optional[torch.nn.Module] = None
face_detector = None
model_lock = threading.Lock()
last_used_time: float = 0.0


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data")
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Classification threshold"
    )


def load_model_internal():
    """Build TopDownDetector from the upstream config and load the checkpoint strictly; raises on any failure."""
    global model_instance, face_detector, last_used_time

    with model_lock:
        if model_instance is not None:
            last_used_time = time.time()
            return
        logger.info(
            f"Loading FakeSTormer from {MODEL_PATH} with {CONFIG_PATH} on {DEVICE}..."
        )
        cwd = os.getcwd()
        os.chdir(
            CODE_DIR
        )  # upstream logs/logger.py creates logs/<date>/ relative to the cwd at import
        try:
            from configs.get_config import load_config
            from models import MODELS, build_model
            from retinaface.pre_trained_models import get_model
        finally:
            os.chdir(cwd)
        try:
            cfg = load_config(CONFIG_PATH)
            _model = build_model(cfg.MODEL, MODELS).to(torch.float)
            state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            _model.load_state_dict(state_dict, strict=True)
            model_instance = _model.to(DEVICE).eval()
            if face_detector is None:
                face_detector = get_model(
                    "resnet50_2020-07-20", max_size=2048, device=DEVICE
                )
                face_detector.eval()
            last_used_time = time.time()
            logger.info("FakeSTormer model loaded successfully")
        finally:
            gc.collect()


def ensure_model_loaded():
    global last_used_time
    if model_instance is None:
        load_model_internal()
    last_used_time = time.time()


def unload_model_if_idle():
    """Unload model if idle for too long."""
    global model_instance

    if model_instance is None or PRELOAD_MODEL:
        return

    with model_lock:
        if model_instance is not None and (
            time.time() - last_used_time > MODEL_TIMEOUT
        ):
            logger.info("Unloading FakeSTormer model due to inactivity")
            model_instance = None
            gc.collect()


def largest_face(frame_bgr: np.ndarray):
    """Upstream package_utils/images_crop.py: RetinaFace on RGB, the largest face (upstream's area-and-score rule),
    25% padding per side, BGR crop resized to 256x256. Returns (crop, [x, y, w, h], score) or None.
    """
    height, width = frame_bgr.shape[:2]
    best, area_max, score_max = None, -1, -1
    for face in face_detector.predict_jsons(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)):
        if len(face["bbox"]) != 4:  # predict_jsons reports "no face" as an empty bbox
            continue
        x0, y0, x1, y1 = map(
            int, face["bbox"]
        )  # retinaface-pytorch 0.0.8 returns float boxes
        w, h = x1 - x0, y1 - y0
        if w * h > area_max and face["score"] > score_max:
            area_max, score_max = w * h, face["score"]
            cx0, cx1 = max(0, x0 - int(w * FACE_PADDING)), min(
                width, x1 + int(w * FACE_PADDING)
            )
            cy0, cy1 = max(0, y0 - int(h * FACE_PADDING)), min(
                height, y1 + int(h * FACE_PADDING)
            )
            best = (frame_bgr[cy0:cy1, cx0:cx1], [x0, y0, w, h], float(face["score"]))
    if best is None or not best[0].size:
        return None
    return (
        cv2.resize(best[0], (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR),
    ) + best[1:]


def face_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """Upstream test transform: RGB, 15 px margin crop, affine to 224x224, /255, ImageNet normalisation -> [3,224,224]."""
    from package_utils.transform import get_affine_transform, get_center_scale

    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)[
        TEST_MARGIN:-TEST_MARGIN, TEST_MARGIN:-TEST_MARGIN
    ]
    c, s = get_center_scale(img.shape[:2], 1.0, 200)
    img = cv2.warpAffine(
        img,
        get_affine_transform(c, s, 0, [IMAGE_SIZE, IMAGE_SIZE], pixel_std=200),
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
    )
    return torch.from_numpy(((img / 255 - MEAN) / STD).transpose(2, 0, 1)).float()


def extract_frames(video_path: str, num_frames: int):
    """(frame_index, timestamp_seconds, BGR frame) for num_frames uniformly sampled frames; [] if undecodable."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or total_frames <= 0 or fps <= 0:
            return frames
        for idx in np.unique(
            np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
        ):  # each frame once
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((int(idx), round(idx / fps, 2), frame))
        return frames
    finally:
        cap.release()


def _abstain(
    reason: str,
    frames_sampled: int = 0,
    faces: int = 0,
    frame_details: Optional[list] = None,
):
    return {
        "probability": None,
        "prediction": None,
        "class": "abstain",
        "abstain_reason": reason,
        "total_frames_sampled": frames_sampled,
        "total_faces_analyzed": faces,
        "per_face_scores": [],
        "frame_details": frame_details or [],
    }


def score_video(path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Contract C11: mean P(fake) over clips of 4 consecutive face frames, or an abstention with its reason."""
    try:
        ensure_model_loaded()
    except Exception as e:
        logger.error(f"FakeSTormer unavailable: {e}")
        return _abstain("model_unavailable")

    frames = extract_frames(path, FRAMES_PER_VIDEO)
    if not frames:
        return _abstain("no_frames")

    faces, frame_details = (
        [],
        [],
    )  # faces: (face dict, [3,224,224] tensor) in frame order
    for frame_idx, timestamp, frame in frames:
        detail = {
            "frame_index": frame_idx,
            "timestamp_seconds": timestamp,
            "faces": [],
            "frame_score": None,
        }
        frame_details.append(detail)
        found = largest_face(frame)
        if found:
            face = {
                "face_id": 0,
                "bbox": found[1],
                "score": None,
                "confidence": found[2],
            }
            detail["faces"].append(face)
            faces.append((face, face_tensor(found[0])))
    if not faces:
        return _abstain("no_faces", len(frames), 0, frame_details)
    n_clips = len(faces) // CLIP_LEN
    if not n_clips:
        return _abstain("too_few_faces", len(frames), len(faces), frame_details)

    try:
        faces = faces[: n_clips * CLIP_LEN]
        clips = (
            torch.stack([f[1] for f in faces])
            .view(n_clips, CLIP_LEN, 3, IMAGE_SIZE, IMAGE_SIZE)
            .transpose(1, 2)
        )
        with torch.no_grad():
            clip_scores = (
                torch.sigmoid(model_instance(clips.to(DEVICE))[0]["cls"])
                .flatten()
                .cpu()
                .tolist()
            )
    except Exception as e:
        logger.exception(f"FakeSTormer inference failed: {e}")
        return _abstain("model_unavailable", len(frames), len(faces), frame_details)

    per_face_scores = [clip_scores[i // CLIP_LEN] for i in range(len(faces))]
    for (face, _), score in zip(faces, per_face_scores):
        face["score"] = score
    for detail in frame_details:
        if detail["faces"] and detail["faces"][0]["score"] is not None:
            detail["frame_score"] = detail["faces"][0]["score"]

    probability = float(np.mean(clip_scores))
    prediction = 1 if probability >= threshold else 0
    return {
        "probability": probability,
        "prediction": prediction,
        "class": "fake" if prediction else "real",
        "abstain_reason": None,
        "total_frames_sampled": len(frames),
        "total_faces_analyzed": len(faces),
        "per_face_scores": per_face_scores,
        "frame_details": frame_details,
    }


@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODEL:
        logger.info("Preloading FakeSTormer model at startup")
        try:
            load_model_internal()
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
    else:
        logger.info("Model will be lazy-loaded on first request")

    # Setup idle unload timer
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:

        def periodic_check():
            unload_model_if_idle()
            if model_instance is not None or PRELOAD_MODEL:
                timer = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_check)
                timer.daemon = True
                timer.start()

        timer = threading.Timer(MODEL_TIMEOUT / 2.0, periodic_check)
        timer.daemon = True
        timer.start()


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": model_instance is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/health")
async def health():
    model_exists = os.path.exists(MODEL_PATH)

    status = "healthy"
    message = "Service healthy"

    if not model_exists:
        status = "error_missing_files"
        message = f"Model weights not found at {MODEL_PATH}"
    elif not model_instance and PRELOAD_MODEL:
        status = "error_preload_failed"
        message = "Model was set to preload but is not loaded"
    elif not model_instance:
        status = "degraded_not_loaded"
        message = "Model ready for lazy loading"
    else:
        message = "Model loaded and ready"

    return {
        "status": status,
        "model_name": MODEL_NAME_DISPLAY,
        "weights_found": model_exists,
        "model_loaded": model_instance is not None,
        "model_currently_loaded": model_instance is not None,
        "message": message,
    }


@app.post("/unload")
async def unload_model():
    global model_instance, last_used_time

    if model_instance is None:
        return {"status": "not_loaded", "message": "Model is not currently loaded"}

    with model_lock:
        if model_instance is not None:
            logger.info("Manually unloading model via /unload endpoint")
            model_instance = None
            last_used_time = 0
            gc.collect()
            return {"status": "unloaded", "message": "Model unloaded successfully"}

    return {"status": "error", "message": "Unload failed"}


@app.post("/predict", response_model=Dict[str, Any])
async def predict_video(input_data: VideoInput):
    req_start = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(base64.b64decode(input_data.video_data))
        f.flush()
        result = score_video(f.name, input_data.threshold)
    inference_time = time.time() - req_start
    logger.info(
        f"FakeSTormer prediction completed in {inference_time:.4f}s. "
        f"Prob Fake: {result['probability']}, Class: {result['class']} ({result['abstain_reason']}), "
        f"Frames: {result['total_frames_sampled']}, Faces: {result['total_faces_analyzed']}"
    )
    return {"model": MODEL_NAME_DISPLAY, **result, "inference_time": inference_time}


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7002))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
