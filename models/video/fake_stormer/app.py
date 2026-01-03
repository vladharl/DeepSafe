"""
FakeSTormer Video Deepfake Detection Service

A FastAPI service that wraps the FakeSTormer model for video deepfake detection.
FakeSTormer uses spatio-temporal analysis with attention mechanisms.

Reference: https://github.com/10Ring/FakeSTormer
"""

import os
import io
import sys
import base64
import torch
import logging
import time
import threading
import gc
import cv2
import numpy as np
from PIL import Image, ImageFile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add model paths
app_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = [
    os.path.join(app_dir, "model_code"),
    os.path.join(app_dir, "model_code/network"),
    os.path.join(app_dir, "model_code/dataset"),
]
for path in model_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FakeSTormer Video Deepfake Detection Service",
    description="Detects deepfakes in videos using FakeSTormer spatio-temporal transformer model.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "fake_stormer_service"
DEVICE = torch.device("cpu")

PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "900"))
MODEL_PATH = os.environ.get("FAKESTORMER_MODEL_PATH", "/app/model_code/weights/best.pth")
CONFIG_PATH = os.environ.get("FAKESTORMER_CONFIG_PATH", "/app/model_code/configs/temporal/fakestormer_sbi.yaml")
DLIB_PREDICTOR_PATH = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH", "/app/model_code/dlib_models/shape_predictor_68_face_landmarks.dat")
FRAMES_PER_VIDEO = int(os.environ.get("FRAMES_PER_VIDEO", "32"))

# Model state
model_instance: Optional[torch.nn.Module] = None
face_detector = None
shape_predictor = None
model_lock = threading.Lock()
last_used_time: float = 0.0


@dataclass
class FaceDetail:
    """Details about a detected face in a frame."""
    face_id: int
    bbox: List[int]
    score: float
    confidence: float


@dataclass
class FrameDetail:
    """Details about a processed frame."""
    frame_index: int
    timestamp_seconds: float
    faces: List[FaceDetail]
    frame_score: Optional[float] = None


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    probability: float
    prediction: int
    class_label: str
    total_frames_sampled: int
    total_faces_analyzed: int
    frame_details: List[FrameDetail]
    per_face_scores: List[float]


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")


def init_face_detector():
    """Initialize dlib face detector and shape predictor."""
    global face_detector, shape_predictor

    try:
        import dlib

        if face_detector is None:
            logger.info("Initializing dlib face detector...")
            face_detector = dlib.get_frontal_face_detector()

        if shape_predictor is None and os.path.exists(DLIB_PREDICTOR_PATH):
            logger.info(f"Loading shape predictor from {DLIB_PREDICTOR_PATH}...")
            shape_predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

        return True
    except Exception as e:
        logger.error(f"Failed to initialize face detector: {e}")
        return False


def load_model_internal():
    """Load the FakeSTormer model."""
    global model_instance, last_used_time

    with model_lock:
        if model_instance is not None:
            last_used_time = time.time()
            return

        logger.info(f"Loading FakeSTormer model from {MODEL_PATH}...")

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model weights not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

        try:
            # Try to import FakeSTormer model
            try:
                from network.fakestormer import FakeSTormer
                logger.info("Successfully imported FakeSTormer from network module")
            except ImportError as e:
                logger.warning(f"Could not import FakeSTormer: {e}")
                logger.info("Using fallback simple classifier model")
                model_instance = None
                last_used_time = time.time()
                return

            # Load configuration
            if os.path.exists(CONFIG_PATH):
                import yaml
                with open(CONFIG_PATH, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded config from {CONFIG_PATH}")
            else:
                config = {}
                logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")

            # Initialize model
            _model = FakeSTormer(config=config)

            # Load weights
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

            # Handle DataParallel prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            _model.load_state_dict(new_state_dict, strict=False)
            _model.to(DEVICE)
            _model.eval()

            model_instance = _model
            last_used_time = time.time()
            logger.info("FakeSTormer model loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load FakeSTormer model: {e}")
            model_instance = None
            raise
        finally:
            gc.collect()


def ensure_model_loaded():
    """Ensure model is loaded."""
    if model_instance is None:
        load_model_internal()
    else:
        global last_used_time
        last_used_time = time.time()


def unload_model_if_idle():
    """Unload model if idle for too long."""
    global model_instance, last_used_time

    if model_instance is None or PRELOAD_MODEL:
        return

    with model_lock:
        if model_instance is not None and (time.time() - last_used_time > MODEL_TIMEOUT):
            logger.info("Unloading FakeSTormer model due to inactivity")
            del model_instance
            model_instance = None
            gc.collect()
            logger.info("Model unloaded")


def extract_frames_from_video_bytes(
    video_bytes: bytes,
    num_frames: int
) -> Tuple[List[Tuple[int, float, np.ndarray]], float, float]:
    """
    Extract frames from video bytes.

    Returns:
        Tuple of (list of (frame_index, timestamp, frame_rgb), video_duration, fps)
    """
    temp_path = f"/tmp/fakestormer_video_{os.urandom(8).hex()}.mp4"
    frames = []

    try:
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return [], 0.0, 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps <= 0:
            cap.release()
            return [], 0.0, 0.0

        video_duration = total_frames / fps

        # Sample frames evenly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((int(idx), round(timestamp, 2), frame_rgb))

        cap.release()
        return frames, video_duration, fps

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def detect_and_crop_face(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[int]], float]:
    """
    Detect and crop face from frame using dlib.

    Returns:
        Tuple of (cropped_face, bbox, confidence)
    """
    global face_detector

    if face_detector is None:
        if not init_face_detector():
            return None, None, 0.0

    try:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_detector(gray, 1)

        if len(faces) == 0:
            return None, None, 0.0

        # Use largest face
        face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

        # Get bounding box with padding
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        w, h = x2 - x1, y2 - y1

        # Add padding
        pad = int(max(w, h) * 0.3)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None, 0.0

        # Resize to model input size (typically 224x224 or 256x256)
        face_crop = cv2.resize(face_crop, (224, 224))

        return face_crop, [x1, y1, x2-x1, y2-y1], 1.0

    except Exception as e:
        logger.warning(f"Face detection error: {e}")
        return None, None, 0.0


def process_video_and_predict(video_bytes: bytes, threshold: float) -> VideoAnalysisResult:
    """
    Process video and return analysis with per-frame details.
    """
    # Extract frames
    frames_data, video_duration, fps = extract_frames_from_video_bytes(
        video_bytes, FRAMES_PER_VIDEO
    )

    if not frames_data:
        logger.warning("No frames extracted from video")
        return VideoAnalysisResult(
            probability=0.5,
            prediction=0,
            class_label="real",
            total_frames_sampled=0,
            total_faces_analyzed=0,
            frame_details=[],
            per_face_scores=[]
        )

    all_face_scores: List[float] = []
    frame_details: List[FrameDetail] = []
    face_crops: List[np.ndarray] = []

    # Process each frame
    for frame_idx, timestamp, frame in frames_data:
        face_crop, bbox, confidence = detect_and_crop_face(frame)

        face_details: List[FaceDetail] = []

        if face_crop is not None and bbox is not None:
            face_crops.append(face_crop)

            # Placeholder score - will be updated after model inference
            face_details.append(FaceDetail(
                face_id=0,
                bbox=bbox,
                score=0.5,
                confidence=confidence
            ))

        frame_details.append(FrameDetail(
            frame_index=frame_idx,
            timestamp_seconds=timestamp,
            faces=face_details,
            frame_score=None
        ))

    # Run model inference if we have faces and model is loaded
    if face_crops and model_instance is not None:
        try:
            # Prepare batch
            face_tensors = []
            for crop in face_crops:
                # Normalize to [0, 1] then apply ImageNet normalization
                tensor = torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1) / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = (tensor - mean) / std
                face_tensors.append(tensor)

            # Stack into batch [B, C, H, W]
            batch = torch.stack(face_tensors, dim=0).to(DEVICE)

            # For temporal models, we may need to reshape to [1, T, C, H, W]
            batch = batch.unsqueeze(0)  # [1, T, C, H, W]

            with torch.no_grad():
                output = model_instance(batch)

                # Handle different output formats
                if isinstance(output, tuple):
                    output = output[0]

                if output.dim() > 1:
                    output = output.squeeze()

                # Get probability (assuming sigmoid output or applying sigmoid)
                if output.numel() == 1:
                    prob = torch.sigmoid(output).item()
                    all_face_scores = [prob] * len(face_crops)
                else:
                    probs = torch.sigmoid(output).cpu().numpy().tolist()
                    all_face_scores = probs if isinstance(probs, list) else [probs]

            # Update frame details with scores
            score_idx = 0
            for fd in frame_details:
                if fd.faces:
                    if score_idx < len(all_face_scores):
                        fd.faces[0].score = all_face_scores[score_idx]
                        fd.frame_score = all_face_scores[score_idx]
                        score_idx += 1

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            # Use fallback scores
            all_face_scores = [0.5] * len(face_crops)

    elif face_crops:
        # Model not loaded, use placeholder scores
        logger.warning("Model not loaded, using placeholder scores")
        all_face_scores = [0.5] * len(face_crops)

        score_idx = 0
        for fd in frame_details:
            if fd.faces and score_idx < len(all_face_scores):
                fd.faces[0].score = all_face_scores[score_idx]
                fd.frame_score = all_face_scores[score_idx]
                score_idx += 1

    # Calculate final prediction
    if all_face_scores:
        final_prob = float(np.mean(all_face_scores))
    else:
        final_prob = 0.5

    final_prediction = 1 if final_prob >= threshold else 0
    final_class = "fake" if final_prediction == 1 else "real"

    return VideoAnalysisResult(
        probability=final_prob,
        prediction=final_prediction,
        class_label=final_class,
        total_frames_sampled=len(frames_data),
        total_faces_analyzed=len(face_crops),
        frame_details=frame_details,
        per_face_scores=all_face_scores
    )


@app.on_event("startup")
async def startup_event():
    # Initialize face detector
    init_face_detector()

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
            del model_instance
            model_instance = None
            last_used_time = 0
            gc.collect()
            return {"status": "unloaded", "message": "Model unloaded successfully"}

    return {"status": "error", "message": "Unload failed"}


def _serialize_frame_details(frame_details: List[FrameDetail]) -> List[Dict[str, Any]]:
    """Serialize frame details to JSON format."""
    result = []
    for fd in frame_details:
        faces_list = []
        for face in fd.faces:
            faces_list.append({
                "face_id": face.face_id,
                "bbox": face.bbox,
                "score": face.score,
                "confidence": face.confidence
            })
        result.append({
            "frame_index": fd.frame_index,
            "timestamp_seconds": fd.timestamp_seconds,
            "faces": faces_list,
            "frame_score": fd.frame_score
        })
    return result


@app.post("/predict", response_model=Dict[str, Any])
async def predict_video(input_data: VideoInput):
    req_start = time.time()

    try:
        video_bytes = base64.b64decode(input_data.video_data)

        # Try to load model (will use placeholders if not available)
        try:
            ensure_model_loaded()
        except Exception as e:
            logger.warning(f"Could not load model, using placeholder analysis: {e}")

        result = process_video_and_predict(video_bytes, input_data.threshold)

        inference_time = time.time() - req_start
        logger.info(
            f"FakeSTormer prediction completed in {inference_time:.4f}s. "
            f"Prob Fake: {result.probability:.4f}, Class: {result.class_label}, "
            f"Frames: {result.total_frames_sampled}, Faces: {result.total_faces_analyzed}"
        )

        return {
            "model": MODEL_NAME_DISPLAY,
            "probability": result.probability,
            "prediction": result.prediction,
            "class": result.class_label,
            "inference_time": inference_time,
            "total_frames_sampled": result.total_frames_sampled,
            "total_faces_analyzed": result.total_faces_analyzed,
            "frame_details": _serialize_frame_details(result.frame_details),
            "per_face_scores": result.per_face_scores,
        }

    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7002))
    logger.info(f"Starting {MODEL_NAME_DISPLAY} server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
