import os
import io
import sys
import base64
import torch
import torchvision.transforms as T  # Added for normalization
import logging
import time
import threading
import gc
import cv2
import numpy as np
import yaml
from PIL import Image, ImageFile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any, Optional, List, Tuple, OrderedDict  # Added OrderedDict
from dataclasses import dataclass, asdict

# Ensure paths are set for model imports
module_paths_to_add = [
    "model_code/cross-efficient-vit",
    "model_code/efficient-vit",
    "model_code/preprocessing",
    "model_code/cross-efficient-vit/efficient_net",
]
# Construct absolute paths based on the location of app.py
app_dir = os.path.dirname(os.path.abspath(__file__))
for path_suffix in module_paths_to_add:
    # The model_code directory is expected to be a sibling of the directory containing app.py
    # or directly at /app/model_code if app.py is at /app
    abs_path = os.path.join(app_dir, path_suffix)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

# Model-specific imports
try:
    from cross_efficient_vit import CrossEfficientViT
    from efficient_vit import (
        EfficientViT,
    )  # Retained for potential EfficientViT variant
    from facenet_pytorch import MTCNN
    from albumentations import Compose, PadIfNeeded
    from transforms.albu import IsotropicResize
except ImportError as e:
    print(
        f"Error importing model-specific modules: {e}. Check PYTHONPATH and cloned repo structure.",
        file=sys.stderr,
    )
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print(
        f"Attempted paths from app_dir ({app_dir}): {[os.path.join(app_dir, p) for p in module_paths_to_add]}",
        file=sys.stderr,
    )
    raise

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cross-Efficient-ViT Video Deepfake Detection Service",
    description="Detects deepfakes in videos using Cross-Efficient-ViT or Efficient-ViT models by processing face crops from video frames.",
    version="1.0.1",  # Incremented
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME_DISPLAY = "cross_efficient_vit_service"
DEVICE = torch.device("cpu")

PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "false").lower() == "true"
MODEL_TIMEOUT = int(os.environ.get("MODEL_TIMEOUT", "900"))
DEFAULT_MODEL_VARIANT = os.environ.get("DEFAULT_MODEL_VARIANT", "cross_efficient_vit")

MODEL_PATHS = {
    "cross_efficient_vit": os.environ.get(
        "CROSS_EFFICIENT_VIT_MODEL_PATH",
        "/app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth",
    ),
    "efficient_vit": os.environ.get(
        "EFFICIENT_VIT_MODEL_PATH",
        "/app/model_code/efficient-vit/pretrained_models/efficient_vit.pth",
    ),
}
CONFIG_PATHS = {
    "cross_efficient_vit": os.environ.get(
        "CROSS_EFFICIENT_VIT_CONFIG_PATH",
        "/app/model_code/cross-efficient-vit/configs/architecture.yaml",
    ),
    "efficient_vit": os.environ.get(
        "EFFICIENT_VIT_CONFIG_PATH",
        "/app/model_code/efficient-vit/configs/architecture.yaml",
    ),
}

FRAMES_PER_VIDEO_TO_SAMPLE = int(os.environ.get("FRAMES_PER_VIDEO", "15"))
FACE_DETECTOR_THRESHOLD = [0.7, 0.8, 0.8]
MTCNN_MIN_FACE_SIZE = 40  # As in original repo

# ImageNet Normalization (crucial for EfficientNet backbones)
IMAGENET_NORMALIZE_TRANSFORM = T.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


@dataclass
class FaceDetail:
    """Details about a detected face in a frame."""
    face_id: int
    bbox: List[int]  # [x, y, width, height]
    score: float
    confidence: float  # MTCNN detection confidence


@dataclass
class FrameDetail:
    """Details about a processed frame."""
    frame_index: int
    timestamp_seconds: float
    faces: List[FaceDetail]
    frame_score: Optional[float] = None  # Average score of all faces in frame


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result with per-frame details."""
    probability: float
    prediction: int
    class_label: str
    total_frames_sampled: int
    total_faces_analyzed: int
    frame_details: List[FrameDetail]
    per_face_scores: List[float]


model_instance: Optional[torch.nn.Module] = None
model_config_loaded: Optional[Dict] = None
model_variant_loaded: Optional[str] = None
face_detector_mtcnn: Optional[MTCNN] = None
face_transform_pipeline: Optional[Compose] = None
model_lock = threading.Lock()
last_used_time: float = 0.0


class VideoInput(BaseModel):
    video_data: str = Field(..., description="Base64 encoded video data string")
    threshold: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for final video score",
    )
    model_variant: Optional[str] = Field(
        DEFAULT_MODEL_VARIANT, pattern="^(cross_efficient_vit|efficient_vit)$"
    )


def _create_face_transform(image_size: int) -> Compose:
    return Compose(
        [
            IsotropicResize(
                max_side=image_size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC,
            ),
            PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ]
    )


def load_model_internal(variant_to_load: str):
    global model_instance, model_config_loaded, model_variant_loaded, last_used_time
    global face_detector_mtcnn, face_transform_pipeline

    with model_lock:
        if model_instance and model_variant_loaded == variant_to_load:
            last_used_time = time.time()
            return

        logger.info(f"Loading model variant '{variant_to_load}' on {DEVICE}...")
        model_path = MODEL_PATHS.get(variant_to_load)
        config_path = CONFIG_PATHS.get(variant_to_load)

        if not model_path or not os.path.exists(model_path):
            logger.error(
                f"Model weights for '{variant_to_load}' not found at {model_path}"
            )
            raise FileNotFoundError(
                f"Model weights for '{variant_to_load}' not found at {model_path}"
            )
        if not config_path or not os.path.exists(config_path):
            logger.error(
                f"Config file for '{variant_to_load}' not found at {config_path}"
            )
            raise FileNotFoundError(
                f"Config file for '{variant_to_load}' not found at {config_path}"
            )

        try:
            with open(config_path, "r") as ymlfile:
                cfg = yaml.safe_load(ymlfile)

            image_size = cfg["model"]["image-size"]

            if variant_to_load == "cross_efficient_vit":
                # Verify channel configurations are present (example, actual check should be done by user in YAML)
                if (
                    "sm-channels" not in cfg["model"]
                    or "lg-channels" not in cfg["model"]
                ):
                    logger.warning(
                        f"CrossEfficientViT config for '{variant_to_load}' is missing 'sm-channels' or 'lg-channels'. Ensure these match EfficientNet B0 outputs at specified blocks (e.g., 1280 for block 16, 24 for block 1)."
                    )
                _model = CrossEfficientViT(config=cfg)
            elif variant_to_load == "efficient_vit":
                channels = 1280  # Default for B0
                if cfg["model"].get("selected_efficient_net", 0) == 7:
                    channels = 2560  # For B7
                _model = EfficientViT(
                    config=cfg,
                    channels=channels,
                    selected_efficient_net=cfg["model"].get(
                        "selected_efficient_net", 0
                    ),
                )
            else:
                raise ValueError(f"Unknown model variant: {variant_to_load}")

            checkpoint = torch.load(model_path, map_location=DEVICE)
            state_dict = checkpoint.get(
                "state_dict", checkpoint.get("model", checkpoint)
            )

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            _model.load_state_dict(new_state_dict, strict=True)

            _model.to(DEVICE)
            _model.eval()

            if face_detector_mtcnn is None:
                logger.info(
                    f"Initializing MTCNN face detector on {DEVICE} with min_face_size={MTCNN_MIN_FACE_SIZE}..."
                )
                face_detector_mtcnn = MTCNN(
                    keep_all=True,
                    device=DEVICE,
                    thresholds=FACE_DETECTOR_THRESHOLD,
                    min_face_size=MTCNN_MIN_FACE_SIZE,
                    select_largest=False,
                )

            # Re-initialize transform pipeline if image size changes or not yet initialized
            current_pipeline_size = (
                model_config_loaded["model"]["image-size"]
                if model_config_loaded
                else -1
            )
            if face_transform_pipeline is None or current_pipeline_size != image_size:
                logger.info(
                    f"Creating face transform pipeline for image size: {image_size}"
                )
                face_transform_pipeline = _create_face_transform(image_size)

            model_instance = _model
            model_config_loaded = cfg
            model_variant_loaded = variant_to_load
            last_used_time = time.time()
            logger.info(f"Model variant '{variant_to_load}' loaded successfully.")

        except Exception as e:
            logger.exception(f"Failed to load model variant '{variant_to_load}': {e}")
            model_instance, model_config_loaded, model_variant_loaded = None, None, None
            raise
        finally:
            gc.collect()


def ensure_model_loaded(variant_to_load: str):
    if not model_instance or model_variant_loaded != variant_to_load:
        load_model_internal(variant_to_load)
    else:
        global last_used_time
        last_used_time = time.time()


def unload_model_if_idle():
    global model_instance, model_config_loaded, model_variant_loaded, last_used_time
    # Removed face_detector_mtcnn, face_transform_pipeline from global to avoid unsetting them if shared
    if model_instance is None or PRELOAD_MODEL:
        return

    with model_lock:
        if model_instance is not None and (
            time.time() - last_used_time > MODEL_TIMEOUT
        ):
            logger.info(f"Unloading model '{model_variant_loaded}' due to inactivity.")
            del model_instance
            model_instance = None
            model_config_loaded, model_variant_loaded = (
                None,
                None,
            )  # Reset these as they are model specific
            # Face detector and main transform pipeline can persist if not image_size dependent or re-created on load
            gc.collect()
            logger.info("Model unloaded.")


@dataclass
class ExtractedFrame:
    """Container for an extracted frame with metadata."""
    frame_index: int
    timestamp_seconds: float
    frame_rgb: np.ndarray


def extract_frames_from_video_bytes(
    video_bytes: bytes, num_frames_to_sample: int
) -> Tuple[List[ExtractedFrame], float, float]:
    """
    Extract frames from video bytes.

    Returns:
        Tuple of (list of ExtractedFrame, video_duration_seconds, fps)
    """
    temp_video_path = f"/tmp/temp_video_{os.urandom(8).hex()}.mp4"  # Using os.urandom for unique filename
    frames: List[ExtractedFrame] = []
    video_duration = 0.0
    fps = 0.0

    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open temporary video file: {temp_video_path}")
            return frames, 0.0, 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps <= 0:
            cap.release()
            return frames, 0.0, 0.0

        video_duration = total_frames / fps

        frame_indices = np.linspace(
            0, total_frames - 1, num_frames_to_sample, dtype=int
        )

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps
                frames.append(ExtractedFrame(
                    frame_index=int(idx),
                    timestamp_seconds=round(timestamp, 2),
                    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ))

        cap.release()
        return frames, video_duration, fps
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as e:
                logger.warning(
                    f"Could not remove temp video file {temp_video_path}: {e}"
                )


def process_video_and_predict(
    video_bytes: bytes, input_threshold: float, variant_to_load: str
) -> VideoAnalysisResult:
    """
    Process video and return detailed analysis with per-frame/per-face scores.

    Returns:
        VideoAnalysisResult with complete analysis details
    """
    ensure_model_loaded(variant_to_load)
    if not all(
        [
            model_instance,
            face_detector_mtcnn,
            face_transform_pipeline,
            model_config_loaded,
        ]
    ):
        raise HTTPException(
            status_code=503,
            detail=f"Model or preprocessor for '{variant_to_load}' not available.",
        )

    extracted_frames, video_duration, fps = extract_frames_from_video_bytes(
        video_bytes, FRAMES_PER_VIDEO_TO_SAMPLE
    )
    if not extracted_frames:
        logger.warning("No frames extracted from video.")
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

    for extracted_frame in extracted_frames:
        frame = extracted_frame.frame_rgb
        frame_idx = extracted_frame.frame_index
        timestamp = extracted_frame.timestamp_seconds

        frame_faces: List[FaceDetail] = []

        try:
            # MTCNN expects RGB numpy array or PIL Image
            # Frame is already HWC RGB numpy array
            boxes, probs, landmarks = face_detector_mtcnn.detect(
                frame, landmarks=True
            )  # Get landmarks for potential alignment if needed

            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(
                    zip(boxes, probs, landmarks)
                ):  # Iterate through detected faces in the frame
                    if prob < FACE_DETECTOR_THRESHOLD[2]:
                        continue  # Using MTCNN's final stage threshold as a filter

                    xmin, ymin, xmax, ymax = [int(b) for b in box]

                    # Face padding strategy (consistent with original repo's train/test)
                    w_face = xmax - xmin
                    h_face = ymax - ymin

                    # Proportional padding
                    pad_h_local = h_face // 3
                    pad_w_local = w_face // 3

                    crop_xmin = max(0, xmin - pad_w_local)
                    crop_ymin = max(0, ymin - pad_h_local)
                    crop_xmax = min(frame.shape[1], xmax + pad_w_local)
                    crop_ymax = min(frame.shape[0], ymax + pad_h_local)

                    face_crop_rgb = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                    if face_crop_rgb.size == 0:
                        logger.debug(
                            f"Empty face crop for frame {frame_idx}, face {i}. Skipping."
                        )
                        continue

                    # Albumentations transform (IsotropicResize, PadIfNeeded)
                    transformed_face = face_transform_pipeline(image=face_crop_rgb)[
                        "image"
                    ]  # HWC, uint8

                    # Convert to tensor, scale to [0,1]
                    face_tensor_float_01 = (
                        torch.from_numpy(transformed_face.astype(np.float32)).permute(
                            2, 0, 1
                        )
                        / 255.0
                    )

                    # Apply ImageNet normalization
                    normalized_face_tensor = IMAGENET_NORMALIZE_TRANSFORM(
                        face_tensor_float_01
                    )

                    face_tensor_final = normalized_face_tensor.unsqueeze(0).to(
                        DEVICE
                    )  # NCHW

                    with torch.no_grad():
                        logits = model_instance(face_tensor_final)
                        score = torch.sigmoid(logits).squeeze().item()

                    all_face_scores.append(score)

                    # Store face details with bbox in [x, y, width, height] format
                    face_details = FaceDetail(
                        face_id=i,
                        bbox=[xmin, ymin, w_face, h_face],
                        score=float(score),
                        confidence=float(prob)
                    )
                    frame_faces.append(face_details)
            else:
                logger.debug(f"No faces detected in frame {frame_idx} with MTCNN.")
        except Exception as e_frame:
            logger.warning(
                f"Error processing a frame/face (frame {frame_idx}): {e_frame}",
                exc_info=False,
            )
            continue

        # Calculate frame-level score (average of all faces in this frame)
        frame_score = None
        if frame_faces:
            frame_score = float(np.mean([f.score for f in frame_faces]))

        frame_details.append(FrameDetail(
            frame_index=frame_idx,
            timestamp_seconds=timestamp,
            faces=frame_faces,
            frame_score=frame_score
        ))

    if not all_face_scores:
        logger.info(
            "No faces detected or processed successfully across all sampled frames."
        )
        return VideoAnalysisResult(
            probability=0.5,
            prediction=0,
            class_label="real",
            total_frames_sampled=len(extracted_frames),
            total_faces_analyzed=0,
            frame_details=frame_details,
            per_face_scores=[]
        )

    # Aggregation: simple mean of scores
    final_video_prob_fake = float(np.mean(all_face_scores))
    final_video_prediction = 1 if final_video_prob_fake >= input_threshold else 0
    final_video_class_label = "fake" if final_video_prediction == 1 else "real"

    return VideoAnalysisResult(
        probability=final_video_prob_fake,
        prediction=final_video_prediction,
        class_label=final_video_class_label,
        total_frames_sampled=len(extracted_frames),
        total_faces_analyzed=len(all_face_scores),
        frame_details=frame_details,
        per_face_scores=all_face_scores
    )


@app.on_event("startup")
async def startup_event_handler():
    if PRELOAD_MODEL:
        logger.info(
            f"Preloading model variant '{DEFAULT_MODEL_VARIANT}' at service startup."
        )
        try:
            load_model_internal(DEFAULT_MODEL_VARIANT)
        except Exception as e:
            logger.error(
                f"Fatal error during model preloading: {e}. Service might not function correctly.",
                exc_info=True,
            )
    else:
        logger.info("Model will be lazy-loaded on first request.")

    active_timer = None
    if not PRELOAD_MODEL and MODEL_TIMEOUT > 0:

        def periodic_unload_check_runner():
            nonlocal active_timer
            unload_model_if_idle()
            if model_instance is not None or PRELOAD_MODEL:
                active_timer = threading.Timer(
                    MODEL_TIMEOUT / 2.0, periodic_unload_check_runner
                )
                active_timer.daemon = True
                active_timer.start()
            else:
                logger.info(
                    f"[{MODEL_NAME_DISPLAY}] Model unloaded, stopping idle check timer."
                )

        active_timer = threading.Timer(
            MODEL_TIMEOUT / 2.0, periodic_unload_check_runner
        )
        active_timer.daemon = True
        active_timer.start()
        logger.info(
            f"Model idle check timer initiated (interval: {MODEL_TIMEOUT / 2.0}s)."
        )


@app.get("/")
async def root():
    return {
        "service_name": MODEL_NAME_DISPLAY,
        "status": "online",
        "device": str(DEVICE),
        "loaded_model_variant": model_variant_loaded if model_instance else "None",
        "default_variant": DEFAULT_MODEL_VARIANT,
        "expected_model_path_crossvit": MODEL_PATHS["cross_efficient_vit"],
        "expected_config_path_crossvit": CONFIG_PATHS["cross_efficient_vit"],
    }


@app.get("/health")
async def health():
    active_variant = model_variant_loaded or DEFAULT_MODEL_VARIANT
    model_ok = True
    message = "Service healthy."

    model_path_to_check = MODEL_PATHS.get(active_variant)
    config_path_to_check = CONFIG_PATHS.get(active_variant)

    if not model_path_to_check or not os.path.exists(model_path_to_check):
        model_ok = False
        message = f"Model weights for variant '{active_variant}' not found at {model_path_to_check}."
    elif not config_path_to_check or not os.path.exists(config_path_to_check):
        model_ok = False
        message = f"Config file for variant '{active_variant}' not found at {config_path_to_check}."

    current_status = "healthy"
    if not model_ok:
        current_status = "error_missing_files"
    elif (
        not model_instance and PRELOAD_MODEL
    ):  # Preload was true, but model is not loaded
        current_status = "error_preload_failed"
        message = f"Model '{active_variant}' was set to preload but is not currently loaded. Check startup logs."
    elif (
        not model_instance and not PRELOAD_MODEL
    ):  # Lazy load enabled, model not yet loaded
        current_status = "degraded_not_loaded"
        message = f"Model '{active_variant}' ready for lazy loading."
    elif model_instance and model_variant_loaded == active_variant:
        message = f"Model '{active_variant}' loaded."

    return {
        "status": current_status,
        "model_name": MODEL_NAME_DISPLAY,
        "active_variant_checked": active_variant,
        "weights_found": (
            os.path.exists(model_path_to_check) if model_path_to_check else False
        ),
        "config_found": (
            os.path.exists(config_path_to_check) if config_path_to_check else False
        ),
        "model_currently_loaded": model_instance is not None
        and model_variant_loaded == active_variant,
        "message": message,
    }


@app.post("/unload", include_in_schema=True)
async def unload_model_endpoint():
    global model_instance, model_config_loaded, model_variant_loaded, last_used_time
    if model_instance is None:
        return {"status": "not_loaded", "message": "Model is not currently loaded."}
    with model_lock:
        if model_instance is not None:
            unloaded_variant = model_variant_loaded
            logger.info(
                f"Manually unloading model '{unloaded_variant}' via /unload endpoint."
            )
            del model_instance
            model_instance = None
            model_config_loaded, model_variant_loaded = None, None
            last_used_time = 0
            gc.collect()
            return {
                "status": "unloaded",
                "message": f"Model '{unloaded_variant}' unloaded successfully.",
            }
    return {
        "status": "error",
        "message": "Unload failed, model might have been unloaded by another request.",
    }


def _serialize_frame_details(frame_details: List[FrameDetail]) -> List[Dict[str, Any]]:
    """Serialize frame details to JSON-serializable format."""
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
    req_start_time = time.time()
    try:
        video_bytes = base64.b64decode(input_data.video_data)

        result = process_video_and_predict(
            video_bytes, input_data.threshold, input_data.model_variant
        )

        inference_time = time.time() - req_start_time
        logger.info(
            f"Video prediction for variant '{input_data.model_variant}' completed in {inference_time:.4f}s. "
            f"Prob Fake: {result.probability:.4f}, Class: {result.class_label}, "
            f"Frames: {result.total_frames_sampled}, Faces: {result.total_faces_analyzed}"
        )

        return {
            "model": f"{MODEL_NAME_DISPLAY}_{input_data.model_variant}",
            "probability": result.probability,
            "prediction": result.prediction,
            "class": result.class_label,
            "inference_time": inference_time,
            # Enhanced details for per-frame/per-face analysis
            "total_frames_sampled": result.total_frames_sampled,
            "total_faces_analyzed": result.total_faces_analyzed,
            "frame_details": _serialize_frame_details(result.frame_details),
            "per_face_scores": result.per_face_scores,
        }
    except FileNotFoundError as e:
        logger.error(f"Model file error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Model resources missing: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during video prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during video prediction: {e}",
        )


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 7001))
    logger.info(
        f"Starting {MODEL_NAME_DISPLAY} server on port {port} with device: {DEVICE}"
    )
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
