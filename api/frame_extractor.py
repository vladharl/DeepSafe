"""
Frame extraction utility for video analysis.
Extracts frames at time-based intervals for processing by image models.
"""

import base64
import io
import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for extracted frame data."""
    frame_index: int
    timestamp_seconds: float
    frame_rgb: np.ndarray  # HWC RGB numpy array
    frame_base64: str  # Base64 encoded JPEG for sending to image models
    thumbnail_base64: Optional[str] = None  # Smaller thumbnail for UI display


def encode_frame_to_base64(frame_rgb: np.ndarray, quality: int = 85) -> str:
    """
    Encode a RGB numpy array to base64 JPEG string.

    Args:
        frame_rgb: RGB numpy array (H, W, 3)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded JPEG string
    """
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)

    if not success:
        raise ValueError("Failed to encode frame to JPEG")

    return base64.b64encode(buffer).decode('utf-8')


def create_thumbnail(frame_rgb: np.ndarray, max_size: int = 200) -> str:
    """
    Create a small thumbnail for UI display.

    Args:
        frame_rgb: RGB numpy array (H, W, 3)
        max_size: Maximum dimension (width or height) of thumbnail

    Returns:
        Base64 encoded JPEG thumbnail
    """
    h, w = frame_rgb.shape[:2]

    # Calculate new size maintaining aspect ratio
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    # Resize
    thumbnail = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return encode_frame_to_base64(thumbnail, quality=70)


def extract_frames_at_interval(
    video_bytes: bytes,
    interval_seconds: float = 1.0,
    include_thumbnails: bool = True,
    max_frames: int = 60
) -> Tuple[List[FrameData], float, float]:
    """
    Extract frames from video at specified time intervals.

    Args:
        video_bytes: Raw video bytes
        interval_seconds: Time interval between frames (e.g., 1.0 for every second)
        include_thumbnails: Whether to generate thumbnails for UI display
        max_frames: Maximum number of frames to extract (safety limit)

    Returns:
        Tuple of (list of FrameData, video_duration_seconds, fps)
    """
    temp_video_path = f"/tmp/temp_video_extract_{os.urandom(8).hex()}.mp4"
    frames: List[FrameData] = []
    video_duration = 0.0
    fps = 0.0

    try:
        # Write video to temp file
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        # Open video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open temporary video file: {temp_video_path}")
            return frames, 0.0, 0.0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps <= 0:
            logger.error(f"Invalid video: total_frames={total_frames}, fps={fps}")
            cap.release()
            return frames, 0.0, 0.0

        video_duration = total_frames / fps

        # Calculate frame indices at specified intervals
        current_time = 0.0
        frame_count = 0

        while current_time < video_duration and frame_count < max_frames:
            # Convert time to frame index
            frame_index = int(current_time * fps)

            # Ensure we don't exceed total frames
            if frame_index >= total_frames:
                break

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Encode frame
                frame_base64 = encode_frame_to_base64(frame_rgb)

                # Create thumbnail if requested
                thumbnail_base64 = None
                if include_thumbnails:
                    thumbnail_base64 = create_thumbnail(frame_rgb)

                frames.append(FrameData(
                    frame_index=frame_index,
                    timestamp_seconds=round(current_time, 2),
                    frame_rgb=frame_rgb,
                    frame_base64=frame_base64,
                    thumbnail_base64=thumbnail_base64
                ))

                frame_count += 1
            else:
                logger.warning(f"Failed to read frame at index {frame_index}")

            current_time += interval_seconds

        cap.release()

        logger.info(
            f"Extracted {len(frames)} frames at {interval_seconds}s intervals "
            f"from {video_duration:.2f}s video ({fps:.2f} fps)"
        )

        return frames, video_duration, fps

    except Exception as e:
        logger.exception(f"Error extracting frames: {e}")
        return frames, video_duration, fps

    finally:
        # Cleanup temp file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as e:
                logger.warning(f"Could not remove temp video file {temp_video_path}: {e}")


def extract_frames_evenly(
    video_bytes: bytes,
    num_frames: int = 15,
    include_thumbnails: bool = True
) -> Tuple[List[FrameData], float, float]:
    """
    Extract a fixed number of frames evenly distributed across the video.
    This is the existing behavior, kept for compatibility.

    Args:
        video_bytes: Raw video bytes
        num_frames: Number of frames to extract
        include_thumbnails: Whether to generate thumbnails for UI display

    Returns:
        Tuple of (list of FrameData, video_duration_seconds, fps)
    """
    temp_video_path = f"/tmp/temp_video_extract_{os.urandom(8).hex()}.mp4"
    frames: List[FrameData] = []
    video_duration = 0.0
    fps = 0.0

    try:
        # Write video to temp file
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        # Open video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open temporary video file: {temp_video_path}")
            return frames, 0.0, 0.0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps <= 0:
            logger.error(f"Invalid video: total_frames={total_frames}, fps={fps}")
            cap.release()
            return frames, 0.0, 0.0

        video_duration = total_frames / fps

        # Calculate evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp
                timestamp = frame_index / fps

                # Encode frame
                frame_base64 = encode_frame_to_base64(frame_rgb)

                # Create thumbnail if requested
                thumbnail_base64 = None
                if include_thumbnails:
                    thumbnail_base64 = create_thumbnail(frame_rgb)

                frames.append(FrameData(
                    frame_index=int(frame_index),
                    timestamp_seconds=round(timestamp, 2),
                    frame_rgb=frame_rgb,
                    frame_base64=frame_base64,
                    thumbnail_base64=thumbnail_base64
                ))

        cap.release()

        logger.info(
            f"Extracted {len(frames)} evenly-spaced frames "
            f"from {video_duration:.2f}s video ({fps:.2f} fps)"
        )

        return frames, video_duration, fps

    except Exception as e:
        logger.exception(f"Error extracting frames: {e}")
        return frames, video_duration, fps

    finally:
        # Cleanup temp file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError as e:
                logger.warning(f"Could not remove temp video file {temp_video_path}: {e}")
