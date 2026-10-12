"""
Video Ingestion Service Module
==============================
A robust, modular, and production-grade video ingestion system designed for
multi-camera surveillance and batch video processing.

Features:
- Dynamic frame sampling: fixed FPS, interval-based, adaptive motion-driven, or custom callback.
- Robust edge case handling: corrupted frames, stream drops, EOF, reconnect retries.
- Fast decoding optimization (using grab/retrieve for skipped frames).
- Multi-camera and batch parallel processing with thread pooling.
- Clean structured output: returns dictionary arrays of frame file paths mapped to timestamps.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Configure module-level logger
logger = logging.getLogger("VideoIngestionService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SamplingMode(str, Enum):
    """Supported frame sampling modes."""
    FIXED_FPS = "fixed_fps"              # Uniform time sampling (e.g. 1 frame every 0.5s)
    EVERY_N_FRAMES = "every_n_frames"    # Skip every N decoded frames
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"# Dynamically varies FPS based on scene motion/activity
    DYNAMIC_CALLBACK = "dynamic_callback"# Uses a user-supplied dynamic rate callback


@dataclass
class FrameRecord:
    """Metadata representing an ingested and stored video frame."""
    frame_path: str
    timestamp: float                    # Timestamp in seconds from video start
    frame_index: int                    # Native frame index in the source video
    source_id: str                      # Camera ID or source identifier
    width: int                          # Frame width in pixels
    height: int                         # Frame height in pixels
    motion_score: Optional[float] = None# Optional motion delta score (0.0 to 1.0)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record into a clean serializable dictionary."""
        data = asdict(self)
        # Format timestamp to 3 decimal places for clean representation
        data["timestamp"] = round(self.timestamp, 3)
        if self.motion_score is not None:
            data["motion_score"] = round(self.motion_score, 4)
        return data


@dataclass
class IngestionConfig:
    """Configuration settings for video ingestion and sampling."""
    # Sampling configuration
    sampling_mode: SamplingMode = SamplingMode.FIXED_FPS
    target_fps: float = 1.0             # Desired FPS for FIXED_FPS mode
    frame_interval: int = 15            # Interval for EVERY_N_FRAMES mode

    # Dynamic Adaptive Motion Sampling parameters
    min_fps: float = 0.5                # Minimum FPS when scene is static
    max_fps: float = 5.0                # Maximum FPS when high motion occurs
    motion_threshold: float = 0.03      # Motion sensitivity threshold (0.01 - 0.20)
    motion_downscale_dim: Tuple[int, int] = (160, 120)  # Downscale size for fast motion calc

    # Dynamic callback function: signature (timestamp_sec, frame_idx, current_frame) -> target_fps
    dynamic_rate_fn: Optional[Callable[[float, int, np.ndarray], float]] = None

    # Storage options
    image_format: str = "jpg"           # "jpg", "png", "webp"
    jpeg_quality: int = 95              # Quality (1-100) for JPG
    resize_output: Optional[Tuple[int, int]] = None  # Optional (width, height) resize
    preserve_aspect_ratio: bool = True

    # Robustness & Stream resilience
    max_consecutive_errors: int = 50    # Tolerable corrupted/empty frames before aborting
    stream_reconnect_attempts: int = 3  # Retries for dropped RTSP/network streams
    reconnect_delay_sec: float = 1.5    # Backoff delay between reconnect attempts
    stream_read_timeout_sec: float = 10.0# Max silence before considering stream dead

    # Clipping and boundaries
    start_time_sec: float = 0.0         # Seek start position in seconds
    end_time_sec: Optional[float] = None# Optional end time in seconds
    max_frames: Optional[int] = None    # Cap on total saved frames per source


class MotionDetector:
    """Lightweight and high-performance motion estimator for adaptive sampling."""

    def __init__(self, threshold: float = 0.03, downscale_dim: Tuple[int, int] = (160, 120)):
        self.threshold = threshold
        self.downscale_dim = downscale_dim
        self.prev_gray: Optional[np.ndarray] = None

    def compute_motion_score(self, frame: np.ndarray) -> float:
        """
        Calculates a normalized motion score between 0.0 (static) and 1.0 (major change).
        Uses ultra-fast downscaled grayscale frame differencing.
        """
        # Downsample and convert to grayscale for minimal CPU overhead
        resized = cv2.resize(frame, self.downscale_dim, interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Apply slight blur to suppress sensor noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        # Binary thresholding on intensity delta
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Ratio of moving pixels
        non_zero = cv2.countNonZero(thresh)
        total_pixels = self.downscale_dim[0] * self.downscale_dim[1]
        motion_ratio = float(non_zero) / float(total_pixels)

        # Update reference frame
        self.prev_gray = gray
        return min(motion_ratio, 1.0)


class VideoIngestionService:
    """
    Modular Video Ingestion Service for single video sources, multi-camera feeds,
    and batch video processing.

    Handles dynamic sampling rates, edge cases (corrupt frames, EOF, drops),
    and outputs a clean list/array of dictionary mappings {frame_path, timestamp, ...}.
    """

    def __init__(self, default_config: Optional[IngestionConfig] = None):
        """
        Initialize the Video Ingestion Service.

        Args:
            default_config: Optional IngestionConfig. If omitted, default settings are used.
        """
        self.config = default_config or IngestionConfig()

    @staticmethod
    def to_timestamp_map(records: List[Dict[str, Any]]) -> Dict[float, str]:
        """Convert frame records array to a dictionary mapping timestamp -> frame_path."""
        return {rec["timestamp"]: rec["frame_path"] for rec in records}

    @staticmethod
    def to_frame_path_map(records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Convert frame records array to a dictionary mapping frame_path -> timestamp."""
        return {rec["frame_path"]: rec["timestamp"] for rec in records}

    def process_video(
        self,
        video_source: Union[str, Path, int],
        output_folder: Union[str, Path],
        source_id: Optional[str] = None,
        config: Optional[IngestionConfig] = None,
        progress_callback: Optional[Callable[[str, float, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest and sample a single video source (file path, stream URL, or webcam index).

        Args:
            video_source: Path to video file, RTSP/HTTP URL, or camera device index.
            output_folder: Target directory where sampled frames will be saved.
            source_id: Unique identifier for this source/camera (defaults to filename).
            config: Optional override of ingestion configuration for this specific source.
            progress_callback: Optional callable(source_id, current_timestamp, frames_saved).

        Returns:
            List of dictionaries, each containing:
            {
                "frame_path": str,
                "timestamp": float,
                "frame_index": int,
                "source_id": str,
                "width": int,
                "height": int,
                "motion_score": Optional[float]
            }
        """
        cfg = config or self.config
        source_str = str(video_source)
        resolved_source_id = source_id or self._derive_source_id(video_source)

        # Ensure output directory exists
        target_dir = Path(output_folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting video ingestion for source='{resolved_source_id}' "
            f"(sampling_mode={cfg.sampling_mode.value}) -> output='{target_dir}'"
        )

        records: List[FrameRecord] = []
        reconnect_count = 0

        while reconnect_count <= cfg.stream_reconnect_attempts:
            cap = self._open_capture(video_source)
            if cap is None or not cap.isOpened():
                reconnect_count += 1
                if reconnect_count <= cfg.stream_reconnect_attempts:
                    logger.warning(
                        f"Failed to open source '{video_source}'. "
                        f"Retrying ({reconnect_count}/{cfg.stream_reconnect_attempts}) in {cfg.reconnect_delay_sec}s..."
                    )
                    time.sleep(cfg.reconnect_delay_sec)
                    continue
                else:
                    logger.error(f"Unable to open video source: '{video_source}' after {cfg.stream_reconnect_attempts} attempts.")
                    break

            try:
                # Perform the frame ingestion loop
                records = self._ingest_stream_loop(
                    cap=cap,
                    video_source=video_source,
                    target_dir=target_dir,
                    source_id=resolved_source_id,
                    config=cfg,
                    existing_records=records,
                    progress_callback=progress_callback,
                )
                # Successful or clean EOF completed
                break

            except Exception as exc:
                logger.error(f"Unexpected error while processing source '{resolved_source_id}': {exc}", exc_info=True)
                reconnect_count += 1
                if reconnect_count <= cfg.stream_reconnect_attempts and self._is_live_stream(video_source):
                    logger.info(f"Attempting live stream reconnection for '{resolved_source_id}'...")
                    time.sleep(cfg.reconnect_delay_sec)
                else:
                    break
            finally:
                if cap is not None:
                    cap.release()

        logger.info(f"Ingestion complete for '{resolved_source_id}'. Extracted {len(records)} frames.")
        return [r.to_dict() for r in records]

    def process_batch(
        self,
        video_sources: Iterable[Union[str, Path, int]],
        output_folder: Union[str, Path],
        config: Optional[IngestionConfig] = None,
        max_workers: Optional[int] = 4,
        separate_source_folders: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple video files or sources concurrently in batch mode.

        Args:
            video_sources: Collection of video paths or stream identifiers.
            output_folder: Base directory for output frames.
            config: Ingestion configuration (defaults to service config).
            max_workers: Maximum parallel workers in the thread pool.
            separate_source_folders: If True, each source gets its own subfolder inside output_folder.

        Returns:
            Dictionary mapping source_id -> clean dictionary array of frame records.
        """
        sources_list = list(video_sources)
        results: Dict[str, List[Dict[str, Any]]] = {}

        if not sources_list:
            logger.warning("Empty source list provided to process_batch.")
            return results

        base_output = Path(output_folder)
        base_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing batch of {len(sources_list)} sources with max_workers={max_workers}...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_src = {}
            for src in sources_list:
                src_id = self._derive_source_id(src)
                dest_dir = base_output / src_id if separate_source_folders else base_output
                future = executor.submit(
                    self.process_video,
                    video_source=src,
                    output_folder=dest_dir,
                    source_id=src_id,
                    config=config,
                )
                future_to_src[future] = src_id

            for future in as_completed(future_to_src):
                src_id = future_to_src[future]
                try:
                    frame_dicts = future.result()
                    results[src_id] = frame_dicts
                except Exception as exc:
                    logger.error(f"Source '{src_id}' failed in batch execution: {exc}")
                    results[src_id] = []

        return results

    def process_multi_camera(
        self,
        camera_sources: Dict[str, Union[str, Path, int]],
        output_folder: Union[str, Path],
        config: Optional[IngestionConfig] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Specialized multi-camera processor mapping camera IDs to their respective feeds.

        Args:
            camera_sources: Dict mapping camera_id -> stream URI / file path / device index.
                            Example: {"cam_front": "rtsp://...", "cam_back": "sources/back.mp4"}
            output_folder: Base target folder where each camera gets a dedicated directory.
            config: Ingestion configuration for the cameras.
            max_workers: Concurrency limit (defaults to number of cameras).

        Returns:
            Dict mapping camera_id -> dictionary array of frame records.
        """
        workers = max_workers or len(camera_sources) or 1
        base_output = Path(output_folder)
        results: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"Initiating multi-camera ingestion for {len(camera_sources)} cameras (workers={workers}).")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_cam = {}
            for cam_id, src in camera_sources.items():
                dest_dir = base_output / cam_id
                future = executor.submit(
                    self.process_video,
                    video_source=src,
                    output_folder=dest_dir,
                    source_id=cam_id,
                    config=config,
                )
                future_to_cam[future] = cam_id

            for future in as_completed(future_to_cam):
                cam_id = future_to_cam[future]
                try:
                    frame_dicts = future.result()
                    results[cam_id] = frame_dicts
                except Exception as exc:
                    logger.error(f"Multi-camera ingestion failed for camera '{cam_id}': {exc}")
                    results[cam_id] = []

        return results

    # ==========================================
    # Internal Stream Ingestion & Processing Core
    # ==========================================

    def _ingest_stream_loop(
        self,
        cap: cv2.VideoCapture,
        video_source: Union[str, Path, int],
        target_dir: Path,
        source_id: str,
        config: IngestionConfig,
        existing_records: List[FrameRecord],
        progress_callback: Optional[Callable[[str, float, int], None]] = None,
    ) -> List[FrameRecord]:
        """Core decoding loop with dynamic sampling rate calculation and error recovery."""
        records = list(existing_records)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or np.isnan(native_fps) or native_fps <= 0:
            native_fps = 30.0  # Fallback for streams with missing FPS header

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_live = self._is_live_stream(video_source)

        # Seek start position if requested and supported
        if config.start_time_sec > 0 and not is_live:
            cap.set(cv2.CAP_PROP_POS_MSEC, config.start_time_sec * 1000.0)

        motion_detector = MotionDetector(
            threshold=config.motion_threshold,
            downscale_dim=config.motion_downscale_dim
        ) if config.sampling_mode == SamplingMode.DYNAMIC_ADAPTIVE else None

        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if not is_live else 0
        last_sampled_timestamp = -999.0
        consecutive_errors = 0
        current_dynamic_fps = config.target_fps

        while True:
            # Check frame cap limit
            if config.max_frames is not None and len(records) >= config.max_frames:
                logger.info(f"Reached max frame limit ({config.max_frames}) for source '{source_id}'.")
                break

            # Read frame from stream
            ret, frame = cap.read()

            # Handle edge cases: frame corrupted or end of file
            if not ret or frame is None or frame.size == 0:
                consecutive_errors += 1

                # If this is a regular video file, check if we naturally reached End-of-File (EOF)
                if not is_live:
                    pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if (total_frames > 0 and pos_frames >= total_frames) or consecutive_errors >= 5:
                        logger.info(f"Reached End-Of-File (EOF) gracefully for source '{source_id}'.")
                        break

                # For live streams or unexpected corrupt frames
                if consecutive_errors >= config.max_consecutive_errors:
                    logger.warning(
                        f"Exceeded max consecutive error limit ({config.max_consecutive_errors}) "
                        f"on source '{source_id}'. Concluding stream extraction."
                    )
                    break

                # Small delay to prevent tight spin on stalled network streams
                if is_live:
                    time.sleep(0.01)
                continue

            # Reset consecutive error counter upon successful decode
            consecutive_errors = 0
            current_frame_idx += 1

            # Determine timestamp
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                current_timestamp = pos_msec / 1000.0
            else:
                # Calculate from frame index and native FPS
                current_timestamp = current_frame_idx / native_fps

            # Check end boundary if configured
            if config.end_time_sec is not None and current_timestamp > config.end_time_sec:
                logger.info(f"Reached configured end time ({config.end_time_sec}s) for source '{source_id}'.")
                break

            # Dynamic Sampling Rate Calculation
            motion_score: Optional[float] = None
            should_sample = False

            if config.sampling_mode == SamplingMode.FIXED_FPS:
                target_interval = 1.0 / max(config.target_fps, 0.001)
                if (current_timestamp - last_sampled_timestamp) >= (target_interval - 1e-4):
                    should_sample = True

            elif config.sampling_mode == SamplingMode.EVERY_N_FRAMES:
                if current_frame_idx % max(config.frame_interval, 1) == 0:
                    should_sample = True

            elif config.sampling_mode == SamplingMode.DYNAMIC_ADAPTIVE:
                assert motion_detector is not None
                motion_score = motion_detector.compute_motion_score(frame)

                # Dynamically calculate adaptive FPS:
                # Linear/curved interpolation between min_fps and max_fps based on motion intensity
                normalized_motion = min(motion_score / max(config.motion_threshold, 1e-5), 1.0)
                current_dynamic_fps = config.min_fps + (config.max_fps - config.min_fps) * normalized_motion
                adaptive_interval = 1.0 / max(current_dynamic_fps, 0.001)

                if (current_timestamp - last_sampled_timestamp) >= (adaptive_interval - 1e-4):
                    should_sample = True

            elif config.sampling_mode == SamplingMode.DYNAMIC_CALLBACK:
                if config.dynamic_rate_fn:
                    try:
                        current_dynamic_fps = config.dynamic_rate_fn(current_timestamp, current_frame_idx, frame)
                    except Exception as fn_err:
                        logger.error(f"Error in dynamic_rate_fn: {fn_err}. Falling back to 1.0 FPS.")
                        current_dynamic_fps = 1.0
                target_interval = 1.0 / max(current_dynamic_fps, 0.001)
                if (current_timestamp - last_sampled_timestamp) >= (target_interval - 1e-4):
                    should_sample = True

            if should_sample:
                # Save the frame to disk
                saved_path = self._save_frame(
                    frame=frame,
                    target_dir=target_dir,
                    source_id=source_id,
                    frame_idx=current_frame_idx,
                    timestamp=current_timestamp,
                    config=config,
                )

                if saved_path:
                    height, width = frame.shape[:2]
                    record = FrameRecord(
                        frame_path=str(saved_path),
                        timestamp=current_timestamp,
                        frame_index=current_frame_idx,
                        source_id=source_id,
                        width=width,
                        height=height,
                        motion_score=motion_score,
                    )
                    records.append(record)
                    last_sampled_timestamp = current_timestamp

                    if progress_callback:
                        progress_callback(source_id, current_timestamp, len(records))

        return records

    # ==========================================
    # Helper & Utility Methods
    # ==========================================

    def _open_capture(self, video_source: Union[str, Path, int]) -> Optional[cv2.VideoCapture]:
        """Safely opens cv2.VideoCapture with format-appropriate flags."""
        try:
            if isinstance(video_source, (int, float)):
                # Webcam device index
                cap = cv2.VideoCapture(int(video_source))
            else:
                src_str = str(video_source)
                # Check if local file exists
                if not (src_str.startswith("rtsp://") or src_str.startswith("http://") or src_str.startswith("https://")):
                    p = Path(src_str)
                    if not p.exists():
                        logger.error(f"Video file does not exist: {p.resolve()}")
                        return None
                cap = cv2.VideoCapture(src_str)

            return cap
        except Exception as exc:
            logger.error(f"Exception opening capture '{video_source}': {exc}")
            return None

    def _save_frame(
        self,
        frame: np.ndarray,
        target_dir: Path,
        source_id: str,
        frame_idx: int,
        timestamp: float,
        config: IngestionConfig,
    ) -> Optional[Path]:
        """Processes frame transforms (resize, quality) and persists to disk safely."""
        try:
            processed_frame = frame

            # Handle optional resizing
            if config.resize_output:
                req_w, req_h = config.resize_output
                if config.preserve_aspect_ratio:
                    h, w = frame.shape[:2]
                    scale = min(req_w / w, req_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    processed_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    processed_frame = cv2.resize(frame, (req_w, req_h), interpolation=cv2.INTER_AREA)

            # Construct clean filename: {source_id}_frame_{frame_index:06d}_{timestamp:.3f}s.{ext}
            clean_source_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_id)
            fmt = config.image_format.lower().strip(".")
            filename = f"{clean_source_id}_frame_{frame_idx:06d}_{timestamp:08.3f}s.{fmt}"
            file_path = target_dir / filename

            # Compression parameters
            params = []
            if fmt in ("jpg", "jpeg"):
                params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, config.jpeg_quality))]
            elif fmt == "png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
            elif fmt == "webp":
                params = [int(cv2.IMWRITE_WEBP_QUALITY), max(1, min(100, config.jpeg_quality))]

            success = cv2.imwrite(str(file_path), processed_frame, params)
            if not success:
                logger.error(f"cv2.imwrite failed for destination '{file_path}'")
                return None

            return file_path

        except Exception as exc:
            logger.error(f"Failed to save frame at timestamp {timestamp:.3f}s: {exc}")
            return None

    def _is_live_stream(self, source: Union[str, Path, int]) -> bool:
        """Determines whether the source is a live stream or local video file."""
        if isinstance(source, (int, float)):
            return True
        s = str(source).lower()
        return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://") or s.startswith("rtmp://")

    def _derive_source_id(self, source: Union[str, Path, int]) -> str:
        """Derives a human-friendly unique identifier from the source path/URI."""
        if isinstance(source, (int, float)):
            return f"device_camera_{int(source)}"
        p = Path(str(source))
        stem = p.stem.strip()
        if stem:
            return stem
        # If URL without clean stem, create safe identifier
        clean = "".join(c if c.isalnum() else "_" for c in str(source))
        return clean[:32]
