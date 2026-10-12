"""
Real-time Video Streaming & Feed Management Service
===================================================
Manages live video sources, categorizes feeds from the `sources/` folder,
and powers parallel real-time frame processing with live telemetry.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("StreamService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StreamService:
    """
    Manages categorized video sources from `sources/` and handles live stream frame processing.
    """

    CATEGORIES = {
        "robbery.mp4": {
            "category": "Retail & Indoor Security",
            "name": "Store Register Surveillance",
            "camera_id": "camera_robbery",
            "description": "Indoor POS checkout register overhead security camera",
        },
        "shop_street.mp4": {
            "category": "Street & Public Surveillance",
            "name": "Street Sidewalk Camera",
            "camera_id": "camera_store_front",
            "description": "Outdoor commercial sidewalk pedestrian overview",
        },
    }

    def __init__(
        self,
        sources_dir: Union[str, Path] = "sources",
        output_dir: Union[str, Path] = "output",
    ):
        self.sources_dir = Path(sources_dir)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)

        # Real-time processing state
        self._lock = threading.Lock()
        self.active_camera_id: str = "camera_robbery"
        self.is_processing: bool = False
        self.current_timestamp: float = 0.0
        self.frames_processed_count: int = 0
        self.last_detection_summary: List[str] = []
        self.latest_face_thumbnail: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def get_categorized_sources(self) -> List[Dict[str, Any]]:
        """
        Scans `sources/` directory and returns categorized camera feeds.
        """
        sources_list: List[Dict[str, Any]] = []
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if not self.sources_dir.exists():
            return sources_list

        for file_path in sorted(self.sources_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                fname = file_path.name
                meta = self.CATEGORIES.get(fname, {})

                category = meta.get("category", "General Security Feeds")
                cam_name = meta.get("name", file_path.stem.replace("_", " ").title())
                cam_id = meta.get("camera_id", f"camera_{file_path.stem.lower()}")
                desc = meta.get("description", f"Local video stream from {fname}")

                # Probe basic video specs
                duration = 0.0
                fps = 30.0
                width = 1280
                height = 720
                try:
                    cap = cv2.VideoCapture(str(file_path))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                        duration = total_frames / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                        cap.release()
                except Exception as e:
                    logger.debug(f"Could not probe {file_path}: {e}")

                thumb_path = self.get_source_thumbnail_path(cam_id, file_path)

                sources_list.append({
                    "id": cam_id,
                    "camera_id": cam_id,
                    "name": cam_name,
                    "category": category,
                    "filename": fname,
                    "file_path": str(file_path),
                    "description": desc,
                    "fps": round(fps, 1),
                    "duration": round(duration, 2),
                    "resolution": f"{width}x{height}",
                    "video_url": f"/api/video/{cam_id}",
                    "thumbnail_url": f"/api/sources/{cam_id}/thumbnail",
                })

        return sources_list

    def get_source_thumbnail_path(self, camera_id: str, file_path: Optional[Path] = None) -> Optional[Path]:
        """Returns or generates a single preview frame thumbnail for this camera source."""
        thumb_dir = self.output_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{camera_id}.jpg"
        if thumb_path.exists():
            return thumb_path

        # Check existing extracted frames
        frames_dir = self.output_dir / "frames" / camera_id
        if frames_dir.exists():
            existing = sorted(frames_dir.glob("*.jpg"))
            if existing:
                try:
                    # Cache downscaled thumbnail
                    img = cv2.imread(str(existing[0]))
                    if img is not None:
                        thumb = cv2.resize(img, (480, 270))
                        cv2.imwrite(str(thumb_path), thumb)
                        return thumb_path
                except Exception:
                    return existing[0]

        # Extract 1 frame from video source file
        vpath = file_path or self.get_source_file_by_camera_id(camera_id)
        if vpath and vpath.exists():
            try:
                cap = cv2.VideoCapture(str(vpath))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        thumb = cv2.resize(frame, (480, 270))
                        cv2.imwrite(str(thumb_path), thumb)
                        return thumb_path
            except Exception as exc:
                logger.debug(f"Failed to extract thumbnail for {camera_id}: {exc}")

        return None

    def get_source_file_by_camera_id(self, camera_id: str) -> Optional[Path]:
        """Resolves camera ID to its video source path in `sources/`."""
        sources = self.get_categorized_sources()
        for s in sources:
            if s["id"] == camera_id:
                return Path(s["file_path"])
        # Fallback search
        for f in self.sources_dir.iterdir():
            if f.is_file() and (f.stem == camera_id or f.stem in camera_id or camera_id in f.stem):
                return f
        return None

    def get_realtime_status(self) -> Dict[str, Any]:
        """Returns current real-time telemetry state."""
        with self._lock:
            return {
                "active_camera_id": self.active_camera_id,
                "is_processing": self.is_processing,
                "current_timestamp": round(self.current_timestamp, 3),
                "frames_processed_count": self.frames_processed_count,
                "last_detections": self.last_detection_summary,
                "latest_face_thumbnail": self.latest_face_thumbnail,
            }

    def process_incoming_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        timestamp: float,
        vision_service: Any,
        person_service: Any,
        clip_service: Any,
        vector_service: Any,
    ) -> Dict[str, Any]:
        """
        Processes a single real-time frame on the fly:
        1. YOLO Object & Person Detection
        2. Unique Person & Face Mapping (De-duplicated)
        3. CLIP Vector Embedding & ChromaDB Indexing
        """
        with self._lock:
            self.active_camera_id = camera_id
            self.current_timestamp = timestamp
            self.frames_processed_count += 1

        h, w = frame_bgr.shape[:2]

        # Save frame to disk
        frame_dir = self.output_dir / "frames" / camera_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_filename = f"{camera_id}_rt_{int(timestamp * 1000):08d}.jpg"
        frame_path = frame_dir / frame_filename
        cv2.imwrite(str(frame_path), frame_bgr)

        # 1. Run YOLO Detection
        if hasattr(vision_service, "analyze_image"):
            detections = vision_service.analyze_image(
                image_source=frame_bgr,
                camera_id=camera_id,
                timestamp=timestamp,
                frame_path_ref=str(frame_path),
                save_to_db=True,
            )
        else:
            detections = vision_service.analyze_single_frame(
                frame_input=frame_bgr,
                camera_id=camera_id,
                timestamp=timestamp,
                frame_path=str(frame_path),
                save_to_db=True,
            )

        detected_summary = [d.class_name for d in detections]
        with self._lock:
            self.last_detection_summary = detected_summary[:5]

        # 2. Map Any People / Faces
        mapped_ids = []
        for det in detections:
            if det.is_person and det.crop_path and Path(det.crop_path).exists():
                crop_img = cv2.imread(det.crop_path)
                if crop_img is not None:
                    crop_emb = clip_service.encode_image(crop_img)
                    pid, is_new = person_service.map_or_identify_person(
                        crop_image=crop_img,
                        frame_path=str(frame_path),
                        camera_id=camera_id,
                        timestamp=timestamp,
                        embedding=crop_emb,
                        upper_color=det.upper_color,
                        lower_color=det.lower_color,
                        dominant_color=det.dominant_color,
                        full_crop_path=det.crop_path,
                    )
                    mapped_ids.append(pid)

        # 3. Compute Frame Embedding & Index into ChromaDB
        frame_emb = clip_service.encode_image(frame_bgr)
        frame_record = {
            "frame_path": str(frame_path),
            "timestamp": timestamp,
            "camera_id": camera_id,
            "frame_index": self.frames_processed_count,
            "width": w,
            "height": h,
            "extra_metadata": {"mapped_persons": ",".join(mapped_ids)},
        }
        vector_service.index_frame(
            frame_path=str(frame_path),
            timestamp=timestamp,
            camera_id=camera_id,
            embedding=frame_emb,
            record=frame_record,
        )

        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frame_path": str(frame_path),
            "detections_count": len(detections),
            "mapped_persons": mapped_ids,
        }

    def switch_stream_and_prewarm(
        self,
        camera_id: str,
        vision_service: Any,
        person_service: Any,
        clip_service: Any,
        vector_service: Any,
        sample_count: int = 6,
    ) -> Dict[str, Any]:
        """
        Switches the active camera feed and immediately samples key frames to detect and map faces.
        Guarantees that switching to any camera feed processes its people right away.
        """
        with self._lock:
            self.active_camera_id = camera_id

        video_path = self.get_source_file_by_camera_id(camera_id)
        if not video_path or not video_path.exists():
            return {"status": "error", "message": f"Source video not found for {camera_id}"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open video {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0

        # Select evenly distributed sample timestamps across video
        if duration <= 2.0:
            sample_timestamps = [0.5]
        else:
            step = max(2.0, duration / (sample_count + 1))
            sample_timestamps = [round(step * (i + 1), 2) for i in range(sample_count)]

        processed_records = []
        newly_mapped = []

        for ts in sample_timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            result = self.process_incoming_frame(
                frame_bgr=frame,
                camera_id=camera_id,
                timestamp=ts,
                vision_service=vision_service,
                person_service=person_service,
                clip_service=clip_service,
                vector_service=vector_service,
            )
            processed_records.append(result)
            if result.get("mapped_persons"):
                newly_mapped.extend(result["mapped_persons"])

        cap.release()

        # Fetch current mapped persons for this camera
        current_persons = person_service.get_all_mapped_persons(camera_id=camera_id)

        return {
            "status": "success",
            "camera_id": camera_id,
            "sampled_frames_count": len(processed_records),
            "mapped_persons_count": len(current_persons),
            "persons": current_persons,
        }
