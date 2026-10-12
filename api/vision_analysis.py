"""
Vision Processing Module for Surveillance Frame Analysis
=========================================================
Uses Ultralytics YOLO (YOLOv8 / YOLO11) to analyze video frames:
- Detects humans and general objects (vehicles, bags, animals, etc.).
- Crops bounding boxes for detected people and saves them to disk.
- Extracts dominant colors (upper body, lower body, overall) and attribute tags.
- Stores structured detection results into a relational SQLite database
  with optimized indices for fast filtering and temporal surveillance queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# Module-level logger
logger = logging.getLogger("VisionAnalysisService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class DetectionRecord:
    """Structured detection record for storage and relational filtering."""
    camera_id: str
    timestamp: float
    frame_path: str
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    is_person: bool = False
    crop_path: Optional[str] = None
    dominant_color: Optional[str] = None
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    attribute_tags: List[str] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        d = asdict(self)
        d["timestamp"] = round(self.timestamp, 3)
        d["confidence"] = round(self.confidence, 4)
        d["bbox"] = [round(c, 2) for c in self.bbox]
        return d


# =====================================================================
# Attribute & Color Extraction
# =====================================================================

class ColorAttributeExtractor:
    """
    Fast and robust HSV-based color analyzer and attribute tagger for object crops.
    """

    COLOR_RANGES = {
        "black": {"s_max": 255, "v_max": 45},
        "white": {"s_max": 35, "v_min": 200},
        "gray": {"s_max": 45, "v_min": 45, "v_max": 200},
        "red": [{"h_min": 0, "h_max": 10}, {"h_min": 165, "h_max": 180}],
        "orange": {"h_min": 11, "h_max": 25},
        "yellow": {"h_min": 26, "h_max": 35},
        "green": {"h_min": 36, "h_max": 85},
        "cyan": {"h_min": 86, "h_max": 100},
        "blue": {"h_min": 101, "h_max": 135},
        "purple": {"h_min": 136, "h_max": 150},
        "pink": {"h_min": 151, "h_max": 164},
    }

    @classmethod
    def classify_pixel_hsv(cls, h: int, s: int, v: int) -> str:
        """Classify an individual HSV pixel into a color category."""
        if v <= 45:
            return "black"
        if s <= 35 and v >= 200:
            return "white"
        if s <= 45 and 45 < v < 200:
            return "gray"

        # Check chromatic ranges
        if h <= 10 or h >= 165:
            return "red"
        elif 10 < h <= 25:
            # Distinguish brown vs orange by brightness
            return "brown" if v < 120 and s > 50 else "orange"
        elif 25 < h <= 35:
            return "yellow"
        elif 35 < h <= 85:
            return "green"
        elif 85 < h <= 100:
            return "cyan"
        elif 100 < h <= 135:
            return "blue"
        elif 135 < h <= 150:
            return "purple"
        elif 150 < h < 165:
            return "pink"
        return "unknown"

    @classmethod
    def get_dominant_color(cls, crop: np.ndarray) -> str:
        """Determines dominant color name for an image crop using downsampled HSV histogram."""
        if crop is None or crop.size == 0:
            return "unknown"

        # Downsample crop for ultra-fast processing
        small = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        color_counts: Dict[str, int] = {}
        for row in hsv:
            for pixel in row:
                c = cls.classify_pixel_hsv(int(pixel[0]), int(pixel[1]), int(pixel[2]))
                if c != "unknown":
                    color_counts[c] = color_counts.get(c, 0) + 1

        if not color_counts:
            return "unknown"

        return max(color_counts, key=color_counts.get)

    @classmethod
    def analyze_person_crop(
        cls,
        crop: np.ndarray,
        bbox_area: float,
        frame_area: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """
        Analyzes a person crop for torso/leg clothing colors and visual attributes.

        Returns:
            (dominant_color, upper_color, lower_color, attribute_tags)
        """
        if crop is None or crop.size == 0:
            return None, None, None, []

        h, w = crop.shape[:2]
        tags: List[str] = ["person"]

        # Size / distance estimate based on relative frame area
        rel_area = (bbox_area / max(frame_area, 1.0)) * 100.0
        if rel_area > 15.0:
            tags.append("size:near")
        elif rel_area > 3.0:
            tags.append("size:mid")
        else:
            tags.append("size:far")

        # Aspect ratio / pose hint
        aspect = h / max(w, 1.0)
        if aspect >= 2.0:
            tags.append("pose:standing")
        elif aspect < 1.2:
            tags.append("pose:sitting_or_crouching")

        # Overall dominant color
        dominant = cls.get_dominant_color(crop)
        if dominant != "unknown":
            tags.append(f"color:{dominant}")

        # Segment upper body (torso ~ 15% to 55% height)
        upper_start = int(h * 0.15)
        upper_end = int(h * 0.55)
        upper_crop = crop[upper_start:upper_end, :]

        # Segment lower body (legs/skirt ~ 55% to 90% height)
        lower_start = int(h * 0.55)
        lower_end = int(h * 0.90)
        lower_crop = crop[lower_start:lower_end, :]

        upper_color = cls.get_dominant_color(upper_crop) if upper_crop.size > 0 else None
        lower_color = cls.get_dominant_color(lower_crop) if lower_crop.size > 0 else None

        if upper_color and upper_color != "unknown":
            tags.append(f"upper:{upper_color}")
        if lower_color and lower_color != "unknown":
            tags.append(f"lower:{lower_color}")

        return dominant, upper_color, lower_color, tags


# =====================================================================
# Database Layer (SQLite with Relational Indexing)
# =====================================================================

class DetectionDatabase:
    """
    High-performance relational SQLite database wrapper for surveillance detections.
    Supports structured indexing for camera, timestamp, class, and attribute filtering.
    """

    def __init__(self, db_path: Union[str, Path] = "surveillance_detections.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Create detection table and search indices."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    frame_path TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    is_person INTEGER NOT NULL DEFAULT 0,
                    crop_path TEXT,
                    dominant_color TEXT,
                    upper_color TEXT,
                    lower_color TEXT,
                    attribute_tags TEXT,       -- JSON string of tags list
                    extra_metadata TEXT,       -- JSON string of extra dict
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Relational Indices for fast queries
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_cam_time ON detections(camera_id, timestamp);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class_name);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_person ON detections(is_person);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_dom_color ON detections(dominant_color);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_upper_color ON detections(upper_color);"
            )
            conn.commit()

    def insert_detection(self, rec: DetectionRecord) -> int:
        """Insert a single detection record and return its row ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO detections (
                    camera_id, timestamp, frame_path, class_name, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, is_person,
                    crop_path, dominant_color, upper_color, lower_color,
                    attribute_tags, extra_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.camera_id,
                    rec.timestamp,
                    rec.frame_path,
                    rec.class_name,
                    rec.confidence,
                    rec.bbox[0],
                    rec.bbox[1],
                    rec.bbox[2],
                    rec.bbox[3],
                    1 if rec.is_person else 0,
                    rec.crop_path,
                    rec.dominant_color,
                    rec.upper_color,
                    rec.lower_color,
                    json.dumps(rec.attribute_tags),
                    json.dumps(rec.extra_metadata),
                ),
            )
            conn.commit()
            rec.id = cursor.lastrowid
            return cursor.lastrowid

    def insert_batch(self, records: Sequence[DetectionRecord]) -> int:
        """Batch insert multiple detections within a single transaction."""
        if not records:
            return 0

        rows = [
            (
                r.camera_id,
                r.timestamp,
                r.frame_path,
                r.class_name,
                r.confidence,
                r.bbox[0],
                r.bbox[1],
                r.bbox[2],
                r.bbox[3],
                1 if r.is_person else 0,
                r.crop_path,
                r.dominant_color,
                r.upper_color,
                r.lower_color,
                json.dumps(r.attribute_tags),
                json.dumps(r.extra_metadata),
            )
            for r in records
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO detections (
                    camera_id, timestamp, frame_path, class_name, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, is_person,
                    crop_path, dominant_color, upper_color, lower_color,
                    attribute_tags, extra_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return cursor.rowcount

    def query_detections(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        is_person: Optional[bool] = None,
        color: Optional[str] = None,
        tag: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fast relational filtering across multiple dimensions.
        """
        conditions = []
        params = []

        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        if class_name is not None:
            conditions.append("class_name = ?")
            params.append(class_name)

        if is_person is not None:
            conditions.append("is_person = ?")
            params.append(1 if is_person else 0)

        if color is not None:
            conditions.append("(dominant_color = ? OR upper_color = ? OR lower_color = ?)")
            params.extend([color, color, color])

        if tag is not None:
            # Query JSON attribute tags array
            conditions.append("attribute_tags LIKE ?")
            params.append(f"%{tag}%")

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM detections
            {where_clause}
            ORDER BY timestamp ASC, confidence DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                item = dict(row)
                if item.get("attribute_tags"):
                    try:
                        item["attribute_tags"] = json.loads(item["attribute_tags"])
                    except Exception:
                        pass
                if item.get("extra_metadata"):
                    try:
                        item["extra_metadata"] = json.loads(item["extra_metadata"])
                    except Exception:
                        pass
                results.append(item)
            return results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Aggregate summary metrics across detections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM detections;")
            total_detections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM detections WHERE is_person = 1;")
            total_people = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT camera_id) FROM detections;")
            unique_cameras = cursor.fetchone()[0]

            cursor.execute(
                "SELECT class_name, COUNT(*) as cnt FROM detections GROUP BY class_name ORDER BY cnt DESC LIMIT 10;"
            )
            top_classes = {row["class_name"]: row["cnt"] for row in cursor.fetchall()}

            return {
                "total_detections": total_detections,
                "total_people": total_people,
                "unique_cameras": unique_cameras,
                "top_classes": top_classes,
            }


# =====================================================================
# Vision Analysis Service
# =====================================================================

class VisionAnalysisService:
    """
    High-level Vision Processing module integrating Ultralytics YOLO with
    person bounding-box cropping, attribute tagging, and database indexing.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        target_classes: Optional[Sequence[str]] = None,
        save_person_crops: bool = True,
        crops_dir: Union[str, Path] = "output/crops",
        db_path: Union[str, Path] = "surveillance_detections.db",
        device: Optional[str] = None,
        custom_attribute_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Args:
            model_name_or_path: YOLO model weights (e.g. 'yolov8n.pt', 'yolo11n.pt', etc.).
            confidence_threshold: Minimum detection confidence score.
            target_classes: Optional list of class names to detect (e.g., ['person', 'car']).
                            If None, detects all classes.
            save_person_crops: Whether to crop and save detected people to disk.
            crops_dir: Folder path where person crops will be stored.
            db_path: Path to SQLite database file.
            device: Computing device ('cpu', 'cuda', '0', etc.).
            custom_attribute_fn: Optional hook for custom ML models (e.g., CLIP / ReID).
        """
        self.conf_threshold = confidence_threshold
        self.target_classes = set(target_classes) if target_classes else None
        self.save_person_crops = save_person_crops
        self.crops_dir = Path(crops_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.custom_attribute_fn = custom_attribute_fn

        # Load YOLO model
        logger.info(f"Loading Ultralytics YOLO model '{model_name_or_path}'...")
        self.model = YOLO(model_name_or_path)
        self.device = device

        # Initialize SQLite database
        self.db = DetectionDatabase(db_path=db_path)

    def analyze_single_frame(
        self,
        frame_input: Union[str, Path, np.ndarray] = None,
        image_source: Union[str, Path, np.ndarray] = None,
        camera_id: str = "camera_default",
        timestamp: float = 0.0,
        frame_path: Optional[str] = None,
        frame_path_ref: Optional[str] = None,
        save_to_db: bool = True,
    ) -> List[DetectionRecord]:
        """
        Alias for analyze_image for real-time frame streaming compatibility.
        """
        src = frame_input if frame_input is not None else image_source
        fref = frame_path if frame_path is not None else frame_path_ref
        return self.analyze_image(
            image_source=src,
            camera_id=camera_id,
            timestamp=timestamp,
            frame_path_ref=fref,
            save_to_db=save_to_db,
        )

    def analyze_image(
        self,
        image_source: Union[str, Path, np.ndarray],
        camera_id: str = "camera_default",
        timestamp: float = 0.0,
        frame_path_ref: Optional[str] = None,
        save_to_db: bool = True,
    ) -> List[DetectionRecord]:
        """
        Analyze a single image or frame.

        Args:
            image_source: Image file path or decoded BGR numpy array.
            camera_id: Identifier for the source camera.
            timestamp: Video timestamp in seconds.
            frame_path_ref: Reference path string to the source frame.
            save_to_db: Whether to immediately commit detections to the database.

        Returns:
            List of DetectionRecord objects.
        """
        # Load image array if given a path
        if isinstance(image_source, (str, Path)):
            img_path = Path(image_source)
            frame_ref = frame_path_ref or str(img_path)
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.error(f"Failed to read image at: {img_path}")
                return []
        else:
            frame = image_source
            frame_ref = frame_path_ref or f"memory_frame_{timestamp:.3f}s"

        img_h, img_w = frame.shape[:2]
        frame_area = float(img_h * img_w)

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionRecord] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        crop_counter = 0
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            class_name = first_res.names.get(cls_idx, f"class_{cls_idx}")
            conf = float(box.conf[0].item())

            # Filter by target classes if configured
            if self.target_classes and class_name not in self.target_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bbox = (x1, y1, x2, y2)
            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))

            is_person = (class_name.lower() == "person")
            crop_path: Optional[str] = None
            dominant_color: Optional[str] = None
            upper_color: Optional[str] = None
            lower_color: Optional[str] = None
            attribute_tags: List[str] = [class_name]

            # Bounding box clamping
            ix1 = max(0, min(img_w - 1, int(x1)))
            iy1 = max(0, min(img_h - 1, int(y1)))
            ix2 = max(0, min(img_w, int(x2)))
            iy2 = max(0, min(img_h, int(y2)))

            crop_w = ix2 - ix1
            crop_h = iy2 - iy1

            if crop_w > 5 and crop_h > 5:
                crop = frame[iy1:iy2, ix1:ix2]

                if is_person:
                    # Detailed color and attribute extraction for person
                    (
                        dominant_color,
                        upper_color,
                        lower_color,
                        person_tags,
                    ) = ColorAttributeExtractor.analyze_person_crop(
                        crop=crop,
                        bbox_area=bbox_area,
                        frame_area=frame_area,
                    )
                    attribute_tags = person_tags

                    # Save crop if requested
                    if self.save_person_crops:
                        crop_counter += 1
                        crop_path = self._persist_crop(
                            crop=crop,
                            camera_id=camera_id,
                            timestamp=timestamp,
                            idx=crop_counter,
                        )
                else:
                    # General object dominant color extraction
                    dominant_color = ColorAttributeExtractor.get_dominant_color(crop)
                    if dominant_color and dominant_color != "unknown":
                        attribute_tags.append(f"color:{dominant_color}")

                # Optional custom attribute model hook (e.g. CLIP / LLM / ReID)
                extra_meta = {}
                if self.custom_attribute_fn:
                    try:
                        hook_res = self.custom_attribute_fn(
                            crop,
                            {
                                "class_name": class_name,
                                "camera_id": camera_id,
                                "timestamp": timestamp,
                                "bbox": bbox,
                            },
                        )
                        if isinstance(hook_res, dict):
                            extra_meta.update(hook_res)
                            if "tags" in hook_res and isinstance(hook_res["tags"], list):
                                attribute_tags.extend(hook_res["tags"])
                    except Exception as err:
                        logger.error(f"Error executing custom_attribute_fn: {err}")

            record = DetectionRecord(
                camera_id=camera_id,
                timestamp=timestamp,
                frame_path=frame_ref,
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                is_person=is_person,
                crop_path=crop_path,
                dominant_color=dominant_color,
                upper_color=upper_color,
                lower_color=lower_color,
                attribute_tags=list(set(attribute_tags)),
                extra_metadata=extra_meta if 'extra_meta' in locals() else {},
            )
            detections.append(record)

        if save_to_db and detections:
            self.db.insert_batch(detections)

        return detections

    def analyze_ingested_frames(
        self,
        frame_records: Sequence[Dict[str, Any]],
        camera_id_override: Optional[str] = None,
        save_to_db: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DetectionRecord]:
        """
        Analyzes the dictionary array of frame records output by VideoIngestionService.

        Args:
            frame_records: Array of dicts [{"frame_path": ..., "timestamp": ..., "source_id": ...}]
            camera_id_override: Optional camera ID override.
            save_to_db: Whether to commit detections to the database.
            progress_callback: Optional callable(current_index, total_frames).

        Returns:
            Flat list of all DetectionRecord instances across the analyzed frames.
        """
        all_detections: List[DetectionRecord] = []
        total = len(frame_records)

        logger.info(f"Analyzing {total} ingested frames with YOLO...")

        for idx, item in enumerate(frame_records):
            frame_path = item.get("frame_path")
            timestamp = item.get("timestamp", 0.0)
            cam_id = camera_id_override or item.get("source_id", "camera_default")

            if not frame_path or not Path(frame_path).exists():
                logger.warning(f"Skipping missing frame path: {frame_path}")
                continue

            frame_dets = self.analyze_image(
                image_source=frame_path,
                camera_id=cam_id,
                timestamp=timestamp,
                frame_path_ref=frame_path,
                save_to_db=save_to_db,
            )
            all_detections.extend(frame_dets)

            if progress_callback:
                progress_callback(idx + 1, total)

        logger.info(f"Analysis complete. Total detections found and indexed: {len(all_detections)}")
        return all_detections

    def query_detections(
        self,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        is_person: Optional[bool] = None,
        color: Optional[str] = None,
        tag: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Convenience method delegating to the relational database query engine."""
        return self.db.query_detections(
            camera_id=camera_id,
            class_name=class_name,
            is_person=is_person,
            color=color,
            tag=tag,
            min_confidence=min_confidence,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def _persist_crop(
        self,
        crop: np.ndarray,
        camera_id: str,
        timestamp: float,
        idx: int,
    ) -> Optional[str]:
        """Saves a cropped person bounding box to disk in organized folders."""
        try:
            cam_clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_id)
            cam_crop_dir = self.crops_dir / cam_clean
            cam_crop_dir.mkdir(parents=True, exist_ok=True)

            filename = f"crop_person_{timestamp:08.3f}s_{idx:02d}.jpg"
            dest = cam_crop_dir / filename
            success = cv2.imwrite(str(dest), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return str(dest) if success else None
        except Exception as exc:
            logger.error(f"Failed to persist crop image: {exc}")
            return None
