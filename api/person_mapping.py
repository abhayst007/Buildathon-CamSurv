"""
Person & Face Mapping Service
==============================
Detects faces, extracts visual descriptions for individuals in surveillance video feeds,
and maps unique persons into a relational SQLite database.

Guarantees de-duplication:
- Computes normalized CLIP appearance embeddings for detected person and face crops.
- Matches against registered persons using cosine similarity threshold (>= 0.82).
- Ensures the same individual is not mapped more than once, incrementing appearance counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("PersonMappingService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PersonMappingService:
    """
    Manages face detection, appearance description, and unique individual identity mapping.
    """

    SIMILARITY_THRESHOLD = 0.82  # Cosine similarity threshold to consider the same person

    def __init__(
        self,
        db_path: Union[str, Path] = "output/detections.db",
        faces_dir: Union[str, Path] = "output/faces",
    ):
        self.db_path = Path(db_path)
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)

        # Load OpenCV Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning("Could not load OpenCV Haar cascade for face detection.")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes relational tables for mapped persons and their appearances."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS mapped_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT UNIQUE NOT NULL,
                    camera_id TEXT NOT NULL,
                    first_seen_timestamp REAL NOT NULL,
                    last_seen_timestamp REAL NOT NULL,
                    avatar_path TEXT NOT NULL,
                    full_crop_path TEXT,
                    frame_path TEXT NOT NULL,
                    visual_description TEXT NOT NULL,
                    upper_color TEXT,
                    lower_color TEXT,
                    dominant_color TEXT,
                    has_face INTEGER DEFAULT 0,
                    occurrence_count INTEGER DEFAULT 1,
                    embedding_vector TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS person_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    frame_path TEXT NOT NULL,
                    crop_path TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES mapped_persons(person_id)
                );
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_mapped_person_id ON mapped_persons(person_id);"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_person_time ON person_appearances(person_id, timestamp);"
            )
            conn.commit()

    def extract_face_or_avatar(
        self,
        person_crop: np.ndarray,
        person_id_tag: str = "temp",
    ) -> Tuple[Optional[str], bool]:
        """
        Detects frontal face in person crop; falls back to head/upper torso crop.
        Returns:
            (saved_avatar_path, has_detected_face)
        """
        if person_crop is None or person_crop.size == 0:
            return None, False

        h, w = person_crop.shape[:2]
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

        has_face = False
        avatar_img = None

        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(24, 24),
            )
            if len(faces) > 0:
                # Pick largest detected face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                # Add slight margin around face
                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.25)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)
                avatar_img = person_crop[y1:y2, x1:x2]
                has_face = True

        if avatar_img is None or avatar_img.size == 0:
            # Fallback: upper 35% of the person crop (head and shoulders)
            head_h = max(20, int(h * 0.35))
            avatar_img = person_crop[0:head_h, :]
            has_face = False

        # Resize to standard avatar dimensions for uniform display
        avatar_img = cv2.resize(avatar_img, (140, 140), interpolation=cv2.INTER_AREA)

        avatar_filename = f"{person_id_tag}_{int(time.time() * 1000)}.jpg"
        avatar_path = self.faces_dir / avatar_filename
        cv2.imwrite(str(avatar_path), avatar_img)
        return str(avatar_path), has_face

    def generate_visual_description(
        self,
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        dominant_color: Optional[str] = None,
        has_face: bool = False,
    ) -> str:
        """Generates an intuitive descriptive text label based on visual appearance."""
        parts = []
        if has_face:
            parts.append("Individual with mapped facial features")
        else:
            parts.append("Person profile")

        color_desc = []
        if upper_color and upper_color != "unknown":
            color_desc.append(f"{upper_color} upper clothing")
        if lower_color and lower_color != "unknown":
            color_desc.append(f"{lower_color} lower wear")

        if color_desc:
            parts.append("wearing " + " and ".join(color_desc))
        elif dominant_color and dominant_color != "unknown":
            parts.append(f"predominantly {dominant_color} attire")
        else:
            parts.append("standard attire")

        return ", ".join(parts).capitalize()

    def map_or_identify_person(
        self,
        crop_image: np.ndarray,
        frame_path: str,
        camera_id: str,
        timestamp: float,
        embedding: List[float],
        upper_color: Optional[str] = None,
        lower_color: Optional[str] = None,
        dominant_color: Optional[str] = None,
        full_crop_path: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Matches person embedding against registered database to ensure DE-DUPLICATION.
        If similarity >= SIMILARITY_THRESHOLD, maps to existing person.
        Otherwise, registers as a new unique individual.

        Returns:
            (person_id, is_new_individual)
        """
        emb_arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT person_id, embedding_vector, occurrence_count, last_seen_timestamp FROM mapped_persons"
            )
            rows = c.fetchall()

            best_sim = -1.0
            matched_id: Optional[str] = None

            for row in rows:
                if not row["embedding_vector"]:
                    continue
                try:
                    reg_emb = np.array(json.loads(row["embedding_vector"]), dtype=np.float32)
                    reg_norm = np.linalg.norm(reg_emb)
                    if reg_norm > 0:
                        reg_emb = reg_emb / reg_norm
                    sim = float(np.dot(emb_arr, reg_emb))
                    if sim > best_sim:
                        best_sim = sim
                        matched_id = row["person_id"]
                except Exception:
                    continue

            # Check if this person is already registered
            if matched_id is not None and best_sim >= self.SIMILARITY_THRESHOLD:
                # Existing person: update appearance without duplicating!
                c.execute(
                    """
                    UPDATE mapped_persons
                    SET occurrence_count = occurrence_count + 1,
                        last_seen_timestamp = MAX(last_seen_timestamp, ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE person_id = ?
                    """,
                    (timestamp, matched_id),
                )
                c.execute(
                    """
                    INSERT INTO person_appearances (person_id, camera_id, timestamp, frame_path, crop_path, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (matched_id, camera_id, timestamp, frame_path, full_crop_path, round(best_sim, 4)),
                )
                conn.commit()
                return matched_id, False

            # New unique person!
            c.execute("SELECT COUNT(*) FROM mapped_persons")
            next_num = c.fetchone()[0] + 1
            new_person_id = f"PER-{next_num:03d}"

            avatar_path, has_face = self.extract_face_or_avatar(
                crop_image,
                person_id_tag=new_person_id,
            )
            description = self.generate_visual_description(
                upper_color=upper_color,
                lower_color=lower_color,
                dominant_color=dominant_color,
                has_face=has_face,
            )

            c.execute(
                """
                INSERT INTO mapped_persons (
                    person_id, camera_id, first_seen_timestamp, last_seen_timestamp,
                    avatar_path, full_crop_path, frame_path, visual_description,
                    upper_color, lower_color, dominant_color, has_face,
                    occurrence_count, embedding_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    new_person_id,
                    camera_id,
                    timestamp,
                    timestamp,
                    avatar_path or "",
                    full_crop_path or "",
                    frame_path,
                    description,
                    upper_color or "unknown",
                    lower_color or "unknown",
                    dominant_color or "unknown",
                    1 if has_face else 0,
                    json.dumps(emb_arr.tolist()),
                ),
            )
            c.execute(
                """
                INSERT INTO person_appearances (person_id, camera_id, timestamp, frame_path, crop_path, confidence)
                VALUES (?, ?, ?, ?, ?, 1.0)
                """,
                (new_person_id, camera_id, timestamp, frame_path, full_crop_path),
            )
            conn.commit()
            logger.info(f"Mapped new unique individual: {new_person_id} ({description})")
            return new_person_id, True

    def get_all_mapped_persons(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all unique mapped individuals."""
        with self._get_connection() as conn:
            c = conn.cursor()
            if camera_id and camera_id != "all":
                c.execute(
                    "SELECT * FROM mapped_persons WHERE camera_id = ? ORDER BY id ASC",
                    (camera_id,),
                )
            else:
                c.execute("SELECT * FROM mapped_persons ORDER BY id ASC")
            rows = c.fetchall()

            persons = []
            for r in rows:
                persons.append({
                    "id": r["id"],
                    "person_id": r["person_id"],
                    "camera_id": r["camera_id"],
                    "first_seen_timestamp": round(r["first_seen_timestamp"], 3),
                    "last_seen_timestamp": round(r["last_seen_timestamp"], 3),
                    "avatar_path": r["avatar_path"],
                    "full_crop_path": r["full_crop_path"],
                    "frame_path": r["frame_path"],
                    "visual_description": r["visual_description"],
                    "upper_color": r["upper_color"],
                    "lower_color": r["lower_color"],
                    "dominant_color": r["dominant_color"],
                    "has_face": bool(r["has_face"]),
                    "occurrence_count": r["occurrence_count"],
                })
            return persons

    def get_person_frames(self, person_id: str) -> List[Dict[str, Any]]:
        """Returns all appearances/frames for a specific mapped individual."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT DISTINCT frame_path, camera_id, timestamp, crop_path, confidence
                FROM person_appearances
                WHERE person_id = ?
                ORDER BY timestamp ASC
                """,
                (person_id,),
            )
            rows = c.fetchall()
            return [
                {
                    "frame_path": r["frame_path"],
                    "camera_id": r["camera_id"],
                    "timestamp": round(r["timestamp"], 3),
                    "crop_path": r["crop_path"],
                    "confidence": r["confidence"],
                }
                for r in rows
            ]

    def backfill_from_existing_crops(self, clip_service: Any) -> int:
        """
        Scans existing person detections in detections table and indexes unique individuals.
        """
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM mapped_persons")
            if c.fetchone()[0] > 0:
                logger.info("Mapped persons already populated. Skipping backfill.")
                return 0

            c.execute(
                """
                SELECT frame_path, crop_path, camera_id, timestamp, upper_color, lower_color, dominant_color
                FROM detections
                WHERE is_person = 1 AND crop_path IS NOT NULL AND crop_path != ''
                ORDER BY timestamp ASC
                """
            )
            rows = c.fetchall()

        if not rows:
            return 0

        logger.info(f"Backfilling person mapping from {len(rows)} detections...")
        mapped_count = 0
        sample_step = max(1, len(rows) // 40)  # Representative sampling across time
        for i in range(0, len(rows), sample_step):
            r = rows[i]
            crop_p = Path(r["crop_path"])
            if not crop_p.exists():
                continue
            crop_img = cv2.imread(str(crop_p))
            if crop_img is None:
                continue

            emb = clip_service.encode_image(crop_img)
            pid, is_new = self.map_or_identify_person(
                crop_image=crop_img,
                frame_path=r["frame_path"],
                camera_id=r["camera_id"],
                timestamp=r["timestamp"],
                embedding=emb,
                upper_color=r["upper_color"],
                lower_color=r["lower_color"],
                dominant_color=r["dominant_color"],
                full_crop_path=str(crop_p),
            )
            if is_new:
                mapped_count += 1

        logger.info(f"Backfill complete: Registered {mapped_count} unique individuals.")
        return mapped_count
