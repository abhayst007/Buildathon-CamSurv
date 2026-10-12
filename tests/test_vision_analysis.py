"""
Test Suite for VisionAnalysisService
====================================
Tests:
1. ColorAttributeExtractor classification and person crop decomposition.
2. DetectionDatabase schema creation, insertion, and relational queries.
3. VisionAnalysisService YOLO inference and person cropping on synthetic frames.
4. End-to-end integration with VideoIngestionService on real video frames.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from api.video_ingestion import VideoIngestionService, IngestionConfig, SamplingMode
from api.vision_analysis import (
    VisionAnalysisService,
    DetectionDatabase,
    DetectionRecord,
    ColorAttributeExtractor,
)

SAMPLE_VIDEO = PROJECT_ROOT / "sources" / "shop_street.mp4"


class TestVisionAnalysisService(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_vision_"))
        self.db_path = self.test_dir / "test_detections.db"
        self.crops_dir = self.test_dir / "crops"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_color_attribute_extractor(self):
        """Test color classification and person crop analysis."""
        # Create a synthetic image with blue top half, black bottom half
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        # Blue in BGR is (255, 0, 0)
        img[:50, :] = [255, 0, 0]
        # Black in BGR is (0, 0, 0)
        img[50:, :] = [0, 0, 0]

        dom_color = ColorAttributeExtractor.get_dominant_color(img)
        self.assertIn(dom_color, ["blue", "black"])

        (
            dom,
            upper,
            lower,
            tags,
        ) = ColorAttributeExtractor.analyze_person_crop(
            crop=img,
            bbox_area=6000.0,
            frame_area=100000.0,
        )
        self.assertEqual(upper, "blue")
        self.assertEqual(lower, "black")
        self.assertIn("person", tags)
        self.assertIn("upper:blue", tags)
        self.assertIn("lower:black", tags)

    def test_database_insert_and_relational_queries(self):
        """Test SQLite relational database schema, indices, and multi-field queries."""
        db = DetectionDatabase(db_path=self.db_path)

        rec1 = DetectionRecord(
            camera_id="cam_gate_1",
            timestamp=2.5,
            frame_path="frame_01.jpg",
            class_name="person",
            confidence=0.91,
            bbox=(50, 50, 150, 250),
            is_person=True,
            crop_path="crop_01.jpg",
            dominant_color="blue",
            upper_color="blue",
            lower_color="black",
            attribute_tags=["person", "upper:blue", "lower:black", "pose:standing"],
        )

        rec2 = DetectionRecord(
            camera_id="cam_gate_2",
            timestamp=4.0,
            frame_path="frame_02.jpg",
            class_name="car",
            confidence=0.85,
            bbox=(100, 100, 300, 250),
            is_person=False,
            crop_path=None,
            dominant_color="red",
            upper_color=None,
            lower_color=None,
            attribute_tags=["car", "color:red"],
        )

        inserted_count = db.insert_batch([rec1, rec2])
        self.assertEqual(inserted_count, 2)

        # 1. Query by camera_id
        res_cam = db.query_detections(camera_id="cam_gate_1")
        self.assertEqual(len(res_cam), 1)
        self.assertEqual(res_cam[0]["class_name"], "person")

        # 2. Query by class_name
        res_car = db.query_detections(class_name="car")
        self.assertEqual(len(res_car), 1)
        self.assertEqual(res_car[0]["camera_id"], "cam_gate_2")

        # 3. Query by color (relational check on dominant, upper, lower)
        res_blue = db.query_detections(color="blue")
        self.assertEqual(len(res_blue), 1)
        self.assertEqual(res_blue[0]["upper_color"], "blue")

        # 4. Query by tag
        res_standing = db.query_detections(tag="pose:standing")
        self.assertEqual(len(res_standing), 1)

        # 5. Query by timestamp range
        res_time = db.query_detections(start_time=3.0, end_time=5.0)
        self.assertEqual(len(res_time), 1)
        self.assertEqual(res_time[0]["class_name"], "car")

        # 6. Summary stats
        stats = db.get_summary_stats()
        self.assertEqual(stats["total_detections"], 2)
        self.assertEqual(stats["total_people"], 1)
        self.assertEqual(stats["unique_cameras"], 2)

    def test_vision_analysis_with_synthetic_frame(self):
        """Test YOLO prediction on synthetic frame and crop saving."""
        service = VisionAnalysisService(
            model_name_or_path="yolov8n.pt",
            crops_dir=self.crops_dir,
            db_path=self.db_path,
            confidence_threshold=0.2,
        )

        # Create a simple synthetic image
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 120
        dets = service.analyze_image(
            image_source=frame,
            camera_id="test_cam",
            timestamp=1.0,
            save_to_db=True,
        )

        # Should execute cleanly without error
        self.assertIsInstance(dets, list)

    def test_end_to_end_integration_with_sample_video(self):
        """Test full pipeline: VideoIngestionService -> VisionAnalysisService -> SQLite."""
        # 1. Ingest 2 frames from sample video
        ingestion_service = VideoIngestionService()
        frames_dir = self.test_dir / "ingested_frames"
        config = IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=1.0,
            max_frames=2,
            end_time_sec=2.0,
        )

        frame_records = ingestion_service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=frames_dir,
            config=config,
            source_id="street_cam_1",
        )

        self.assertGreaterEqual(len(frame_records), 1)

        # 2. Analyze extracted frames with VisionAnalysisService
        vision_service = VisionAnalysisService(
            model_name_or_path="yolov8n.pt",
            crops_dir=self.crops_dir,
            db_path=self.db_path,
            confidence_threshold=0.25,
            save_person_crops=True,
        )

        detections = vision_service.analyze_ingested_frames(
            frame_records=frame_records,
            save_to_db=True,
        )

        # Verify detections were found (the street video has people and cars)
        self.assertGreater(len(detections), 0)

        # Check database records
        db_records = vision_service.query_detections(camera_id="street_cam_1")
        self.assertEqual(len(db_records), len(detections))

        # Check that people detected have crops saved to disk
        people_dets = [d for d in detections if d.is_person]
        if people_dets:
            first_person = people_dets[0]
            self.assertIsNotNone(first_person.crop_path)
            self.assertTrue(Path(first_person.crop_path).exists())
            self.assertGreater(len(first_person.attribute_tags), 0)


if __name__ == "__main__":
    unittest.main()
