"""
Test Suite for VideoIngestionService
====================================
Tests:
1. Basic single video ingestion with Fixed FPS.
2. Interval-based sampling (every N frames).
3. Dynamic Adaptive motion sampling.
4. Custom dynamic callback sampling.
5. Batch processing of videos.
6. Multi-camera processing with simulated multi-camera feeds.
7. Graceful handling of End-Of-File (EOF).
8. Graceful handling of missing files / corrupted sources.
9. Verify dictionary array output format with frame file paths and timestamps.
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

from api.video_ingestion import (
    VideoIngestionService,
    IngestionConfig,
    SamplingMode,
    FrameRecord,
)

SAMPLE_VIDEO = Path("sources/shop_street.mp4").resolve()


class TestVideoIngestionService(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_ingestion_"))
        self.service = VideoIngestionService()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fixed_fps_sampling(self):
        """Test fixed FPS sampling (e.g., 0.5 FPS over first 4 seconds)."""
        output_dir = self.test_dir / "fixed_fps"
        config = IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=0.5,
            end_time_sec=4.0,  # Process first 4 seconds
            image_format="jpg",
        )

        results = self.service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=output_dir,
            config=config,
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for record in results:
            self.assertIn("frame_path", record)
            self.assertIn("timestamp", record)
            self.assertIn("frame_index", record)
            self.assertIn("source_id", record)
            self.assertTrue(Path(record["frame_path"]).exists())
            self.assertGreaterEqual(record["timestamp"], 0.0)
            self.assertLessEqual(record["timestamp"], 5.0)

        # Check that consecutive timestamps are roughly 2 seconds apart (0.5 FPS)
        if len(results) > 1:
            diff = results[1]["timestamp"] - results[0]["timestamp"]
            self.assertAlmostEqual(diff, 2.0, delta=0.5)

    def test_every_n_frames_sampling(self):
        """Test every N frames sampling mode."""
        output_dir = self.test_dir / "every_n"
        config = IngestionConfig(
            sampling_mode=SamplingMode.EVERY_N_FRAMES,
            frame_interval=30,  # ~1 frame per second at 30fps
            max_frames=3,
        )

        results = self.service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=output_dir,
            config=config,
        )

        self.assertEqual(len(results), 3)
        for record in results:
            self.assertTrue(Path(record["frame_path"]).exists())
            self.assertEqual(record["frame_index"] % 30, 0)

    def test_adaptive_motion_sampling(self):
        """Test dynamic adaptive motion sampling."""
        output_dir = self.test_dir / "adaptive"
        config = IngestionConfig(
            sampling_mode=SamplingMode.DYNAMIC_ADAPTIVE,
            min_fps=0.5,
            max_fps=3.0,
            motion_threshold=0.02,
            end_time_sec=3.0,
        )

        results = self.service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=output_dir,
            config=config,
        )

        self.assertGreater(len(results), 0)
        for record in results:
            self.assertTrue(Path(record["frame_path"]).exists())
            self.assertIn("motion_score", record)
            self.assertIsNotNone(record["motion_score"])

    def test_dynamic_callback_sampling(self):
        """Test dynamic sampling rate via user-defined callback."""
        output_dir = self.test_dir / "dynamic_cb"

        # Rate increases over time
        def dynamic_rate(t: float, idx: int, frame: np.ndarray) -> float:
            return 0.5 if t < 2.0 else 2.0

        config = IngestionConfig(
            sampling_mode=SamplingMode.DYNAMIC_CALLBACK,
            dynamic_rate_fn=dynamic_rate,
            end_time_sec=4.0,
        )

        results = self.service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=output_dir,
            config=config,
        )

        self.assertGreater(len(results), 0)
        for record in results:
            self.assertTrue(Path(record["frame_path"]).exists())

    def test_multi_camera_processing(self):
        """Test concurrent multi-camera processing."""
        output_dir = self.test_dir / "multicam"
        cameras = {
            "cam_entrance": str(SAMPLE_VIDEO),
            "cam_parking": str(SAMPLE_VIDEO),
        }

        config = IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=1.0,
            max_frames=2,
        )

        results = self.service.process_multi_camera(
            camera_sources=cameras,
            output_folder=output_dir,
            config=config,
            max_workers=2,
        )

        self.assertIn("cam_entrance", results)
        self.assertIn("cam_parking", results)
        self.assertEqual(len(results["cam_entrance"]), 2)
        self.assertEqual(len(results["cam_parking"]), 2)

        # Check subdirectories
        self.assertTrue((output_dir / "cam_entrance").is_dir())
        self.assertTrue((output_dir / "cam_parking").is_dir())

    def test_batch_processing(self):
        """Test batch processing of multiple videos."""
        output_dir = self.test_dir / "batch"
        sources = [str(SAMPLE_VIDEO), str(SAMPLE_VIDEO)]

        config = IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=1.0,
            max_frames=2,
        )

        results = self.service.process_batch(
            video_sources=sources,
            output_folder=output_dir,
            config=config,
            max_workers=2,
        )

        self.assertGreater(len(results), 0)

    def test_corrupted_or_nonexistent_source(self):
        """Test graceful failure when video source does not exist."""
        output_dir = self.test_dir / "nonexistent"
        config = IngestionConfig(stream_reconnect_attempts=1, reconnect_delay_sec=0.1)

        results = self.service.process_video(
            video_source="invalid/nonexistent/path.mp4",
            output_folder=output_dir,
            config=config,
        )

        # Should return an empty list without crashing
        self.assertEqual(results, [])

    def test_resize_and_image_formatting(self):
        """Test frame resizing and PNG formatting."""
        output_dir = self.test_dir / "resize"
        config = IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=1.0,
            max_frames=1,
            image_format="png",
            resize_output=(640, 360),
        )

        results = self.service.process_video(
            video_source=str(SAMPLE_VIDEO),
            output_folder=output_dir,
            config=config,
        )

        self.assertEqual(len(results), 1)
        frame_path = results[0]["frame_path"]
        self.assertTrue(frame_path.endswith(".png"))
        img = cv2.imread(frame_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[1], 640)
        self.assertEqual(img.shape[0], 360)

    def test_corrupted_frame_stream_resilience(self):
        """Test that corrupted frames or sudden empty frames do not crash the service."""
        from unittest.mock import MagicMock, patch

        output_dir = self.test_dir / "corrupted_stream"
        config = IngestionConfig(
            max_consecutive_errors=5,
            target_fps=1.0,
            sampling_mode=SamplingMode.FIXED_FPS,
        )

        # Mock VideoCapture to return 2 valid frames, then 10 corrupted (None) frames
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 100.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
            cv2.CAP_PROP_POS_FRAMES: 0.0,
        }.get(prop, 0.0)

        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # 2 valid frames, then consecutive None/corrupted frames
        mock_cap.read.side_effect = [
            (True, valid_frame),
            (True, valid_frame),
            (False, None),
            (False, None),
            (False, None),
            (False, None),
            (False, None),
            (False, None),
        ]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            results = self.service.process_video(
                video_source="rtsp://dummy/stream",
                output_folder=output_dir,
                config=config,
            )

        # Should have captured the 2 valid frames and gracefully terminated on errors
        self.assertGreaterEqual(len(results), 1)
        for r in results:
            self.assertTrue(Path(r["frame_path"]).exists())

    def test_mapping_helpers(self):
        """Test to_timestamp_map and to_frame_path_map static converters."""
        dummy_records = [
            {"frame_path": "frame_1.jpg", "timestamp": 1.25},
            {"frame_path": "frame_2.jpg", "timestamp": 2.50},
        ]
        ts_map = VideoIngestionService.to_timestamp_map(dummy_records)
        self.assertEqual(ts_map, {1.25: "frame_1.jpg", 2.50: "frame_2.jpg"})

        fp_map = VideoIngestionService.to_frame_path_map(dummy_records)
        self.assertEqual(fp_map, {"frame_1.jpg": 1.25, "frame_2.jpg": 2.50})


if __name__ == "__main__":
    unittest.main()


