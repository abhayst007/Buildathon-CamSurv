"""
VideoIngestionService Demonstration
===================================
Demonstrates:
- Single video dynamic adaptive frame sampling on `sources/shop_street.mp4`
- Multi-camera parallel processing
- Output verification of dictionary array [{frame_path, timestamp, ...}]
"""

import sys
import json
from pathlib import Path
from pprint import pprint

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.video_ingestion import (
    VideoIngestionService,
    IngestionConfig,
    SamplingMode,
)

def run_demo():
    video_path = Path("sources/shop_street.mp4").resolve()
    output_dir = Path("output/demo_frames").resolve()

    if not video_path.exists():
        print(f"Sample video not found at: {video_path}")
        return

    print("=" * 60)
    print("1. Single Video Ingestion with Dynamic Adaptive Motion Sampling")
    print("=" * 60)

    service = VideoIngestionService()
    adaptive_config = IngestionConfig(
        sampling_mode=SamplingMode.DYNAMIC_ADAPTIVE,
        min_fps=0.5,             # 1 frame every 2 seconds during low motion
        max_fps=3.0,             # Up to 3 frames per second during high motion
        motion_threshold=0.03,   # Threshold for sensitivity
        end_time_sec=5.0,        # Sample first 5 seconds for demonstration
        image_format="jpg",
        jpeg_quality=95,
    )

    frame_records = service.process_video(
        video_source=str(video_path),
        output_folder=output_dir / "single_adaptive",
        config=adaptive_config,
    )

    print(f"\nExtracted {len(frame_records)} frames:")
    print(json.dumps(frame_records, indent=2))

    print("\n" + "=" * 60)
    print("2. Multi-Camera Concurrent Ingestion")
    print("=" * 60)

    multicam_config = IngestionConfig(
        sampling_mode=SamplingMode.FIXED_FPS,
        target_fps=1.0,          # 1 frame per second
        end_time_sec=3.0,        # Sample first 3 seconds
    )

    multicam_results = service.process_multi_camera(
        camera_sources={
            "camera_street_north": str(video_path),
            "camera_street_south": str(video_path),
        },
        output_folder=output_dir / "multicam",
        config=multicam_config,
        max_workers=2,
    )

    for cam_id, frames in multicam_results.items():
        print(f"\nCamera: {cam_id} -> {len(frames)} frames extracted.")
        if frames:
            print(f"Sample frame: {frames[0]['frame_path']} @ {frames[0]['timestamp']}s")

    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    run_demo()
