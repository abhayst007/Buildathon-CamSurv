"""
Complete Surveillance Pipeline Demo
===================================
1. Ingests video frames from `sources/shop_street.mp4` via VideoIngestionService.
2. Analyzes frames using Ultralytics YOLO via VisionAnalysisService.
3. Detects people & objects, crops person bounding boxes, and extracts colors & attributes.
4. Persists structured detection records to a local SQLite database table.
5. Performs fast relational filtering queries (by camera, class, color, and timestamp).
"""

import json
import sys
from pathlib import Path
from pprint import pprint

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.video_ingestion import VideoIngestionService, IngestionConfig, SamplingMode
from api.vision_analysis import VisionAnalysisService

def run_pipeline():
    video_path = PROJECT_ROOT / "sources" / "shop_street.mp4"
    output_dir = PROJECT_ROOT / "output" / "pipeline_demo"
    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"
    db_path = output_dir / "detections.db"

    print("=" * 65)
    print("STEP 1: Dynamic Video Ingestion (VideoIngestionService)")
    print("=" * 65)

    ingestion_service = VideoIngestionService()
    ingest_config = IngestionConfig(
        sampling_mode=SamplingMode.FIXED_FPS,
        target_fps=1.0,         # 1 frame per second
        end_time_sec=4.0,       # Process first 4 seconds for demo
    )

    frame_records = ingestion_service.process_video(
        video_source=str(video_path),
        output_folder=frames_dir,
        source_id="camera_street_01",
        config=ingest_config,
    )

    print(f"Ingested {len(frame_records)} frames from '{video_path.name}'.")

    print("\n" + "=" * 65)
    print("STEP 2: Vision Analysis with YOLO & Attribute Extraction")
    print("=" * 65)

    vision_service = VisionAnalysisService(
        model_name_or_path="yolov8n.pt",
        confidence_threshold=0.30,
        save_person_crops=True,
        crops_dir=crops_dir,
        db_path=db_path,
    )

    detections = vision_service.analyze_ingested_frames(
        frame_records=frame_records,
        save_to_db=True,
    )

    print(f"Extracted {len(detections)} total object detections across frames.")

    print("\n" + "=" * 65)
    print("STEP 3: Fast Relational Database Queries (SQLite)")
    print("=" * 65)

    # 1. Summary statistics
    stats = vision_service.db.get_summary_stats()
    print("\nDatabase Summary Stats:")
    print(json.dumps(stats, indent=2))

    # 2. Query all detected people
    people = vision_service.query_detections(
        camera_id="camera_street_01",
        is_person=True,
        limit=5,
    )
    print(f"\nSample Person Detections (showing first {len(people)}):")
    for p in people:
        print(f"- Time: {p['timestamp']:.2f}s | Conf: {p['confidence']:.2f} | "
              f"Upper: {p['upper_color']} | Lower: {p['lower_color']} | "
              f"Tags: {p['attribute_tags']}")
        if p.get("crop_path"):
            print(f"  Crop Saved: {p['crop_path']}")

    # 3. Query specific clothing / object color
    color_matches = vision_service.query_detections(
        color="black",
        limit=5,
    )
    print(f"\nDetections with 'black' (dominant/upper/lower): {len(color_matches)} matches.")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()
