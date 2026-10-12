"""
Video Ingestion & Indexing Pipeline CLI
=======================================
Allows adding any MP4 video file (or camera stream) to the Surveillance System.

What this does:
1. Ingests video frames at a configurable sampling rate (VideoIngestionService).
2. Detects objects, crops people, and extracts color attributes into SQLite (VisionAnalysisService).
3. Generates CLIP embeddings and indexes frames into ChromaDB (VectorSearchService).
4. Makes the video immediately searchable by natural language in the Next.js dashboard!

Usage:
------
# 1. Ingest a single video file:
python ingest_video.py --source sources/new_feed.mp4 --camera-id cam_hallway --fps 1.0

# 2. Ingest with adaptive motion sampling:
python ingest_video.py --source sources/new_feed.mp4 --camera-id cam_parking --adaptive

# 3. Ingest an entire folder of MP4 videos:
python ingest_video.py --folder sources/ --fps 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.video_ingestion import VideoIngestionService, IngestionConfig, SamplingMode
from api.vision_analysis import VisionAnalysisService
from api.clip_service import CLIPEmbeddingService
from api.vector_service import VectorSearchService


def ingest_video_file(
    video_source: str,
    camera_id: str,
    target_fps: float = 1.0,
    adaptive_motion: bool = False,
    run_yolo: bool = True,
    output_base_dir: str = "output",
) -> int:
    """Ingests a video file, detects objects, crops people, and indexes into ChromaDB."""
    source_path = Path(video_source)
    if not source_path.exists() and not str(video_source).startswith(("rtsp://", "http://")):
        print(f"[ERROR] Video source not found: {source_path.resolve()}", file=sys.stderr)
        return 0

    print("=" * 65)
    print(f"INGESTING VIDEO: {video_source}")
    print(f"CAMERA ID:       {camera_id}")
    print(f"SAMPLING:        {'ADAPTIVE MOTION' if adaptive_motion else f'{target_fps} FPS'}")
    print("=" * 65)

    base_dir = Path(output_base_dir)
    frames_dir = base_dir / "frames" / camera_id
    crops_dir = base_dir / "crops"
    db_path = base_dir / "detections.db"
    chroma_dir = base_dir / "chroma_db"

    # 1. Extract Frames
    print("\n[1/3] Extracting video frames...")
    ingestion_service = VideoIngestionService()
    sampling_mode = SamplingMode.DYNAMIC_ADAPTIVE if adaptive_motion else SamplingMode.FIXED_FPS
    config = IngestionConfig(
        sampling_mode=sampling_mode,
        target_fps=target_fps,
        min_fps=0.5,
        max_fps=3.0,
    )

    t0 = time.time()
    frame_records = ingestion_service.process_video(
        video_source=str(video_source),
        output_folder=frames_dir,
        source_id=camera_id,
        config=config,
    )
    print(f"  -> Extracted {len(frame_records)} frames in {time.time() - t0:.2f}s")

    if not frame_records:
        print("[WARNING] No frames were extracted. Check video codec or permissions.")
        return 0

    # 2. Vision Analysis (YOLO Object Detection & Person Cropping)
    if run_yolo:
        print("\n[2/3] Running YOLO detection & person attribute extraction...")
        t1 = time.time()
        vision_service = VisionAnalysisService(
            model_name_or_path="yolov8n.pt",
            confidence_threshold=0.35,
            save_person_crops=True,
            crops_dir=crops_dir,
            db_path=db_path,
        )
        detections = vision_service.analyze_ingested_frames(
            frame_records=frame_records,
            save_to_db=True,
        )
        print(f"  -> Found {len(detections)} object detections saved to SQLite ({time.time() - t1:.2f}s)")
    else:
        print("\n[2/3] Skipping YOLO analysis as requested.")

    # 3. Compute CLIP Embeddings & Index into ChromaDB
    print("\n[3/3] Generating CLIP embeddings & indexing into ChromaDB...")
    t2 = time.time()
    clip_service = CLIPEmbeddingService.get_instance(lazy_load=False)
    vector_service = VectorSearchService(persist_dir=chroma_dir)

    image_paths = [rec["frame_path"] for rec in frame_records]
    embeddings = clip_service.encode_image_batch(image_paths)

    indexed_count = vector_service.index_frames_batch(
        records=frame_records,
        embeddings=embeddings,
    )
    print(f"  -> Successfully indexed {indexed_count} frames into ChromaDB ({time.time() - t2:.2f}s)")

    print("\n" + "=" * 65)
    print(f"SUCCESS! Camera '{camera_id}' is now searchable in the dashboard.")
    print("=" * 65 + "\n")
    return indexed_count


def main():
    parser = argparse.ArgumentParser(
        description="Ingest, analyze, and index any MP4 video into CamSurveillance."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Path to the MP4 video file or stream URL (e.g. sources/my_video.mp4)",
    )
    parser.add_argument(
        "--camera-id",
        "-c",
        type=str,
        help="Unique camera identifier (e.g. cam_gate_01, front_porch). Defaults to filename.",
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        help="Folder containing multiple MP4 files to batch ingest.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target frame sampling rate in frames per second (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use dynamic adaptive motion sampling instead of fixed FPS.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip YOLO object detection and only compute CLIP embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base directory for frames, crops, SQLite DB, and ChromaDB.",
    )

    args = parser.parse_args()

    if not args.source and not args.folder:
        parser.print_help()
        print("\n[!] Please specify either --source <file.mp4> or --folder <directory>.")
        sys.exit(1)

    if args.source:
        cam_id = args.camera_id or Path(args.source).stem
        ingest_video_file(
            video_source=args.source,
            camera_id=cam_id,
            target_fps=args.fps,
            adaptive_motion=args.adaptive,
            run_yolo=not args.no_yolo,
            output_base_dir=args.output_dir,
        )

    elif args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"[ERROR] Folder not found: {folder_path.resolve()}", file=sys.stderr)
            sys.exit(1)

        video_files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.mkv")) + list(folder_path.glob("*.avi"))
        if not video_files:
            print(f"[WARNING] No video files found in {folder_path.resolve()}.")
            sys.exit(0)

        print(f"Found {len(video_files)} video files in {folder_path.resolve()} to ingest...")
        total_indexed = 0
        for vid in video_files:
            cam_id = vid.stem
            count = ingest_video_file(
                video_source=str(vid),
                camera_id=cam_id,
                target_fps=args.fps,
                adaptive_motion=args.adaptive,
                run_yolo=not args.no_yolo,
                output_base_dir=args.output_dir,
            )
            total_indexed += count

        print(f"\n[DONE] Batch ingestion complete. Total frames indexed: {total_indexed}")


if __name__ == "__main__":
    main()
