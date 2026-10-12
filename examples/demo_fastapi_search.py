"""
FastAPI Natural Language Surveillance Search Demo
==================================================
Demonstrates:
1. Ingesting sample frames from `sources/shop_street.mp4`.
2. Indexing them into ChromaDB via the FastAPI client.
3. Querying the FastAPI endpoint with natural language prompts:
   - "people walking on the sidewalk"
   - "black car or vehicle"
   - "person wearing dark clothing"
4. Displaying matching frame paths, timestamps, camera IDs, and similarity scores.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.app import app
from api.video_ingestion import VideoIngestionService, IngestionConfig, SamplingMode

def run_demo():
    client = TestClient(app)
    video_path = PROJECT_ROOT / "sources" / "shop_street.mp4"
    frames_dir = PROJECT_ROOT / "output" / "api_demo_frames"

    print("=" * 65)
    print("STEP 1: Ingesting video frames for indexing")
    print("=" * 65)

    ingestion_service = VideoIngestionService()
    frame_records = ingestion_service.process_video(
        video_source=str(video_path),
        output_folder=frames_dir,
        source_id="camera_street_01",
        config=IngestionConfig(
            sampling_mode=SamplingMode.FIXED_FPS,
            target_fps=1.0,
            end_time_sec=4.0,
        ),
    )
    print(f"Ingested {len(frame_records)} frames from '{video_path.name}'.")

    print("\n" + "=" * 65)
    print("STEP 2: Indexing frames into ChromaDB (POST /index)")
    print("=" * 65)

    index_response = client.post("/index", json={"frames": frame_records})
    print("Index Response:", index_response.json())

    print("\n" + "=" * 65)
    print("STEP 3: Natural Language Queries (POST /search & GET /search)")
    print("=" * 65)

    queries = [
        "people walking on the street sidewalk",
        "person wearing dark jacket or clothing",
        "car or vehicle on the road",
    ]

    for q in queries:
        print(f"\n---> Searching: '{q}'")
        res = client.post(
            "/search",
            json={
                "query": q,
                "top_k": 3,
                "camera_id": "camera_street_01",
            },
        )
        data = res.json()
        print(f"Total Matches: {data['total_matches']}")
        for match in data["results"]:
            print(f"  * Similarity: {match['similarity_score']:.4f} | "
                  f"Time: {match['timestamp']}s | "
                  f"Cam: {match['camera_id']} | "
                  f"Path: {match['frame_path']}")

    print("\n" + "=" * 65)
    print("STEP 4: Health Check (GET /health)")
    print("=" * 65)
    health = client.get("/health").json()
    print(json.dumps(health, indent=2))

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
