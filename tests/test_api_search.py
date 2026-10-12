"""
Test Suite for FastAPI Surveillance Semantic Search Endpoints
=============================================================
Tests:
1. Health check endpoint (GET /health).
2. Indexing frames into ChromaDB (POST /index).
3. Semantic text search via POST /search (e.g. 'person wearing a red t-shirt').
4. Semantic text search via GET /search with query parameters.
5. Filtering by camera_id and min_similarity.
6. Validation errors for empty queries.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api.app as app_module
from api.vector_service import VectorSearchService

class TestFastAPISearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_api_search_"))
        cls.test_chroma = cls.test_dir / "chroma_db"
        cls.orig_vector_service = app_module.vector_service
        cls.isolated_vector_service = VectorSearchService(persist_dir=cls.test_chroma)
        app_module.vector_service = cls.isolated_vector_service

        cls.client = TestClient(app_module.app)

        # Setup test images
        cls.img1_path = cls.test_dir / "red_shirt_person.jpg"
        cls.img2_path = cls.test_dir / "blue_car.jpg"

        # Create synthetic images: red block and blue block
        red_img = np.zeros((224, 224, 3), dtype=np.uint8)
        red_img[:, :] = [0, 0, 255]  # Red in BGR
        cv2.imwrite(str(cls.img1_path), red_img)

        blue_img = np.zeros((224, 224, 3), dtype=np.uint8)
        blue_img[:, :] = [255, 0, 0]  # Blue in BGR
        cv2.imwrite(str(cls.img2_path), blue_img)

        # Index the test images via the /index endpoint
        index_payload = {
            "frames": [
                {
                    "frame_path": str(cls.img1_path),
                    "timestamp": 1.5,
                    "camera_id": "cam_entrance",
                    "description": "person with red shirt",
                },
                {
                    "frame_path": str(cls.img2_path),
                    "timestamp": 3.0,
                    "camera_id": "cam_parking",
                    "description": "blue car in lot",
                },
            ]
        }
        res = cls.client.post("/index", json=index_payload)
        assert res.status_code == 200, f"Indexing failed: {res.text}"

    @classmethod
    def tearDownClass(cls):
        app_module.vector_service = cls.orig_vector_service
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_health_check(self):
        """Test GET /health returns service status and vector collection info."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["vector_db"], "ChromaDB")
        self.assertGreaterEqual(data["indexed_frames_count"], 2)

    def test_post_search_red_shirt(self):
        """Test POST /search for query 'person wearing a red t-shirt'."""
        payload = {
            "query": "person wearing a red t-shirt",
            "top_k": 2,
        }
        res = self.client.post("/search", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["query"], "person wearing a red t-shirt")
        self.assertGreater(data["total_matches"], 0)
        results = data["results"]
        self.assertIsInstance(results, list)

        first_match = results[0]
        self.assertIn("frame_path", first_match)
        self.assertIn("timestamp", first_match)
        self.assertIn("camera_id", first_match)
        self.assertIn("similarity_score", first_match)
        self.assertGreaterEqual(first_match["similarity_score"], 0.0)
        self.assertLessEqual(first_match["similarity_score"], 1.0)

        # The red image should have a higher similarity score for red t-shirt
        self.assertEqual(first_match["frame_path"], str(self.img1_path))

    def test_get_search_query_param(self):
        """Test GET /search with URL query parameters."""
        res = self.client.get("/search?query=blue+vehicle+or+car&top_k=2")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["query"], "blue vehicle or car")
        self.assertGreater(data["total_matches"], 0)
        results = data["results"]

        first_match = results[0]
        # The blue car image should match closest
        self.assertEqual(first_match["frame_path"], str(self.img2_path))

    def test_camera_id_filter(self):
        """Test filtering matches by camera_id."""
        payload = {
            "query": "something",
            "top_k": 5,
            "camera_id": "cam_parking",
        }
        res = self.client.post("/search", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        for match in data["results"]:
            self.assertEqual(match["camera_id"], "cam_parking")

    def test_empty_query_validation(self):
        """Test that whitespace/empty queries return 400/422 status."""
        res_post = self.client.post("/search", json={"query": "   ", "top_k": 5})
        self.assertEqual(res_post.status_code, 400)

        res_get = self.client.get("/search?query=")
        self.assertIn(res_get.status_code, [400, 422])


if __name__ == "__main__":
    unittest.main()
