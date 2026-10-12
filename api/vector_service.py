"""
Vector Database Service Module (ChromaDB)
=========================================
Manages persistent vector indexing and semantic retrieval for video surveillance frames.

- Indexes 512-dim normalized CLIP visual embeddings with associated metadata
  (frame_path, timestamp, camera_id, etc.).
- Performs cosine-similarity nearest neighbor search for natural language queries.
- Supports relational metadata filtering (e.g., filtering matches by specific camera_id).
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("VectorSearchService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VectorSearchService:
    """
    ChromaDB-backed vector search service for surveillance video frames.
    """

    DEFAULT_COLLECTION_NAME = "surveillance_frames"

    def __init__(
        self,
        persist_dir: Union[str, Path] = "output/chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        Args:
            persist_dir: Path to local directory where ChromaDB stores its vector index.
            collection_name: Name of the vector collection.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB PersistentClient at '{self.persist_dir}'...")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection configured for cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector collection '{collection_name}' ready. Total items: {self.collection.count()}")

    def index_frame(
        self,
        frame_path: Optional[str] = None,
        timestamp: Optional[float] = None,
        camera_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        record: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a single frame vector and its metadata into the vector database.
        Supports direct keyword arguments or a record dictionary.
        """
        if record:
            frame_path = frame_path or str(record.get("frame_path", ""))
            timestamp = timestamp if timestamp is not None else float(record.get("timestamp", 0.0))
            camera_id = camera_id or str(record.get("camera_id", "camera_default"))
            if not extra_metadata:
                extra_metadata = record.get("extra_metadata")
            doc_id = doc_id or record.get("id")

        if embedding is None:
            raise ValueError("Embedding must be provided to index_frame")

        record_id = doc_id or str(uuid.uuid4())
        meta: Dict[str, Union[str, int, float, bool]] = {
            "frame_path": str(frame_path or ""),
            "timestamp": float(timestamp or 0.0),
            "camera_id": str(camera_id or "camera_default"),
        }

        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        self.collection.upsert(
            ids=[record_id],
            embeddings=[embedding],
            metadatas=[meta],
        )
        return record_id

    def index_frames_batch(
        self,
        records: Sequence[Dict[str, Any]],
        embeddings: Sequence[List[float]],
    ) -> int:
        """
        Batch index multiple frames and embeddings into the collection.

        Args:
            records: List of dicts with keys: frame_path, timestamp, camera_id, etc.
            embeddings: Parallel list of vector embeddings.

        Returns:
            Number of indexed records.
        """
        if not records or not embeddings:
            return 0
        if len(records) != len(embeddings):
            raise ValueError(f"Length mismatch: {len(records)} records vs {len(embeddings)} embeddings")

        ids: List[str] = []
        metadatas: List[Dict[str, Union[str, int, float, bool]]] = []

        for item in records:
            doc_id = item.get("id") or str(uuid.uuid4())
            ids.append(str(doc_id))

            meta: Dict[str, Union[str, int, float, bool]] = {
                "frame_path": str(item.get("frame_path", "")),
                "timestamp": float(item.get("timestamp", 0.0)),
                "camera_id": str(item.get("camera_id") or item.get("source_id", "camera_default")),
            }
            # Add other primitive fields if present
            for k, v in item.items():
                if k not in ("id", "frame_path", "timestamp", "camera_id", "source_id"):
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif v is not None:
                        meta[k] = str(v)
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=list(embeddings),
            metadatas=metadatas,
        )
        logger.info(f"Successfully batch indexed {len(ids)} frames into ChromaDB.")
        return len(ids)

    @staticmethod
    def calibrate_similarity(cos_sim: float, s0: float = 0.22, k: float = 24.0) -> float:
        """
        Calibrates raw CLIP cosine similarity into an intuitive match confidence probability [0.0 - 1.0].
        - cos_sim <= 0.15 -> ~11% (unrelated noise)
        - cos_sim = 0.22  -> ~50% (neutral baseline)
        - cos_sim = 0.28  -> ~78% (strong match)
        - cos_sim >= 0.32 -> ~92%+ (very high match)
        """
        logits = k * (cos_sim - s0)
        prob = 1.0 / (1.0 + math.exp(-logits))
        return round(float(prob), 4)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        camera_id: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find top matching frames given a query vector embedding.

        Args:
            query_embedding: Normalized query vector from CLIP text encoder.
            top_k: Maximum number of matches to return.
            camera_id: Optional camera ID filter.
            min_similarity: Optional minimum similarity threshold [0.0 - 1.0].

        Returns:
            List of dictionaries:
            [
              {
                "frame_path": str,
                "timestamp": float,
                "camera_id": str,
                "similarity_score": float,
                "metadata": dict
              },
              ...
            ]
        """
        total_items = self.collection.count()
        if total_items == 0:
            logger.warning("Vector database collection is empty.")
            return []

        where_filter = None
        if camera_id is not None:
            where_filter = {"camera_id": camera_id}

        limit = min(top_k, total_items)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        matches: List[Dict[str, Any]] = []
        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            return matches

        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        for meta, dist in zip(metadatas, distances):
            # In cosine space, cosine_distance = 1.0 - cosine_similarity
            raw_similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            calibrated_similarity = self.calibrate_similarity(raw_similarity)

            if min_similarity is not None:
                if calibrated_similarity < min_similarity and raw_similarity < min_similarity:
                    continue

            frame_path = str(meta.get("frame_path", ""))
            timestamp = float(meta.get("timestamp", 0.0))
            cam_id = str(meta.get("camera_id", ""))

            matches.append({
                "frame_path": frame_path,
                "timestamp": round(timestamp, 3),
                "camera_id": cam_id,
                "similarity_score": round(calibrated_similarity, 4),
                "raw_similarity": round(raw_similarity, 4),
                "metadata": meta,
            })

        # Sort matches by similarity descending
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches

    def count(self) -> int:
        """Returns total number of items indexed in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Clears all records in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
