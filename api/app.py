"""
Surveillance Semantic Search FastAPI Application
================================================
Exposes endpoints for natural language frame search over video surveillance data:
- GET /search and POST /search:
  Converts text queries (e.g. 'person wearing a red t-shirt') into CLIP embeddings
  and queries local ChromaDB vector index to return matching frame paths, timestamps,
  camera IDs, and cosine similarity scores.
- POST /index:
  Embeds and indexes frames extracted by VideoIngestionService into ChromaDB.
- GET /health and GET /stats:
  Reports system health and vector index statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Internal surveillance services
from api.clip_service import CLIPEmbeddingService
from api.vector_service import VectorSearchService
from api.person_mapping import PersonMappingService
from api.stream_service import StreamService
from api.video_ingestion import (
    VideoIngestionService,
    IngestionConfig,
    SamplingMode,
    FrameRecord,
)
from api.vision_analysis import (
    VisionAnalysisService,
    DetectionDatabase,
    DetectionRecord,
    ColorAttributeExtractor,
)

logger = logging.getLogger("SurveillanceAPI")

# =====================================================================
# Pydantic Request & Response Schemas
# =====================================================================

class ROIBox(BaseModel):
    """Normalized spatial Region-of-Interest box on the video frame."""
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left X (0.0 to 1.0)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left Y (0.0 to 1.0)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right X (0.0 to 1.0)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right Y (0.0 to 1.0)")


class ActivityTimespan(BaseModel):
    """A contiguous period of surveillance activity within the search zone."""
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    duration: float = Field(..., description="Duration in seconds")
    camera_id: str = Field(..., description="Camera ID where activity was detected")
    formatted_timespan: str = Field(..., description="Human-readable timespan, e.g. 00:02.0s - 00:08.5s")
    description: str = Field(..., description="Summary of activity in the ROI zone")
    peak_similarity: float = Field(..., description="Peak confidence score during this interval")
    frame_count: int = Field(..., description="Number of frames recorded in this timespan")
    key_frame_url: Optional[str] = Field(default=None, description="URL of representative frame")
    key_crop_url: Optional[str] = Field(default=None, description="URL of cropped detection")
    subject_class: Optional[str] = Field(default=None, description="Class of detected subject (e.g. person)")


class SearchRequest(BaseModel):
    """Payload for natural language semantic frame search with optional spatial ROI."""
    query: str = Field(
        ...,
        description="Natural language query string describing the target scene or subject",
        examples=["person wearing a red t-shirt", "person passed through", "someone near counter"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of top matching frames to return",
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict search to a specific camera ID",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity threshold score (0.0 to 1.0)",
    )
    roi: Optional[ROIBox] = Field(
        default=None,
        description="Optional drawn Region-of-Interest search zone within the frame",
    )


class SearchResultItem(BaseModel):
    """Metadata representing an individual matching frame."""
    frame_path: str = Field(..., description="File path to the matching extracted frame image")
    timestamp: float = Field(..., description="Timestamp in seconds from video start")
    camera_id: str = Field(..., description="Identifier of the capturing camera")
    similarity_score: float = Field(..., description="Calibrated match confidence score (0.0 to 1.0)")
    raw_similarity: Optional[float] = Field(default=None, description="Raw CLIP cosine similarity (-1.0 to 1.0)")
    image_url: Optional[str] = Field(default=None, description="HTTP URL to view the full frame snapshot")
    crop_path: Optional[str] = Field(default=None, description="File path to the detected cropped subject")
    crop_url: Optional[str] = Field(default=None, description="HTTP URL to view the cropped subject")
    crop_class: Optional[str] = Field(default=None, description="Detected object class (e.g. person, car)")
    crop_details: Optional[Dict[str, Any]] = Field(default=None, description="Attributes and bbox of cropped detection")
    in_roi: Optional[bool] = Field(default=None, description="Whether this frame detection intersected the drawn ROI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional frame metadata")


class SearchResponse(BaseModel):
    """Response containing matched surveillance frames and activity timespans."""
    query: str = Field(..., description="The original natural language query string")
    total_matches: int = Field(..., description="Total number of frames returned")
    results: List[SearchResultItem] = Field(..., description="Ranked list of matching frame records")
    roi: Optional[ROIBox] = Field(default=None, description="Echo of the search ROI box if provided")
    timespans: List[ActivityTimespan] = Field(default_factory=list, description="Contiguous activity timespans when subject passed through the ROI")


class IndexFramesRequest(BaseModel):
    """Payload to batch index video frames into the vector database."""
    frames: List[Dict[str, Any]] = Field(
        ...,
        description="List of frame dictionaries containing at least frame_path, timestamp, and camera_id",
    )


class IndexResponse(BaseModel):
    """Response returned upon completion of frame indexing."""
    status: str
    indexed_count: int
    total_collection_count: int


# =====================================================================
# FastAPI Application & Dependencies
# =====================================================================

app = FastAPI(
    title="Surveillance Semantic Search API",
    description="Multi-modal video surveillance semantic frame search powered by CLIP and ChromaDB.",
    version="1.0.0",
)

# Enable CORS for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service singletons
clip_service = CLIPEmbeddingService.get_instance(lazy_load=True)
vector_service = VectorSearchService(persist_dir="output/chroma_db")
ingestion_service = VideoIngestionService()
vision_svc = VisionAnalysisService(
    model_name_or_path="yolov8n.pt",
    crops_dir="output/crops",
    db_path="output/detections.db",
)
person_service = PersonMappingService(db_path="output/detections.db", faces_dir="output/faces")
stream_service = StreamService(sources_dir="sources", output_dir="output")


def execute_search(
    query_str: str,
    top_k: int = 5,
    camera_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
    roi: Optional[ROIBox] = None,
) -> SearchResponse:
    """Core logic to encode query with CLIP, retrieve candidate frames, apply hybrid SQLite re-ranking, and compute spatial ROI activity timespans."""
    clean_query = query_str.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    try:
        import sqlite3
        import urllib.parse

        # 1. Convert text query into normalized CLIP embedding using multi-prompt ensembling
        text_embedding = clip_service.encode_text(clean_query, use_ensemble=True)

        # 2. Retrieve top candidates from ChromaDB
        candidate_k = min(max(top_k * 4, 30), 100)
        raw_matches = vector_service.search(
            query_embedding=text_embedding,
            top_k=candidate_k,
            camera_id=camera_id,
            min_similarity=None,
        )

        # 3. Hybrid Re-ranking & Spatial ROI Overlap Analysis
        db_path = Path("output/detections.db")
        colors = ["red", "black", "blue", "white", "gray", "green", "yellow", "orange", "brown", "dark", "light"]
        q_lower = clean_query.lower()
        matching_colors = [c for c in colors if c in q_lower]
        mentions_person = any(w in q_lower for w in ["person", "man", "woman", "guy", "someone", "suspect", "thief", "hoodie", "jacket", "shirt", "pants", "clothing", "pass", "through", "walk"])

        conn = None
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
            except Exception:
                conn = None

        spatial_matches: List[Dict[str, Any]] = []
        spatial_frames_map: Dict[str, List[Dict[str, Any]]] = {}

        # 4. If ROI box is provided, find all detections geometrically inside or intersecting the ROI
        if conn is not None and roi is not None:
            rx1 = min(roi.x1, roi.x2)
            rx2 = max(roi.x1, roi.x2)
            ry1 = min(roi.y1, roi.y2)
            ry2 = max(roi.y1, roi.y2)
            # Reference 1280x720 pixel space
            roi_px_x1 = rx1 * 1280.0
            roi_px_y1 = ry1 * 720.0
            roi_px_x2 = rx2 * 1280.0
            roi_px_y2 = ry2 * 720.0

            try:
                c = conn.cursor()
                if camera_id and camera_id != "all":
                    c.execute("""
                        SELECT frame_path, camera_id, timestamp, class_name, confidence,
                               bbox_x1, bbox_y1, bbox_x2, bbox_y2, upper_color, lower_color, dominant_color, crop_path
                        FROM detections
                        WHERE camera_id = ?
                        ORDER BY timestamp ASC
                    """, (camera_id,))
                else:
                    c.execute("""
                        SELECT frame_path, camera_id, timestamp, class_name, confidence,
                               bbox_x1, bbox_y1, bbox_x2, bbox_y2, upper_color, lower_color, dominant_color, crop_path
                        FROM detections
                        ORDER BY timestamp ASC
                    """)

                for row in c.fetchall():
                    f_p, c_id, ts, cls_n, conf, bx1, by1, bx2, by2, up_c, low_c, dom_c, crp_p = row
                    ix1 = max(bx1, roi_px_x1)
                    iy1 = max(by1, roi_px_y1)
                    ix2 = min(bx2, roi_px_x2)
                    iy2 = min(by2, roi_px_y2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        box_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                        # Check overlap threshold
                        if (inter_area / box_area) >= 0.08 or inter_area > 150:
                            det_info = {
                                "frame_path": f_p,
                                "camera_id": c_id,
                                "timestamp": ts,
                                "class_name": cls_n,
                                "confidence": conf,
                                "upper_color": up_c,
                                "lower_color": low_c,
                                "dominant_color": dom_c,
                                "crop_path": crp_p,
                                "bbox": [bx1, by1, bx2, by2],
                            }
                            spatial_matches.append(det_info)
                            if f_p not in spatial_frames_map:
                                spatial_frames_map[f_p] = []
                            spatial_frames_map[f_p].append(det_info)
            except Exception as e:
                logger.warning(f"Spatial ROI lookup error: {e}")

        # Compute activity timespans from spatial matches
        timespans: List[ActivityTimespan] = []
        if spatial_matches:
            clusters: List[List[Dict[str, Any]]] = []
            current_cluster: List[Dict[str, Any]] = []
            for sm in spatial_matches:
                if not current_cluster:
                    current_cluster.append(sm)
                else:
                    prev_ts = current_cluster[-1]["timestamp"]
                    if (sm["timestamp"] - prev_ts) <= 3.5:
                        current_cluster.append(sm)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [sm]
            if current_cluster:
                clusters.append(current_cluster)

            for cl in clusters:
                t_start = cl[0]["timestamp"]
                t_end = cl[-1]["timestamp"]
                dur = round(max(0.5, t_end - t_start), 2)
                cam_id = cl[0]["camera_id"]
                best_item = max(cl, key=lambda x: x["confidence"])
                cls_name = best_item["class_name"]

                s_min, s_sec = int(t_start // 60), t_start % 60
                e_min, e_sec = int(t_end // 60), t_end % 60
                fmt_str = f"{s_min:02d}:{s_sec:05.2f}s - {e_min:02d}:{e_sec:05.2f}s ({dur}s activity)"

                k_frame_url = f"/api/frame?path={urllib.parse.quote(best_item['frame_path'])}"
                k_crop_url = f"/api/frame?path={urllib.parse.quote(best_item['crop_path'])}" if (best_item.get("crop_path") and Path(best_item["crop_path"]).exists()) else None

                timespans.append(
                    ActivityTimespan(
                        start_time=round(t_start, 2),
                        end_time=round(t_end, 2),
                        duration=dur,
                        camera_id=cam_id,
                        formatted_timespan=fmt_str,
                        description=f"{cls_name.capitalize()} passed through designated search zone",
                        peak_similarity=round(best_item["confidence"], 4),
                        frame_count=len(cl),
                        key_frame_url=k_frame_url,
                        key_crop_url=k_crop_url,
                        subject_class=cls_name,
                    )
                )

        # Merge candidate matches with spatial matches
        matched_fpaths = {m["frame_path"] for m in raw_matches}
        augmented_matches = list(raw_matches)
        for f_p, dets in spatial_frames_map.items():
            if f_p not in matched_fpaths and Path(f_p).exists():
                augmented_matches.append({
                    "frame_path": f_p,
                    "timestamp": dets[0]["timestamp"],
                    "camera_id": dets[0]["camera_id"],
                    "similarity_score": 0.30,
                    "raw_similarity": 0.22,
                    "metadata": {},
                })

        enhanced_results: List[SearchResultItem] = []
        for m in augmented_matches:
            fpath = m["frame_path"]
            raw_sim = m.get("raw_similarity", m["similarity_score"])
            cal_score = m["similarity_score"]
            meta = dict(m.get("metadata", {}))
            matched_detection = None
            boost = 0.0
            in_roi_flag = False

            if roi is not None:
                if fpath in spatial_frames_map:
                    in_roi_flag = True
                    boost += 0.25
                else:
                    boost -= 0.35

            dets = []
            if conn is not None:
                try:
                    c = conn.cursor()
                    c.execute(
                        "SELECT class_name, upper_color, lower_color, dominant_color, crop_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM detections WHERE frame_path = ?",
                        (fpath,),
                    )
                    dets = c.fetchall()
                    for det_class, up_col, low_col, dom_col, crop_p, x1, y1, x2, y2 in dets:
                        # Boost if detected color matches query color
                        if matching_colors and (up_col in matching_colors or low_col in matching_colors or dom_col in matching_colors):
                            matched_color = up_col if up_col in matching_colors else (low_col if low_col in matching_colors else dom_col)
                            boost += 0.08
                            matched_detection = {
                                "class": det_class,
                                "matched_color": matched_color,
                                "upper_color": up_col,
                                "lower_color": low_col,
                                "crop_path": crop_p,
                                "bbox": [x1, y1, x2, y2],
                            }
                            break
                        elif mentions_person and det_class == "person":
                            boost += 0.05
                            if not matched_detection:
                                matched_detection = {
                                    "class": "person",
                                    "upper_color": up_col,
                                    "lower_color": low_col,
                                    "crop_path": crop_p,
                                    "bbox": [x1, y1, x2, y2],
                                }
                except Exception as e:
                    logger.debug(f"SQLite lookup skipped for {fpath}: {e}")

            final_raw = raw_sim + boost
            final_cal = clip_service.calibrate_similarity(final_raw)

            if matched_detection:
                meta["matched_detection"] = matched_detection

            # Filter against min_similarity threshold if provided
            if min_similarity is not None:
                if final_cal < min_similarity and final_raw < min_similarity:
                    continue

            # Resolve crop property for this frame
            crop_p = None
            crop_cls = None
            crop_dtl = None

            # Prioritize detection that fell inside the ROI
            if in_roi_flag and fpath in spatial_frames_map:
                roi_det = spatial_frames_map[fpath][0]
                crop_p = roi_det.get("crop_path")
                crop_cls = roi_det.get("class_name")
                crop_dtl = roi_det
            elif matched_detection and matched_detection.get("crop_path"):
                crop_p = matched_detection["crop_path"]
                crop_cls = matched_detection.get("class", "object")
                crop_dtl = matched_detection
            elif dets:
                for det_class, up_col, low_col, dom_col, c_p, x1, y1, x2, y2 in dets:
                    if c_p and Path(c_p).exists():
                        crop_p = c_p
                        crop_cls = det_class
                        crop_dtl = {
                            "class": det_class,
                            "upper_color": up_col,
                            "lower_color": low_col,
                            "dominant_color": dom_col,
                            "crop_path": c_p,
                            "bbox": [x1, y1, x2, y2],
                        }
                        break

            crop_u = f"/api/frame?path={urllib.parse.quote(crop_p)}" if (crop_p and Path(crop_p).exists()) else None

            enhanced_results.append(
                SearchResultItem(
                    frame_path=fpath,
                    timestamp=m["timestamp"],
                    camera_id=m["camera_id"],
                    similarity_score=round(final_cal, 4),
                    raw_similarity=round(final_raw, 4),
                    image_url=f"/api/frame?path={urllib.parse.quote(fpath)}",
                    crop_path=crop_p,
                    crop_url=crop_u,
                    crop_class=crop_cls,
                    crop_details=crop_dtl,
                    in_roi=in_roi_flag if roi is not None else None,
                    metadata=meta,
                )
            )

        if conn is not None:
            conn.close()

        # Sort by calibrated confidence descending and return top_k
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        final_results = enhanced_results[:top_k]

        return SearchResponse(
            query=clean_query,
            total_matches=len(final_results),
            results=final_results,
            roi=roi,
            timespans=timespans,
        )

    except Exception as exc:
        logger.error(f"Error during semantic search for query '{clean_query}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(exc)}",
        )


# =====================================================================
# API Endpoints
# =====================================================================

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Natural Language Frame Search (POST)",
    description="Takes a JSON payload with a natural language query string and optional spatial ROI box, generates a CLIP text embedding, and queries ChromaDB for top matching frames and activity timespans.",
)
def search_frames_post(payload: SearchRequest) -> SearchResponse:
    return execute_search(
        query_str=payload.query,
        top_k=payload.top_k,
        camera_id=payload.camera_id,
        min_similarity=payload.min_similarity,
        roi=payload.roi,
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Natural Language Frame Search (GET)",
    description="Query-parameter endpoint for natural language surveillance search with optional ROI bounding box coordinates.",
)
def search_frames_get(
    query: str = Query(..., description="Natural language search query string", examples=["person wearing a red t-shirt"]),
    top_k: int = Query(5, ge=1, le=100, description="Max matches to return"),
    camera_id: Optional[str] = Query(None, description="Optional camera ID filter"),
    min_similarity: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    roi_x1: Optional[float] = Query(None, ge=0.0, le=1.0, description="ROI top-left X"),
    roi_y1: Optional[float] = Query(None, ge=0.0, le=1.0, description="ROI top-left Y"),
    roi_x2: Optional[float] = Query(None, ge=0.0, le=1.0, description="ROI bottom-right X"),
    roi_y2: Optional[float] = Query(None, ge=0.0, le=1.0, description="ROI bottom-right Y"),
) -> SearchResponse:
    roi = None
    if roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
        roi = ROIBox(x1=roi_x1, y1=roi_y1, x2=roi_x2, y2=roi_y2)
    return execute_search(
        query_str=query,
        top_k=top_k,
        camera_id=camera_id,
        min_similarity=min_similarity,
        roi=roi,
    )


@app.post(
    "/index",
    response_model=IndexResponse,
    summary="Index Frames into Vector Database",
    description="Embeds extracted frames using CLIP visual encoder and stores them into the local ChromaDB index.",
)
def index_frames(payload: IndexFramesRequest) -> IndexResponse:
    frames = payload.frames
    if not frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Frame list cannot be empty.",
        )

    valid_frames = []
    images_to_encode = []

    for item in frames:
        fpath = item.get("frame_path")
        if fpath and Path(fpath).exists():
            valid_frames.append(item)
            images_to_encode.append(fpath)

    if not valid_frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid existing frame files found in the provided list.",
        )

    try:
        # Encode frames using CLIP image encoder
        embeddings = clip_service.encode_image_batch(images_to_encode)
        indexed_count = vector_service.index_frames_batch(
            records=valid_frames,
            embeddings=embeddings,
        )

        return IndexResponse(
            status="success",
            indexed_count=indexed_count,
            total_collection_count=vector_service.count(),
        )

    except Exception as exc:
        logger.error(f"Error indexing frames: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(exc)}",
        )


@app.get("/health", summary="Health Check")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "clip_model": clip_service.model_name,
        "clip_device": clip_service.device,
        "indexed_frames_count": vector_service.count(),
        "vector_db": "ChromaDB",
    }


@app.get("/api/frame", summary="Get Frame Snapshot Image")
def get_frame(path: str):
    """Streams a frame snapshot image to the frontend."""
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Frame image file not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(p))


@app.get("/api/cameras", summary="Get Available Camera Sources")
def get_cameras() -> Dict[str, List[str]]:
    """Returns unique list of camera IDs available in surveillance."""
    try:
        all_meta = vector_service.collection.get(include=["metadatas"])
        cams = set()
        for m in all_meta.get("metadatas", []):
            if m and "camera_id" in m:
                cams.add(str(m["camera_id"]))
        cam_list = sorted(list(cams)) if cams else ["camera_street_01"]
        return {"cameras": cam_list}
    except Exception:
        return {"cameras": ["camera_street_01"]}


class IngestVideoRequest(BaseModel):
    video_path: str = Field(..., description="Path to MP4 video file on disk or URL (e.g. sources/shop_street.mp4)")
    camera_id: Optional[str] = Field(default=None, description="Unique camera identifier")
    target_fps: float = Field(default=1.0, ge=0.1, le=30.0, description="Sampling rate (frames per second)")
    adaptive: bool = Field(default=False, description="Whether to use dynamic adaptive motion sampling")
    run_yolo: bool = Field(default=True, description="Whether to run YOLO object detection & person cropping")


class IngestVideoResponse(BaseModel):
    status: str
    camera_id: str
    frames_extracted: int
    frames_indexed: int
    detections_found: int


@app.post("/api/ingest", response_model=IngestVideoResponse, summary="Ingest and Index Video File")
def ingest_video_endpoint(payload: IngestVideoRequest) -> IngestVideoResponse:
    """Ingests a video file, runs vision analysis, and indexes embeddings into ChromaDB."""
    vpath = Path(payload.video_path)
    if not vpath.exists() and not payload.video_path.startswith(("rtsp://", "http://")):
        raise HTTPException(status_code=400, detail=f"Video file does not exist: {payload.video_path}")

    cam_id = payload.camera_id or vpath.stem
    frames_dir = Path("output/frames") / cam_id
    sampling_mode = SamplingMode.DYNAMIC_ADAPTIVE if payload.adaptive else SamplingMode.FIXED_FPS
    config = IngestionConfig(sampling_mode=sampling_mode, target_fps=payload.target_fps)

    # 1. Extract frames
    frame_records = ingestion_service.process_video(
        video_source=str(payload.video_path),
        output_folder=frames_dir,
        source_id=cam_id,
        config=config,
    )
    if not frame_records:
        raise HTTPException(status_code=400, detail="No frames could be extracted from video.")

    # 2. Vision analysis (optional)
    det_count = 0
    if payload.run_yolo:
        vision_svc = VisionAnalysisService(
            model_name_or_path="yolov8n.pt",
            crops_dir="output/crops",
            db_path="output/detections.db",
        )
        dets = vision_svc.analyze_ingested_frames(frame_records, save_to_db=True)
        det_count = len(dets)

    # 3. CLIP visual embeddings & ChromaDB indexing
    image_paths = [r["frame_path"] for r in frame_records]
    embeddings = clip_service.encode_image_batch(image_paths)
    indexed_count = vector_service.index_frames_batch(records=frame_records, embeddings=embeddings)

    return IngestVideoResponse(
        status="success",
        camera_id=cam_id,
        frames_extracted=len(frame_records),
        frames_indexed=indexed_count,
        detections_found=det_count,
    )


@app.get("/api/sources", summary="Get Categorized Video Sources")
def get_sources_endpoint():
    """Returns categorized camera video sources available in the sources folder."""
    sources = stream_service.get_categorized_sources()
    return {"sources": sources}


@app.get("/api/sources/{camera_id}/thumbnail", summary="Get Video Source Preview Frame Thumbnail")
def get_source_thumbnail_endpoint(camera_id: str):
    """Returns a single representative frame snapshot for this video source."""
    p = stream_service.get_source_thumbnail_path(camera_id)
    if not p or not p.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thumbnail not found for camera '{camera_id}'",
        )
    return FileResponse(path=str(p), media_type="image/jpeg")


@app.get("/api/video/{camera_id}", summary="Stream Video File for HTML5 Player")
def stream_video_endpoint(camera_id: str):
    """Streams video file supporting HTTP 206 range requests for seamless browser playback."""
    vpath = stream_service.get_source_file_by_camera_id(camera_id)
    if not vpath or not vpath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source video not found for camera '{camera_id}'",
        )
    return FileResponse(path=str(vpath), media_type="video/mp4")


@app.get("/api/persons", summary="Get Mapped Unique Individuals")
def get_mapped_persons_endpoint(camera_id: Optional[str] = Query(None)):
    """Returns all unique mapped individuals with facial features and visual descriptions."""
    persons = person_service.get_all_mapped_persons(camera_id=camera_id)
    for p in persons:
        if p.get("avatar_path") and Path(p["avatar_path"]).exists():
            p["avatar_url"] = f"/api/frame?path={urllib.parse.quote(p['avatar_path'])}"
        else:
            p["avatar_url"] = None
        if p.get("full_crop_path") and Path(p["full_crop_path"]).exists():
            p["crop_url"] = f"/api/frame?path={urllib.parse.quote(p['full_crop_path'])}"
        else:
            p["crop_url"] = None
        if p.get("frame_path") and Path(p["frame_path"]).exists():
            p["frame_url"] = f"/api/frame?path={urllib.parse.quote(p['frame_path'])}"
        else:
            p["frame_url"] = None
    return {"total": len(persons), "persons": persons}


@app.get("/api/persons/{person_id}/frames", summary="Get Frames for a Mapped Individual")
def get_person_frames_endpoint(person_id: str):
    """Returns all frames where this unique person appeared."""
    frames = person_service.get_person_frames(person_id)
    for f in frames:
        f["image_url"] = f"/api/frame?path={urllib.parse.quote(f['frame_path'])}"
        if f.get("crop_path") and Path(f["crop_path"]).exists():
            f["crop_url"] = f"/api/frame?path={urllib.parse.quote(f['crop_path'])}"
        else:
            f["crop_url"] = None
    return {"person_id": person_id, "total": len(frames), "frames": frames}


@app.get("/api/realtime/status", summary="Get Real-time Feed Processing Telemetry")
def get_realtime_status_endpoint():
    """Returns telemetry of real-time stream processing."""
    return stream_service.get_realtime_status()


class SwitchStreamRequest(BaseModel):
    camera_id: str
    prewarm_samples: int = Field(6, ge=1, le=20)


@app.post("/api/realtime/switch-stream", summary="Switch Active Stream & Prewarm Person Mapping")
def switch_stream_endpoint(payload: SwitchStreamRequest):
    """Switches active camera stream and immediately samples keyframes to map faces."""
    res = stream_service.switch_stream_and_prewarm(
        camera_id=payload.camera_id,
        vision_service=vision_svc,
        person_service=person_service,
        clip_service=clip_service,
        vector_service=vector_service,
        sample_count=payload.prewarm_samples,
    )
    return res


class ProcessRealtimeFrameRequest(BaseModel):
    camera_id: str
    timestamp: float
    frame_base64: Optional[str] = None


@app.post("/api/realtime/process-frame", summary="Process Live Feed Frame in Real Time")
def process_realtime_frame_endpoint(payload: ProcessRealtimeFrameRequest):
    """Processes an incoming real-time video snapshot: runs YOLO, face mapping, and CLIP indexing."""
    import base64
    import cv2
    import numpy as np

    frame_bgr = None
    if payload.frame_base64:
        try:
            raw_b64 = payload.frame_base64.split(",")[-1]
            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Error decoding base64 frame: {e}")

    # Fallback: grab frame directly from video file at exact timestamp
    if frame_bgr is None:
        vpath = stream_service.get_source_file_by_camera_id(payload.camera_id)
        if vpath and vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_MSEC, payload.timestamp * 1000.0)
            ret, frame_bgr = cap.read()
            cap.release()

    if frame_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve frame image for processing.",
        )

    res = stream_service.process_incoming_frame(
        frame_bgr=frame_bgr,
        camera_id=payload.camera_id,
        timestamp=payload.timestamp,
        vision_service=vision_svc,
        person_service=person_service,
        clip_service=clip_service,
        vector_service=vector_service,
    )
    return res


# =====================================================================
# Programmatic Helper Function
# =====================================================================

def search_frames(
    query: str,
    top_k: int = 5,
    camera_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Direct Python helper function to perform search without starting an HTTP server.
    """
    response = execute_search(
        query_str=query,
        top_k=top_k,
        camera_id=camera_id,
        min_similarity=min_similarity,
    )
    return [item.model_dump() for item in response.results]


__all__ = [
    "app",
    "search_frames",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "IndexFramesRequest",
    "IndexResponse",
    "clip_service",
    "vector_service",
    "ingestion_service",
    "VideoIngestionService",
    "IngestionConfig",
    "SamplingMode",
    "FrameRecord",
    "VisionAnalysisService",
    "DetectionDatabase",
    "DetectionRecord",
    "ColorAttributeExtractor",
    "CLIPEmbeddingService",
    "VectorSearchService",
]
