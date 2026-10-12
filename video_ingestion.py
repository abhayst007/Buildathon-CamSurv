"""
Video Ingestion Service root entrypoint.
Allows importing directly from `video_ingestion` or `api.video_ingestion`.
"""

from api.video_ingestion import (
    VideoIngestionService,
    IngestionConfig,
    SamplingMode,
    FrameRecord,
    MotionDetector,
)

__all__ = [
    "VideoIngestionService",
    "IngestionConfig",
    "SamplingMode",
    "FrameRecord",
    "MotionDetector",
]
