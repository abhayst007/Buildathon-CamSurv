"""
Vision Processing Module Root Entrypoint
Allows importing directly from `vision_analysis` or `api.vision_analysis`.
"""

from api.vision_analysis import (
    VisionAnalysisService,
    DetectionDatabase,
    DetectionRecord,
    ColorAttributeExtractor,
)

__all__ = [
    "VisionAnalysisService",
    "DetectionDatabase",
    "DetectionRecord",
    "ColorAttributeExtractor",
]
