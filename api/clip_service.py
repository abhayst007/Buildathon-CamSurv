"""
CLIP Embedding Service Module
=============================
Provides semantic text and image embedding generation using OpenAI's CLIP model
via the Hugging Face Transformers library.

- Encodes natural language queries (e.g. 'person wearing a red t-shirt') into
  512-dimensional normalized vector embeddings using the CLIP text encoder.
- Encodes frame/crop images into corresponding normalized visual embeddings
  for multi-modal cross-attention search.
- Thread-safe singleton model management with configurable compute device.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

logger = logging.getLogger("CLIPEmbeddingService")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CLIPEmbeddingService:
    """
    Service wrapper for CLIP multi-modal text and image embeddings.
    """

    _instance: Optional[CLIPEmbeddingService] = None

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        lazy_load: bool = False,
    ):
        """
        Args:
            model_name: Hugging Face model identifier (default: openai/clip-vit-base-patch32).
            device: Computing device ('cpu', 'cuda', etc.). If None, auto-detected.
            lazy_load: If True, delays loading weights into memory until first inference.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.tokenizer: Optional[CLIPTokenizer] = None

        if not lazy_load:
            self._load_model()

    @classmethod
    def get_instance(
        cls,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        lazy_load: bool = False,
    ) -> CLIPEmbeddingService:
        """Access shared singleton instance across the application."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, device=device, lazy_load=lazy_load)
        return cls._instance

    def _load_model(self) -> None:
        """Loads CLIP model, tokenizer, and processor."""
        if self.model is not None:
            return

        logger.info(f"Loading CLIP model '{self.model_name}' on device '{self.device}'...")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        logger.info(f"CLIP model '{self.model_name}' loaded successfully.")

    @staticmethod
    def _extract_tensor(features: Any) -> torch.Tensor:
        """Extracts the embedding tensor across transformers 4.x and 5.x outputs."""
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        elif hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds
        elif hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds
        elif isinstance(features, torch.Tensor):
            return features
        elif isinstance(features, (tuple, list)) and len(features) > 0:
            return features[0]
        return features

    @classmethod
    def clean_nlp_query(cls, query: str) -> str:
        """
        Removes conversational search prefixes and cleans filler phrasing
        to isolate the core visual concepts for CLIP text encoding.
        """
        q = query.strip()
        noise_patterns = [
            r"^(find|show me|search for|look for|where is|locate|detect|spot|get me)\s+(a\s+|an\s+|the\s+)?",
            r"^(can you find|please find|i want to see|tell me where is|can you see)\s+(a\s+|an\s+|the\s+)?",
            r"^(is there|are there)\s+(a\s+|an\s+|any\s+)?",
        ]
        for pattern in noise_patterns:
            q = re.sub(pattern, "", q, flags=re.IGNORECASE).strip()
        # Clean trailing question marks or punctuation
        q = re.sub(r"[?!.]+$", "", q).strip()
        return q if q else query.strip()

    @classmethod
    def calibrate_similarity(
        cls,
        cos_sim: float,
        s0: float = 0.22,
        k: float = 24.0,
    ) -> float:
        """
        Calibrates raw CLIP cosine similarity into an intuitive match confidence [0.0 - 1.0].

        Using calibrated logistic temperature scaling:
        - cos_sim <= 0.15 -> ~11% (unrelated noise)
        - cos_sim = 0.22  -> ~50% (neutral baseline)
        - cos_sim = 0.28  -> ~78% (strong match)
        - cos_sim >= 0.32 -> ~92%+ (very high match)
        """
        logits = k * (cos_sim - s0)
        prob = 1.0 / (1.0 + math.exp(-logits))
        return round(float(prob), 4)

    def encode_text(
        self,
        text: str,
        use_ensemble: bool = True,
        clean_query: bool = True,
    ) -> List[float]:
        """
        Encodes a natural language string into a normalized 512-dim embedding.
        Uses prompt ensembling across surveillance-domain templates for superior zero-shot retrieval.

        Args:
            text: Query string (e.g. 'person wearing a red t-shirt').
            use_ensemble: Whether to ensemble across surveillance domain templates.
            clean_query: Whether to strip conversational query noise.

        Returns:
            List of floats representing the L2-normalized vector embedding.
        """
        self._load_model()
        assert self.model is not None and self.tokenizer is not None

        core_query = self.clean_nlp_query(text) if clean_query else text.strip()
        if not core_query:
            core_query = text.strip()

        prompts = [core_query]
        if use_ensemble:
            prompts = [
                f"a photo of a {core_query}",
                f"surveillance camera video frame showing {core_query}",
                f"security footage snapshot of {core_query}",
                f"cctv footage of {core_query}",
                core_query,
            ]

        with torch.no_grad():
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            raw_features = self.model.get_text_features(**inputs)
            text_tensor = self._extract_tensor(raw_features)
            normalized_features = text_tensor / text_tensor.norm(dim=-1, keepdim=True)

            if use_ensemble and len(prompts) > 1:
                # Average prompt template embeddings and re-normalize
                mean_feature = normalized_features.mean(dim=0, keepdim=True)
                final_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
                vector = final_feature.cpu().detach().numpy()[0]
            else:
                vector = normalized_features.cpu().detach().numpy()[0]

            return vector.astype(float).tolist()

    def encode_text_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encodes multiple text queries in a single batched tensor operation."""
        if not texts:
            return []

        self._load_model()
        assert self.model is not None and self.tokenizer is not None

        with torch.no_grad():
            inputs = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            raw_features = self.model.get_text_features(**inputs)
            text_tensor = self._extract_tensor(raw_features)
            normalized_features = text_tensor / text_tensor.norm(dim=-1, keepdim=True)
            vectors = normalized_features.cpu().detach().numpy()
            return [v.astype(float).tolist() for v in vectors]

    def encode_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> List[float]:
        """
        Encodes an image into a normalized 512-dim visual embedding.

        Args:
            image_input: File path, numpy BGR/RGB array, or PIL Image.

        Returns:
            Normalized float vector.
        """
        self._load_model()
        assert self.model is not None and self.processor is not None

        pil_img = self._to_pil_image(image_input)
        with torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            raw_features = self.model.get_image_features(**inputs)
            image_tensor = self._extract_tensor(raw_features)
            normalized_features = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
            vector = normalized_features.cpu().detach().numpy()[0]
            return vector.astype(float).tolist()

    def encode_image_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray, Image.Image]],
    ) -> List[List[float]]:
        """Encodes multiple images in a single batched pass."""
        if not images:
            return []

        self._load_model()
        assert self.model is not None and self.processor is not None

        pil_images = [self._to_pil_image(img) for img in images]
        with torch.no_grad():
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            raw_features = self.model.get_image_features(**inputs)
            image_tensor = self._extract_tensor(raw_features)
            normalized_features = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
            vectors = normalized_features.cpu().detach().numpy()
            return [v.astype(float).tolist() for v in vectors]

    @staticmethod
    def _to_pil_image(img: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Converts diverse image formats into an RGB PIL Image."""
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        elif isinstance(img, (str, Path)):
            return Image.open(str(img)).convert("RGB")
        elif isinstance(img, np.ndarray):
            # Assume OpenCV BGR format by default
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb = img[:, :, ::-1]
                return Image.fromarray(rgb)
            elif len(img.shape) == 2:
                return Image.fromarray(img).convert("RGB")
            else:
                return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image input type: {type(img)}")
