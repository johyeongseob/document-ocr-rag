"""Shared utilities for pretrained PaddleOCR inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps


def create_ocr(
    score_threshold: float = 0.5, detection_limit_side_length: int = 1280
) -> PaddleOCR:
    """Create the Korean PP-OCRv5 detection and recognition pipeline."""
    return PaddleOCR(
        lang="korean",
        ocr_version="PP-OCRv5",
        # Paddle 3.3.1's Windows oneDNN path fails on the PP-OCRv5 detector.
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=score_threshold,
        text_det_limit_type="max",
        text_det_limit_side_len=detection_limit_side_length,
    )


def load_document_image(path: Path) -> np.ndarray:
    """Load an image in its displayed orientation as a BGR NumPy array.

    The AI Hub sample contains JPEG files whose raw pixel dimensions are
    landscape while their annotations use the EXIF-rotated portrait view.
    Applying ``exif_transpose`` keeps pixels and annotation coordinates aligned.
    """
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_predictions(result: object) -> list[dict]:
    """Convert a PaddleOCR result into JSON-serializable records."""
    return [
        {
            "text": text,
            "confidence": round(float(score), 4),
            "polygon": polygon.tolist(),
        }
        for text, score, polygon in zip(
            result["rec_texts"], result["rec_scores"], result["rec_polys"]
        )
    ]
