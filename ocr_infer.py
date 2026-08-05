"""Run pretrained Korean PaddleOCR detection + recognition inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a document image")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    args.output.mkdir(parents=True, exist_ok=True)

    # PP-OCR consists of a text detector followed by a text recognizer.
    # We disable optional document/orientation modules to keep the first lab simple.
    ocr = PaddleOCR(
        lang="korean",
        ocr_version="PP-OCRv5",
        # Paddle 3.3.1's Windows oneDNN path currently fails on an attribute
        # used by the PP-OCRv5 detector. The plain CPU kernels are reliable.
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=args.threshold,
    )
    results = ocr.predict(str(args.image))

    records: list[dict] = []
    for page_index, result in enumerate(results):
        texts = result["rec_texts"]
        scores = result["rec_scores"]
        polygons = result["rec_polys"]

        for text, score, polygon in zip(texts, scores, polygons):
            record = {
                "page": page_index,
                "text": text,
                "confidence": round(float(score), 4),
                "polygon": polygon.tolist(),
            }
            records.append(record)
            print(f"[{record['confidence']:.4f}] {record['text']}")

        # PaddleOCR draws the detected polygons and recognized strings.
        result.save_to_img(str(args.output))

    json_path = args.output / "result.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDetected/recognized {len(records)} text lines")
    print(f"JSON: {json_path.resolve()}")
    print(f"Visualization directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
