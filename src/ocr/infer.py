"""Run pretrained Korean PaddleOCR detection + recognition inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .utils import create_ocr, extract_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a document image")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--det-limit-side-len", type=int, default=1280)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Some OCR outputs contain symbols that the Windows CP949 console cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    args.output.mkdir(parents=True, exist_ok=True)

    ocr = create_ocr(args.threshold, args.det_limit_side_len)
    results = ocr.predict(str(args.image))

    records: list[dict] = []
    for page_index, result in enumerate(results):
        for prediction in extract_predictions(result):
            record = {"page": page_index, **prediction}
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
