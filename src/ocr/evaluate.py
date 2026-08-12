"""Evaluate pretrained PaddleOCR on the AI Hub financial OCR sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .utils import create_ocr, extract_predictions, load_document_image


DEFAULT_DATASET = Path("data/financial_document_OCR_dataset_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--det-limit-side-len", type=int, default=1280)
    parser.add_argument(
        "--image-name", help="Evaluate one image by filename, for example bank_00095.jpg"
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N image/annotation pairs"
    )
    return parser.parse_args()


def polygon_iou(first: list, second: list) -> float:
    """Calculate IoU between two convex four-point polygons."""
    poly_a = np.asarray(first, dtype=np.float32)
    poly_b = np.asarray(second, dtype=np.float32)
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a == 0.0 or area_b == 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def ground_truth_coverage(
    ground_truth_polygon: list, prediction_polygon: list
) -> float:
    """Return the fraction of a ground-truth polygon covered by a prediction.

    Unlike IoU, this score does not penalize a line-level prediction merely for
    being wider than one word-level ground-truth polygon.
    """
    ground_truth = np.asarray(ground_truth_polygon, dtype=np.float32)
    prediction = np.asarray(prediction_polygon, dtype=np.float32)
    ground_truth_area = abs(float(cv2.contourArea(ground_truth)))
    if ground_truth_area == 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(ground_truth, prediction)
    return min(float(intersection) / ground_truth_area, 1.0)


def match_polygons(
    ground_truth: list[dict], predictions: list[dict], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedily create one-to-one polygon matches in descending IoU order."""
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = polygon_iou(gt["polygon"], pred["polygon"])
            if iou >= threshold:
                candidates.append((iou, gt_index, pred_index))

    matches = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def assign_ground_truth_to_predictions(
    ground_truth: list[dict], predictions: list[dict], threshold: float
) -> tuple[dict[int, list[tuple[int, float]]], list[int]]:
    """Assign each word-level ground truth to its best covering prediction line."""
    assignments: dict[int, list[tuple[int, float]]] = {}
    unmatched_ground_truth = []
    for gt_index, gt in enumerate(ground_truth):
        best_prediction = None
        best_coverage = 0.0
        for pred_index, prediction in enumerate(predictions):
            coverage = ground_truth_coverage(gt["polygon"], prediction["polygon"])
            if coverage > best_coverage:
                best_prediction = pred_index
                best_coverage = coverage

        if best_prediction is None or best_coverage < threshold:
            unmatched_ground_truth.append(gt_index)
            continue
        assignments.setdefault(best_prediction, []).append((gt_index, best_coverage))
    return assignments, unmatched_ground_truth


def polygon_center_x(item: dict) -> float:
    return sum(float(point[0]) for point in item["polygon"]) / len(item["polygon"])


def normalize_whitespace(text: str) -> str:
    return "".join(text.split())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Calculate Levenshtein distance without an additional dependency."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def load_annotation(path: Path, image_name: str) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["name"] != image_name:
        raise ValueError(f"Annotation/image mismatch: {path.name} -> {document['name']}")

    ground_truth = []
    for source_index, polygon in enumerate(
        document["annotations"][0]["polygons"]
    ):
        text = polygon["text"].strip()
        annotation_type = int(polygon["type"])

        # Type 0 is an empty/ignore region in almost every sample occurrence.
        # Types 1 and 2 represent printed and handwritten text, respectively.
        if not text or annotation_type not in (1, 2):
            continue

        ground_truth.append(
            {
                "id": polygon["id"],
                "source_index": source_index,
                "type": annotation_type,
                "text": text,
                "polygon": polygon["points"],
            }
        )
    return ground_truth


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def main() -> None:
    args = parse_args()
    annotation_dir = args.dataset / "annotations"
    image_dir = args.dataset / "images"
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'annotations' and 'images' under {args.dataset.resolve()}"
        )

    annotation_paths = sorted(annotation_dir.glob("*.json"))
    if args.image_name:
        target_stem = Path(args.image_name).stem
        annotation_paths = [annotation_dir / f"{target_stem}.json"]
        if not annotation_paths[0].is_file():
            raise FileNotFoundError(f"Annotation not found for {args.image_name}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        annotation_paths = annotation_paths[: args.limit]

    ocr = create_ocr(args.score_threshold, args.det_limit_side_len)
    totals = {
        "ground_truth": 0,
        "predictions": 0,
        "matches": 0,
        "exact_matches": 0,
        "edit_distance": 0,
        "matched_characters": 0,
    }
    coverage_totals = {
        "ground_truth": 0,
        "covered_ground_truth": 0,
        "printed_ground_truth": 0,
        "covered_printed_ground_truth": 0,
        "handwritten_ground_truth": 0,
        "covered_handwritten_ground_truth": 0,
        "predictions": 0,
        "assigned_predictions": 0,
        "raw_edit_distance": 0,
        "raw_reference_characters": 0,
        "normalized_edit_distance": 0,
        "normalized_reference_characters": 0,
        "raw_exact_matches": 0,
        "normalized_exact_matches": 0,
    }
    documents = []

    for index, annotation_path in enumerate(annotation_paths, start=1):
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = image_dir / raw["name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {annotation_path.name}")

        image = load_document_image(image_path)
        result = ocr.predict(image)[0]
        predictions = extract_predictions(result)
        ground_truth = load_annotation(annotation_path, image_path.name)
        matches = match_polygons(ground_truth, predictions, args.iou_threshold)
        assignments, unmatched_gt = assign_ground_truth_to_predictions(
            ground_truth, predictions, args.coverage_threshold
        )

        exact_matches = 0
        distance = 0
        characters = 0
        match_records = []
        for gt_index, pred_index, iou in matches:
            reference = ground_truth[gt_index]["text"]
            hypothesis = predictions[pred_index]["text"]
            item_distance = edit_distance(reference, hypothesis)
            exact_matches += int(reference == hypothesis)
            distance += item_distance
            characters += len(reference)
            match_records.append(
                {
                    "ground_truth_index": gt_index,
                    "ground_truth_id": ground_truth[gt_index]["id"],
                    "ground_truth_source_index": ground_truth[gt_index][
                        "source_index"
                    ],
                    "ground_truth_type": ground_truth[gt_index]["type"],
                    "prediction_index": pred_index,
                    "iou": round(iou, 4),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "edit_distance": item_distance,
                }
            )

        line_records = []
        for pred_index, assigned_items in sorted(assignments.items()):
            assigned_items = sorted(
                assigned_items,
                key=lambda item: polygon_center_x(ground_truth[item[0]]),
            )
            assigned_gt = [ground_truth[gt_index] for gt_index, _ in assigned_items]
            reference = " ".join(item["text"] for item in assigned_gt)
            hypothesis = predictions[pred_index]["text"]
            normalized_reference = normalize_whitespace(reference)
            normalized_hypothesis = normalize_whitespace(hypothesis)
            raw_distance = edit_distance(reference, hypothesis)
            normalized_distance = edit_distance(
                normalized_reference, normalized_hypothesis
            )

            coverage_totals["raw_edit_distance"] += raw_distance
            coverage_totals["raw_reference_characters"] += len(reference)
            coverage_totals["normalized_edit_distance"] += normalized_distance
            coverage_totals["normalized_reference_characters"] += len(
                normalized_reference
            )
            coverage_totals["raw_exact_matches"] += int(reference == hypothesis)
            coverage_totals["normalized_exact_matches"] += int(
                normalized_reference == normalized_hypothesis
            )
            line_records.append(
                {
                    "prediction_index": pred_index,
                    "prediction_text": hypothesis,
                    "prediction_confidence": predictions[pred_index]["confidence"],
                    "reference_line": reference,
                    "normalized_reference_line": normalized_reference,
                    "normalized_prediction_text": normalized_hypothesis,
                    "raw_edit_distance": raw_distance,
                    "normalized_edit_distance": normalized_distance,
                    "ground_truth_words": [
                        {
                            "ground_truth_index": gt_index,
                            "id": ground_truth[gt_index]["id"],
                            "source_index": ground_truth[gt_index]["source_index"],
                            "type": ground_truth[gt_index]["type"],
                            "text": ground_truth[gt_index]["text"],
                            "coverage": round(coverage, 4),
                        }
                        for gt_index, coverage in assigned_items
                    ],
                }
            )

        totals["ground_truth"] += len(ground_truth)
        totals["predictions"] += len(predictions)
        totals["matches"] += len(matches)
        totals["exact_matches"] += exact_matches
        totals["edit_distance"] += distance
        totals["matched_characters"] += characters
        covered_gt = {
            gt_index
            for assigned_items in assignments.values()
            for gt_index, _ in assigned_items
        }
        printed_gt = {
            index for index, item in enumerate(ground_truth) if item["type"] == 1
        }
        handwritten_gt = {
            index for index, item in enumerate(ground_truth) if item["type"] == 2
        }
        coverage_totals["ground_truth"] += len(ground_truth)
        coverage_totals["covered_ground_truth"] += len(covered_gt)
        coverage_totals["printed_ground_truth"] += len(printed_gt)
        coverage_totals["covered_printed_ground_truth"] += len(
            covered_gt & printed_gt
        )
        coverage_totals["handwritten_ground_truth"] += len(handwritten_gt)
        coverage_totals["covered_handwritten_ground_truth"] += len(
            covered_gt & handwritten_gt
        )
        coverage_totals["predictions"] += len(predictions)
        coverage_totals["assigned_predictions"] += len(assignments)
        documents.append(
            {
                "image": image_path.name,
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
                "matched_count": len(matches),
                "one_to_one_matches": match_records,
                "line_assignments": line_records,
                "unmatched_ground_truth_indices": unmatched_gt,
                "unmatched_prediction_indices": [
                    index
                    for index in range(len(predictions))
                    if index not in assignments
                ],
            }
        )
        print(
            f"[{index}/{len(annotation_paths)}] {image_path.name}: "
            f"GT={len(ground_truth)}, pred={len(predictions)}, "
            f"IoU-matched={len(matches)}, covered={len(covered_gt)}, "
            f"assigned-lines={len(assignments)}"
        )

    precision = safe_ratio(totals["matches"], totals["predictions"])
    recall = safe_ratio(totals["matches"], totals["ground_truth"])
    hmean = safe_ratio(2 * precision * recall, precision + recall)
    metrics = {
        "evaluated_documents": len(annotation_paths),
        "evaluation_note": (
            "AI Hub annotations are mostly word-level while PaddleOCR detections "
            "may be line-level. Coverage-based one-to-many line metrics are the "
            "primary project diagnostics; one-to-one IoU metrics are retained as "
            "legacy diagnostics. Neither is an official benchmark score."
        ),
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "one_to_one_detection_precision": round(precision, 6),
        "one_to_one_detection_recall": round(recall, 6),
        "one_to_one_detection_hmean": round(hmean, 6),
        "recognition_cer_on_matched_regions": round(
            safe_ratio(totals["edit_distance"], totals["matched_characters"]), 6
        ),
        "recognition_exact_match_on_matched_regions": round(
            safe_ratio(totals["exact_matches"], totals["matches"]), 6
        ),
        "end_to_end_exact_match_recall": round(
            safe_ratio(totals["exact_matches"], totals["ground_truth"]), 6
        ),
        "counts": totals,
        "coverage_threshold": args.coverage_threshold,
        "word_detection_coverage_recall": round(
            safe_ratio(
                coverage_totals["covered_ground_truth"],
                coverage_totals["ground_truth"],
            ),
            6,
        ),
        "printed_word_coverage_recall": round(
            safe_ratio(
                coverage_totals["covered_printed_ground_truth"],
                coverage_totals["printed_ground_truth"],
            ),
            6,
        ),
        "handwritten_word_coverage_recall": round(
            safe_ratio(
                coverage_totals["covered_handwritten_ground_truth"],
                coverage_totals["handwritten_ground_truth"],
            ),
            6,
        ),
        "prediction_line_assignment_rate": round(
            safe_ratio(
                coverage_totals["assigned_predictions"],
                coverage_totals["predictions"],
            ),
            6,
        ),
        "line_raw_cer": round(
            safe_ratio(
                coverage_totals["raw_edit_distance"],
                coverage_totals["raw_reference_characters"],
            ),
            6,
        ),
        "line_normalized_cer": round(
            safe_ratio(
                coverage_totals["normalized_edit_distance"],
                coverage_totals["normalized_reference_characters"],
            ),
            6,
        ),
        "line_raw_exact_match": round(
            safe_ratio(
                coverage_totals["raw_exact_matches"],
                coverage_totals["assigned_predictions"],
            ),
            6,
        ),
        "line_normalized_exact_match": round(
            safe_ratio(
                coverage_totals["normalized_exact_matches"],
                coverage_totals["assigned_predictions"],
            ),
            6,
        ),
        "coverage_counts": coverage_totals,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metrics": metrics, "documents": documents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nMetrics")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        "\nWarning: ground truth is generally word-level, while PaddleOCR may "
        "return line-level boxes. Prefer coverage and line-level metrics; interpret "
        "one-to-one IoU metrics as legacy diagnostics."
    )
    print(f"\nReport: {args.output.resolve()}")


if __name__ == "__main__":
    main()
