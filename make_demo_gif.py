"""Create a README GIF from PaddleOCR visualization results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from ocr_utils import create_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/financial_document_OCR_dataset_sample/images"),
    )
    parser.add_argument("--frames", type=Path, default=Path("outputs/demo_frames"))
    parser.add_argument("--output", type=Path, default=Path("assets/ocr_demo.gif"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--duration", type=int, default=2000, help="Milliseconds")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--colors", type=int, default=128)
    parser.add_argument("--det-limit-side-len", type=int, default=960)
    return parser.parse_args()


def resize_frame(path: Path, width: int, colors: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return resized.quantize(colors=colors)


def select_same_size_images(directory: Path, limit: int) -> tuple[list[Path], tuple[int, int]]:
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as image:
            displayed_size = ImageOps.exif_transpose(image).size
        groups[displayed_size].append(path)

    displayed_size, paths = max(groups.items(), key=lambda item: len(item[1]))
    if len(paths) < limit:
        raise ValueError(
            f"Largest same-size group has only {len(paths)} images ({displayed_size})"
        )
    return paths[:limit], displayed_size


def main() -> None:
    args = parse_args()
    image_paths, displayed_size = select_same_size_images(args.images, args.limit)
    print(f"Selected {len(image_paths)} images with displayed size {displayed_size}")

    args.frames.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ocr = None

    frame_paths: list[Path] = []
    for index, image_path in enumerate(image_paths, start=1):
        frame_path = args.frames / f"{image_path.stem}_ocr_res_img{image_path.suffix}"
        if not frame_path.is_file():
            if ocr is None:
                ocr = create_ocr(detection_limit_side_length=args.det_limit_side_len)
            results = ocr.predict(str(image_path))
            for result in results:
                result.save_to_img(str(args.frames))
        if not frame_path.is_file():
            raise FileNotFoundError(f"Visualization was not created: {frame_path}")
        frame_paths.append(frame_path)
        print(f"[{index}/{len(image_paths)}] {image_path.name}")

    frames = [resize_frame(path, args.width, args.colors) for path in frame_paths]
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"GIF: {args.output.resolve()}")


if __name__ == "__main__":
    main()
