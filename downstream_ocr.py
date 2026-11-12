"""Visualize EasyOCR detections on enhanced OCST-MAIE output images."""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clean_text(text: str) -> str:
    """Keep only digit characters for the serial-number OCR task."""
    return "".join(re.findall(r"\d+", text))


def box_center(bbox: list[list[float]]) -> tuple[float, float]:
    points = np.asarray(bbox, dtype=np.float32)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def group_results_into_rows(results: list) -> list[str]:
    """Order numeric detections into the upper and lower serial-number rows."""
    entries = []
    for bbox, text, _confidence in results:
        digits = clean_text(text)
        if digits:
            x_center, y_center = box_center(bbox)
            entries.append({"text": digits, "x": x_center, "y": y_center})

    if not entries:
        return ["", ""]

    entries.sort(key=lambda entry: entry["y"])
    if len(entries) == 1:
        rows = [entries, []]
    else:
        gaps = [entries[index + 1]["y"] - entries[index]["y"] for index in range(len(entries) - 1)]
        split_index = max(range(len(gaps)), key=gaps.__getitem__) + 1
        rows = [entries[:split_index], entries[split_index:]]

    serial_rows = []
    for row in rows:
        serial_rows.append("".join(entry["text"] for entry in sorted(row, key=lambda entry: entry["x"])))
    return serial_rows


def draw_detections(image: np.ndarray, results: list) -> np.ndarray:
    """Draw EasyOCR quadrilaterals and numeric predictions on an image."""
    annotated = image.copy()
    scale = max(0.45, min(annotated.shape[:2]) / 900) + 0.2
    thickness = max(1, round(scale * 2))

    for bbox, text, confidence in results:
        digits = clean_text(text)
        if not digits:
            continue

        points = np.round(np.asarray(bbox)).astype(np.int32)
        cv2.polylines(annotated, [points], True, (0, 255, 0), thickness, cv2.LINE_AA)
        x = int(points[:, 0].min())
        y = max(int(points[:, 1].min()) - 7, int(22 * scale))
        label = f"{digits} ({confidence:.2f})"
        cv2.putText(
            annotated,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def run_downstream_ocr(
    input_dir: Path, output_dir: Path, languages: list[str], gpu: bool
) -> None:
    """Run EasyOCR once over an enhanced-image folder and save annotated copies."""
    import easyocr

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(languages, gpu=gpu)
    failures = []
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(image_path.name)
            continue

        results = reader.readtext(image)
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), draw_detections(image, results)):
            raise RuntimeError(f"Could not write annotated image: {output_path}")

        top_row, bottom_row = group_results_into_rows(results)
        print(f"[{index}/{len(image_paths)}] {image_path.name}: {top_row} / {bottom_row}")

    print(f"Saved annotated images to: {output_dir}")
    print(f"Unreadable images: {len(failures)}")
    for filename in failures:
        print(f"  {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EasyOCR on OCST-MAIE outputs and save one annotated image per input."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/mamba_test_images"),
        help="Folder containing enhanced images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mamba_test_images_ocr"),
        help="Folder for annotated OCR output images.",
    )
    parser.add_argument("--languages", nargs="+", default=["en"], help="EasyOCR language codes.")
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU if EasyOCR and PyTorch can access one.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_downstream_ocr(args.input_dir, args.output_dir, args.languages, args.gpu)
