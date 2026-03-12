from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from docx import Document

from DocumentLayoutAnalysis.image_importance_google import classify_figures_with_google
from .gemma_common import (
    GOOGLE_IMPORTANCE_MODEL,
    IGNORED_SOURCE_FILENAMES,
    LAYOUT_CACHE_DIRNAME,
    MCQ_OUTPUT_DIRNAME,
    PROJECT_ROOT,
    SUPPORTED_SOURCE_EXTENSIONS,
    FigureRegion,
    TextRegion,
    bbox_area,
    canonical_filename,
    normalize_positive_int,
    sanitize_path_part,
)

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None

try:
    import fitz
except Exception:  # noqa: BLE001
    try:
        import pymupdf as fitz
    except Exception:  # noqa: BLE001
        fitz = None

PDF_RENDER_SCALE = 2.0
MAX_PDF_PAGES_PER_FILE = 18
MAX_FIGURES_PER_FILE = 12
YOLO_LAYOUT_MODEL_PATH = PROJECT_ROOT / "yolov10x_best.pt"
YOLO_LAYOUT_CONFIDENCE = 0.20
YOLO_LAYOUT_IOU = 0.45
YOLO_LAYOUT_TARGET_LABELS = {"picture", "table"}

NATIVE_TEXT_CHAR_THRESHOLD = 120
OCR_WORD_CONFIDENCE_MIN = 40.0
OCR_MIN_BOX_AREA = 45

FIGURE_FOREGROUND_THRESHOLD = 245
FIGURE_MORPH_KERNEL = 9
FIGURE_MIN_AREA = 15000
FIGURE_MIN_WIDTH = 90
FIGURE_MIN_HEIGHT = 90
FIGURE_MIN_FILL_RATIO = 0.06
FIGURE_MAX_PAGE_COVERAGE = 0.92
FIGURE_NMS_IOU_THRESHOLD = 0.55
FIGURE_MAX_IMAGE_DIM = 900

ASSOCIATION_MAX_BLOCKS = 4
ASSOCIATION_MAX_CHARS = 900
CAPTION_VERTICAL_GAP_MAX = 140
NEAR_TEXT_DISTANCE_RATIO_MAX = 0.55

_YOLO_LAYOUT_MODEL: Any = None
_YOLO_LAYOUT_CLASS_NAMES: dict[int, str] = {}
_YOLO_LAYOUT_TARGET_CLASS_IDS: set[int] = set()


def _horizontal_overlap_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    b_width = max(1.0, bx1 - bx0)
    return overlap / b_width


def _rect_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _parse_ocr_line_regions(
    image_bgr: Any,
    *,
    source_file: str,
    source_page: int,
) -> list[TextRegion]:
    if pytesseract is None:
        return []

    try:
        ocr = pytesseract.image_to_data(image_bgr, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001
        return []

    line_groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    count = len(ocr.get("text", []))

    for index in range(count):
        text_raw = str(ocr["text"][index] or "").strip()
        if not text_raw:
            continue

        conf_raw = str(ocr["conf"][index]).strip()
        try:
            confidence = float(conf_raw)
        except ValueError:
            confidence = -1.0

        if confidence < OCR_WORD_CONFIDENCE_MIN:
            continue

        x = int(ocr["left"][index])
        y = int(ocr["top"][index])
        w = int(ocr["width"][index])
        h = int(ocr["height"][index])
        if w * h < OCR_MIN_BOX_AREA:
            continue

        key = (
            int(ocr.get("block_num", [0] * count)[index]),
            int(ocr.get("par_num", [0] * count)[index]),
            int(ocr.get("line_num", [index] * count)[index]),
        )
        row = line_groups.setdefault(
            key,
            {
                "x0": x,
                "y0": y,
                "x1": x + w,
                "y1": y + h,
                "parts": [],
                "confidences": [],
            },
        )
        row["x0"] = min(row["x0"], x)
        row["y0"] = min(row["y0"], y)
        row["x1"] = max(row["x1"], x + w)
        row["y1"] = max(row["y1"], y + h)
        row["parts"].append(text_raw)
        row["confidences"].append(confidence)

    regions: list[TextRegion] = []
    for row in line_groups.values():
        text = " ".join(row["parts"]).strip()
        if not text:
            continue
        avg_conf = sum(row["confidences"]) / max(1, len(row["confidences"]))
        regions.append(
            TextRegion(
                source_file=source_file,
                source_page=source_page,
                bbox=(float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])),
                text=text,
                confidence=avg_conf,
                source_kind="ocr",
            )
        )

    regions.sort(key=lambda region: (region.bbox[1], region.bbox[0]))
    return regions


def _pdf_block_regions(
    page: Any,
    *,
    source_file: str,
    source_page: int,
    scale_x: float,
    scale_y: float,
) -> list[TextRegion]:
    try:
        blocks = page.get_text("blocks")
    except Exception:  # noqa: BLE001
        return []

    regions: list[TextRegion] = []
    for block in blocks:
        if not isinstance(block, tuple) or len(block) < 5:
            continue

        x0, y0, x1, y1 = float(block[0]), float(block[1]), float(block[2]), float(block[3])
        text = str(block[4] or "").strip()
        if not text:
            continue

        regions.append(
            TextRegion(
                source_file=source_file,
                source_page=source_page,
                bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                text=text,
                confidence=1.0,
                source_kind="native",
            )
        )

    regions.sort(key=lambda region: (region.bbox[1], region.bbox[0]))
    return regions


def _render_page_to_bgr(page: Any, *, scale: float) -> tuple[Any, float, float]:
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    if np is None:
        raise RuntimeError("numpy is required for PDF layout detection.")
    if cv2 is None:
        raise RuntimeError("opencv-python is required for PDF layout detection.")

    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    page_rect = page.rect
    scale_x = pixmap.width / max(float(page_rect.width), 1.0)
    scale_y = pixmap.height / max(float(page_rect.height), 1.0)
    return image, scale_x, scale_y


def _load_yolo_layout_detector() -> tuple[Any, dict[int, str], set[int]]:
    global _YOLO_LAYOUT_MODEL, _YOLO_LAYOUT_CLASS_NAMES, _YOLO_LAYOUT_TARGET_CLASS_IDS

    if YOLO is None:
        raise RuntimeError(
            "YOLO figure detection requires the 'ultralytics' package. "
            "Install it with: python3 -m pip install ultralytics"
        )
    if not YOLO_LAYOUT_MODEL_PATH.exists():
        raise RuntimeError(f"YOLO model file not found: {YOLO_LAYOUT_MODEL_PATH}")

    if _YOLO_LAYOUT_MODEL is not None and _YOLO_LAYOUT_TARGET_CLASS_IDS:
        return _YOLO_LAYOUT_MODEL, _YOLO_LAYOUT_CLASS_NAMES, _YOLO_LAYOUT_TARGET_CLASS_IDS

    model = YOLO(str(YOLO_LAYOUT_MODEL_PATH))
    names_raw = getattr(model, "names", {})
    if not isinstance(names_raw, dict):
        names_raw = {}

    normalized_names: dict[int, str] = {}
    for class_id_raw, class_name in names_raw.items():
        try:
            class_id = int(class_id_raw)
        except (ValueError, TypeError):
            continue
        normalized_names[class_id] = str(class_name).strip()

    target_class_ids = {
        class_id
        for class_id, class_name in normalized_names.items()
        if class_name.lower() in YOLO_LAYOUT_TARGET_LABELS
    }
    if not target_class_ids:
        labels = ", ".join(sorted(name for name in normalized_names.values() if name))
        raise RuntimeError(
            "YOLO model does not expose required detection labels for picture/table. "
            f"Available labels: {labels or 'none'}"
        )

    _YOLO_LAYOUT_MODEL = model
    _YOLO_LAYOUT_CLASS_NAMES = normalized_names
    _YOLO_LAYOUT_TARGET_CLASS_IDS = target_class_ids
    return _YOLO_LAYOUT_MODEL, _YOLO_LAYOUT_CLASS_NAMES, _YOLO_LAYOUT_TARGET_CLASS_IDS


def _detect_figure_boxes(image_bgr: Any) -> list[tuple[tuple[int, int, int, int], str, float]]:
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for layout detection.")

    model, class_names, target_class_ids = _load_yolo_layout_detector()
    try:
        predictions = model.predict(
            image_bgr,
            verbose=False,
            conf=YOLO_LAYOUT_CONFIDENCE,
            iou=YOLO_LAYOUT_IOU,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"YOLO detection failed on rendered PDF page: {exc}") from exc

    detections: list[tuple[tuple[int, int, int, int], str, float]] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue

        xyxy_rows = getattr(boxes, "xyxy", None)
        conf_rows = getattr(boxes, "conf", None)
        cls_rows = getattr(boxes, "cls", None)
        if xyxy_rows is None or conf_rows is None or cls_rows is None:
            continue

        for xyxy, conf_raw, cls_raw in zip(xyxy_rows, conf_rows, cls_rows):
            try:
                class_id = int(float(cls_raw))
            except Exception:  # noqa: BLE001
                continue
            if class_id not in target_class_ids:
                continue

            try:
                x0, y0, x1, y1 = [int(round(float(value))) for value in xyxy.tolist()]
                confidence = float(conf_raw)
            except Exception:  # noqa: BLE001
                continue

            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(((x0, y0, x1, y1), class_names.get(class_id, "figure"), confidence))

    detections.sort(key=lambda row: row[2], reverse=True)
    return detections[:MAX_FIGURES_PER_FILE]


def _extract_page_text_chunk(text_regions: list[TextRegion], *, max_chars: int) -> str:
    chunk = " ".join(region.text for region in text_regions if region.text)
    if len(chunk) <= max_chars:
        return chunk.strip()
    return chunk[:max_chars].rsplit(" ", 1)[0].strip()


def _associate_text_to_figure(
    figure_bbox: tuple[int, int, int, int],
    text_regions: list[TextRegion],
    *,
    max_blocks: int = ASSOCIATION_MAX_BLOCKS,
) -> tuple[str, float]:
    fx0, fy0, fx1, fy1 = figure_bbox
    f_center = _rect_center((float(fx0), float(fy0), float(fx1), float(fy1)))
    page_width = max(float(max([region.bbox[2] for region in text_regions], default=fx1)), float(fx1))
    page_height = max(float(max([region.bbox[3] for region in text_regions], default=fy1)), float(fy1))
    page_diag = max(1.0, math.hypot(page_width, page_height))

    scored_rows: list[tuple[float, TextRegion]] = []
    for region in text_regions:
        rx0, ry0, rx1, ry1 = region.bbox
        vertical_gap = min(abs(ry1 - fy0), abs(fy1 - ry0))
        overlap_ratio = _horizontal_overlap_ratio(region.bbox, (float(fx0), float(fy0), float(fx1), float(fy1)))
        r_center = _rect_center(region.bbox)
        center_distance_ratio = math.hypot(r_center[0] - f_center[0], r_center[1] - f_center[1]) / page_diag

        score = 0.0
        if vertical_gap <= CAPTION_VERTICAL_GAP_MAX:
            score += 2.5
        if overlap_ratio >= 0.5:
            score += 2.0
        if center_distance_ratio <= NEAR_TEXT_DISTANCE_RATIO_MAX:
            score += 1.5
        if region.source_kind == "native":
            score += 0.5
        if not region.text.strip():
            score = 0.0
        if score > 0:
            scored_rows.append((score, region))

    if not scored_rows:
        return ("", 0.0)

    scored_rows.sort(key=lambda row: row[0], reverse=True)
    selected_regions = [region for _, region in scored_rows[:max_blocks]]
    associated_text = " ".join(region.text.strip() for region in selected_regions if region.text).strip()
    if len(associated_text) > ASSOCIATION_MAX_CHARS:
        associated_text = associated_text[:ASSOCIATION_MAX_CHARS].rsplit(" ", 1)[0].strip()
    best_score = scored_rows[0][0]
    return associated_text, best_score


def _build_figure_regions_from_detections(
    image_bgr: Any,
    detections: list[tuple[tuple[int, int, int, int], str, float]],
    *,
    source_file: str,
    source_page: int,
    text_regions: list[TextRegion],
) -> list[FigureRegion]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for figure extraction.")

    figure_regions: list[FigureRegion] = []
    for index, (bbox, detection_label, detection_confidence) in enumerate(detections, start=1):
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        if crop is None or getattr(crop, "size", 0) <= 0:
            continue

        crop_height, crop_width = crop.shape[:2]
        if crop_width < FIGURE_MIN_WIDTH or crop_height < FIGURE_MIN_HEIGHT:
            continue

        area = crop_width * crop_height
        if area < FIGURE_MIN_AREA:
            continue

        page_area = image_bgr.shape[0] * image_bgr.shape[1]
        if area / max(1, page_area) > FIGURE_MAX_PAGE_COVERAGE:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        foreground_mask = gray < FIGURE_FOREGROUND_THRESHOLD
        fill_ratio = float(foreground_mask.sum()) / max(1, foreground_mask.size)
        if fill_ratio < FIGURE_MIN_FILL_RATIO:
            continue

        resized_height, resized_width = crop_height, crop_width
        if max(crop_height, crop_width) > FIGURE_MAX_IMAGE_DIM:
            scale = FIGURE_MAX_IMAGE_DIM / float(max(crop_height, crop_width))
            resized_width = max(1, int(round(crop_width * scale)))
            resized_height = max(1, int(round(crop_height * scale)))
            crop = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        encoded_ok, encoded_png = cv2.imencode(".png", crop)
        if not encoded_ok:
            continue

        associated_text, association_score = _associate_text_to_figure(
            bbox,
            text_regions,
        )
        figure_regions.append(
            FigureRegion(
                source_file=source_file,
                source_page=source_page,
                source_figure_id=f"fig#{index}",
                bbox=bbox,
                detection_label=detection_label,
                detection_confidence=detection_confidence,
                associated_text=associated_text,
                association_score=association_score,
                image_bytes=encoded_png.tobytes(),
            )
        )

    return figure_regions


def _extract_pdf_layout_evidence(
    file_path: str | Path,
    *,
    max_chars_for_file: int,
) -> tuple[list[dict[str, Any]], list[FigureRegion]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF processing.")

    file_path = Path(file_path).expanduser().resolve()
    doc = fitz.open(str(file_path))

    source_chunks: list[dict[str, Any]] = []
    figure_regions: list[FigureRegion] = []
    total_chars = 0
    try:
        for page_index in range(min(len(doc), MAX_PDF_PAGES_PER_FILE)):
            page = doc.load_page(page_index)
            source_page = page_index + 1
            image_bgr, scale_x, scale_y = _render_page_to_bgr(page, scale=PDF_RENDER_SCALE)
            text_regions = _pdf_block_regions(
                page,
                source_file=file_path.name,
                source_page=source_page,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            page_text = _extract_page_text_chunk(text_regions, max_chars=max(0, max_chars_for_file - total_chars))
            if page_text:
                source_chunks.append(
                    {
                        "source_file": file_path.name,
                        "source_page": source_page,
                        "text": page_text,
                    }
                )
                total_chars += len(page_text)

            native_char_count = sum(len(region.text) for region in text_regions)
            ocr_regions: list[TextRegion] = []
            if native_char_count < NATIVE_TEXT_CHAR_THRESHOLD:
                ocr_regions = _parse_ocr_line_regions(
                    image_bgr,
                    source_file=file_path.name,
                    source_page=source_page,
                )

            evidence_regions = text_regions or ocr_regions
            detections = _detect_figure_boxes(image_bgr)
            if detections and evidence_regions:
                figure_regions.extend(
                    _build_figure_regions_from_detections(
                        image_bgr,
                        detections,
                        source_file=file_path.name,
                        source_page=source_page,
                        text_regions=evidence_regions,
                    )
                )
    finally:
        doc.close()

    return source_chunks, figure_regions


def _load_image_file_bgr(file_path: str | Path) -> Any:
    file_path = Path(file_path).expanduser().resolve()
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for image processing.")

    try:
        encoded = np.fromfile(str(file_path), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        encoded = None

    image_bgr = None
    if encoded is not None and getattr(encoded, "size", 0) > 0:
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        image_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unable to load image file: {file_path}")
    return image_bgr


def _extract_image_layout_evidence(
    file_path: str | Path,
    *,
    max_chars_for_file: int,
) -> tuple[list[dict[str, Any]], list[FigureRegion]]:
    file_path = Path(file_path).expanduser().resolve()
    image_bgr = _load_image_file_bgr(file_path)
    text_regions = _parse_ocr_line_regions(
        image_bgr,
        source_file=file_path.name,
        source_page=1,
    )
    source_chunks: list[dict[str, Any]] = []
    page_text = _extract_page_text_chunk(text_regions, max_chars=max_chars_for_file)
    if page_text:
        source_chunks.append(
            {
                "source_file": file_path.name,
                "source_page": 1,
                "text": page_text,
            }
        )

    detections = _detect_figure_boxes(image_bgr)
    figure_regions = (
        _build_figure_regions_from_detections(
            image_bgr,
            detections,
            source_file=file_path.name,
            source_page=1,
            text_regions=text_regions,
        )
        if detections and text_regions
        else []
    )
    return source_chunks, figure_regions


def _extract_text_file(file_path: str | Path, *, max_chars: int) -> str:
    file_path = Path(file_path).expanduser().resolve()
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    if len(raw_text) <= max_chars:
        return raw_text.strip()
    return raw_text[:max_chars].rsplit(" ", 1)[0].strip()


def _extract_docx_text(file_path: str | Path, *, max_chars: int) -> str:
    file_path = Path(file_path).expanduser().resolve()
    document = Document(str(file_path))
    raw_text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)
    if len(raw_text) <= max_chars:
        return raw_text.strip()
    return raw_text[:max_chars].rsplit(" ", 1)[0].strip()


def _extract_non_pdf_source_chunks(file_path: str | Path, *, max_chars: int) -> list[dict[str, Any]]:
    file_path = Path(file_path).expanduser().resolve()
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = _extract_text_file(file_path, max_chars=max_chars)
    elif suffix == ".docx":
        text = _extract_docx_text(file_path, max_chars=max_chars)
    else:
        return []

    return (
        [{"source_file": file_path.name, "source_page": 1, "text": text}]
        if text
        else []
    )


def _build_tagged_study_text(
    source_chunks: list[dict[str, Any]],
    figure_regions: list[FigureRegion],
) -> str:
    lines: list[str] = []
    for chunk in source_chunks:
        source_file = str(chunk.get("source_file") or "").strip()
        source_page = normalize_positive_int(chunk.get("source_page"))
        text = str(chunk.get("text") or "").strip()
        if source_file and source_page is not None and text:
            lines.append(f'[SRC file="{source_file}" page={source_page}] {text}')

    for figure in figure_regions:
        safe_file = figure.source_file.replace('"', "'")
        safe_text = re.sub(r"\s+", " ", figure.associated_text).strip()
        lines.append(
            f'[FIG_SRC file="{safe_file}" page={figure.source_page} fig="{figure.source_figure_id}"] '
            f"{safe_text}"
        )
    return "\n\n".join(lines).strip()


def _layout_cache_dir_from_summary_path(summary_path: str | Path) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    return summary_path.parent.parent / LAYOUT_CACHE_DIRNAME


def _mcq_output_dir_from_summary_path(summary_path: str | Path) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    return summary_path.parent.parent / MCQ_OUTPUT_DIRNAME


def _load_figure_image_map(summary_path: str | Path) -> dict[tuple[str, int, str], str]:
    layout_dir = _layout_cache_dir_from_summary_path(summary_path)
    figures_root = layout_dir / "figures"
    image_map: dict[tuple[str, int, str], str] = {}
    if not figures_root.exists():
        return image_map

    for image_path in figures_root.rglob("*.png"):
        relative_parts = image_path.relative_to(figures_root).parts
        if len(relative_parts) != 2:
            continue
        source_file = relative_parts[0]
        match = re.match(r"page_(\d+)__(fig#\d+)\.png$", relative_parts[1])
        if not match:
            continue
        image_map[(source_file, int(match.group(1)), match.group(2))] = str(image_path)
    return image_map


def _write_layout_cache(
    *,
    layout_dir: Path,
    summary_payload: dict[str, Any],
    source_chunks: list[dict[str, Any]],
    figure_regions: list[FigureRegion],
) -> None:
    if layout_dir.exists():
        shutil.rmtree(layout_dir)
    layout_dir.mkdir(parents=True, exist_ok=True)

    (layout_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (layout_dir / "source_chunks.json").write_text(json.dumps(source_chunks, indent=2), encoding="utf-8")

    figures_root = layout_dir / "figures"
    for figure in figure_regions:
        source_dir = figures_root / sanitize_path_part(figure.source_file, "unknown_source")
        source_dir.mkdir(parents=True, exist_ok=True)
        image_path = source_dir / f"page_{figure.source_page}__{figure.source_figure_id}.png"
        image_path.write_bytes(figure.image_bytes)

    figure_rows = []
    for figure in figure_regions:
        figure_rows.append(
            {
                "source_file": figure.source_file,
                "source_page": figure.source_page,
                "source_figure_id": figure.source_figure_id,
                "bbox": list(figure.bbox),
                "detection_label": figure.detection_label,
                "detection_confidence": figure.detection_confidence,
                "associated_text": figure.associated_text,
                "association_score": figure.association_score,
                "mime_type": figure.mime_type,
                "importance_score": figure.importance_score,
                "importance_level": figure.importance_level,
                "importance_type": figure.importance_type,
                "importance_reasons": list(figure.importance_reasons),
                "importance_metrics": dict(figure.importance_metrics),
                "selected_for_prompt": figure.selected_for_prompt,
                "importance_model": figure.importance_model,
                "importance_error": figure.importance_error,
            }
        )
    (layout_dir / "figures.json").write_text(json.dumps(figure_rows, indent=2), encoding="utf-8")


def collect_tasked_module_text(
    summary_payload: dict[str, Any],
    *,
    max_chars_per_file: int,
    max_total_chars: int,
) -> tuple[list[dict[str, Any]], list[str], list[FigureRegion]]:
    items = summary_payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Tasked module summary does not contain an items list.")

    source_chunks: list[dict[str, Any]] = []
    figure_regions: list[FigureRegion] = []
    selected_sources: list[str] = []
    seen_canonical_files: set[str] = set()
    total_chars = 0

    for item in items:
        if total_chars >= max_total_chars:
            break
        if not isinstance(item, dict):
            continue

        item_dir = item.get("item_dir")
        if not isinstance(item_dir, str) or not item_dir.strip():
            continue

        root = Path(item_dir).expanduser().resolve()
        if not root.exists():
            continue

        for candidate in sorted(root.rglob("*")):
            if total_chars >= max_total_chars:
                break
            if not candidate.is_file():
                continue
            if candidate.name in IGNORED_SOURCE_FILENAMES:
                continue
            if candidate.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
                continue

            canonical_name = canonical_filename(candidate.name)
            if canonical_name in seen_canonical_files:
                continue

            allowed_chars = min(max_chars_per_file, max_total_chars - total_chars)
            if allowed_chars <= 0:
                break

            # Apply one shared evidence budget across every discovered file so
            # the prompt stays bounded even when a module contains many assets.
            file_chunks: list[dict[str, Any]] = []
            file_figures: list[FigureRegion] = []
            suffix = candidate.suffix.lower()

            if suffix == ".pdf":
                file_chunks, file_figures = _extract_pdf_layout_evidence(
                    candidate,
                    max_chars_for_file=allowed_chars,
                )
            elif suffix in {".png", ".jpg", ".jpeg"}:
                file_chunks, file_figures = _extract_image_layout_evidence(
                    candidate,
                    max_chars_for_file=allowed_chars,
                )
            else:
                file_chunks = _extract_non_pdf_source_chunks(candidate, max_chars=allowed_chars)

            if not file_chunks and not file_figures:
                continue

            source_chunks.extend(file_chunks)
            figure_regions.extend(file_figures)
            selected_sources.append(candidate.name)
            seen_canonical_files.add(canonical_name)
            total_chars += sum(len(str(chunk.get("text") or "")) for chunk in file_chunks)

    if not source_chunks and not figure_regions:
        raise RuntimeError("No parseable study content found in tasked module files.")

    return source_chunks, selected_sources, figure_regions


def _attach_figure_image_paths(
    questions: list[dict[str, Any]],
    figure_image_map: dict[tuple[str, int, str], str],
) -> None:
    for question in questions:
        source_file = str(question.get("source_file") or "").strip()
        source_page = normalize_positive_int(question.get("source_page"))
        figure_id = str(question.get("source_figure_id") or "").strip()
        if not source_file or source_page is None or not figure_id:
            question["source_figure_image_path"] = None
            continue
        question["source_figure_image_path"] = figure_image_map.get((source_file, source_page, figure_id))


def _classify_figures_with_gemma(
    figure_regions: list[FigureRegion],
    *,
    api_key: str,
    cache_path: Path | None = None,
    model: str = GOOGLE_IMPORTANCE_MODEL,
) -> tuple[list[FigureRegion], int]:
    if not figure_regions:
        return ([], 0)

    figure_rows: list[dict[str, Any]] = []
    for figure in figure_regions:
        figure.selected_for_prompt = False
        figure_rows.append(
            {
                "source_file": figure.source_file,
                "source_page": figure.source_page,
                "source_figure_id": figure.source_figure_id,
                "associated_text": figure.associated_text,
                "image_bytes": figure.image_bytes,
            }
        )

    decisions = classify_figures_with_google(
        figure_rows,
        api_key=api_key,
        model=model,
        cache_path=cache_path,
    )

    kept: list[FigureRegion] = []
    for figure, decision in zip(figure_regions, decisions):
        figure.importance_score = float(decision.confidence)
        figure.importance_level = "high" if decision.important else "low"
        figure.importance_type = decision.category or "other"
        figure.importance_reasons = [decision.reason] if decision.reason else []
        figure.importance_metrics = {}
        figure.importance_model = model
        figure.importance_error = decision.error
        figure.selected_for_prompt = bool(decision.important)
        if figure.selected_for_prompt:
            kept.append(figure)

    kept.sort(
        key=lambda figure: (
            figure.importance_score,
            figure.association_score,
            bbox_area(tuple(float(v) for v in figure.bbox)),
        ),
        reverse=True,
    )

    deleted_count = len(figure_regions) - len(kept)
    return kept, deleted_count
