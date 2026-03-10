from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types
from DocumentLayoutAnalysis.image_importance_google import classify_figures_with_google
from QG.Distractors.misconception_mining import build_or_load_module_misconceptions

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

SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx"}
IGNORED_SOURCE_FILENAMES = {"item.json", "item_links.txt", "submission.json"}

IMAGE_GROUNDED_RATIO = 0.25
LAYOUT_CACHE_ENABLED = True
LAYOUT_CACHE_DIRNAME = "layout"
MCQ_OUTPUT_DIRNAME = "mcq"

PDF_RENDER_SCALE = 2.0
MAX_PDF_PAGES_PER_FILE = 18
MAX_FIGURES_PER_FILE = 12
MAX_FIGURE_IMAGE_PARTS_FOR_PROMPT = 18
GOOGLE_IMPORTANCE_MODEL = "models/gemma-3-27b-it"
FIGURE_MAX_IMAGE_DIM = 900
YOLO_LAYOUT_MODEL_PATH = Path(__file__).resolve().parent / "yolov10x_best.pt"
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

ASSOCIATION_MAX_BLOCKS = 4
ASSOCIATION_MAX_CHARS = 900
CAPTION_VERTICAL_GAP_MAX = 140
NEAR_TEXT_DISTANCE_RATIO_MAX = 0.55

DEPTH_MIN_PASS_RATIO = 0.70
RECALL_MAX_RATIO = 0.20
DEPTH_PASS_SCORE = 3
BLUEPRINT_SKILLS = (
    "conceptual understanding",
    "application",
    "analysis",
    "calculation",
    "graph interpretation",
    "comparison",
    "policy reasoning",
    "error analysis",
)
BLUEPRINT_DIFFICULTIES = ("foundation", "standard", "challenge")
BLUEPRINT_ALIGNMENT_MIN_TOKEN_HITS = 2
BLUEPRINT_ALIGNMENT_MIN_PHRASE_HITS = 1
DIFFICULTY_CALIBRATION_PATH = Path(__file__).resolve().parent / "User" / "DifficultyCalibration" / "calibration.json"
MIN_DIFFICULTY_PILOT_SAMPLES = 4
DIFFICULTY_CALIBRATION_MARGIN = 0.10
DEFAULT_DIFFICULTY_DIALS: dict[str, dict[str, Any]] = {
    "foundation": {
        "target_correct_rate": 0.75,
        "steps": [1, 2],
        "concepts": [1, 2],
        "computation": [0, 1],
        "inference": [1, 2],
        "reading_load": [1, 2],
    },
    "standard": {
        "target_correct_rate": 0.60,
        "steps": [2, 3],
        "concepts": [2, 3],
        "computation": [1, 2],
        "inference": [2, 3],
        "reading_load": [2, 3],
    },
    "challenge": {
        "target_correct_rate": 0.40,
        "steps": [3, 4],
        "concepts": [3, 4],
        "computation": [1, 3],
        "inference": [3, 4],
        "reading_load": [3, 4],
    },
}
_BLUEPRINT_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "being",
    "between",
    "chapter",
    "class",
    "course",
    "curve",
    "demand",
    "effect",
    "from",
    "into",
    "market",
    "markets",
    "model",
    "module",
    "more",
    "notes",
    "other",
    "question",
    "supply",
    "than",
    "that",
    "their",
    "them",
    "these",
    "this",
    "unit",
    "using",
    "with",
}

LAST_GENERATION_REPORT: dict[str, Any] = {}
_YOLO_LAYOUT_MODEL: Any = None
_YOLO_LAYOUT_CLASS_NAMES: dict[int, str] = {}
_YOLO_LAYOUT_TARGET_CLASS_IDS: set[int] = set()


@dataclass
class TextRegion:
    source_file: str
    source_page: int
    bbox: tuple[float, float, float, float]
    text: str
    confidence: float
    source_kind: str


@dataclass
class FigureRegion:
    source_file: str
    source_page: int
    source_figure_id: str
    bbox: tuple[int, int, int, int]
    detection_label: str
    detection_confidence: float
    associated_text: str
    association_score: float
    image_bytes: bytes
    mime_type: str = "image/png"
    importance_score: float = 0.0
    importance_level: str = "low"
    importance_type: str = "unknown"
    importance_reasons: list[str] = field(default_factory=list)
    importance_metrics: dict[str, float] = field(default_factory=dict)
    selected_for_prompt: bool = False
    importance_model: str = ""
    importance_error: str | None = None


def load_tasked_items_summary(summary_path: str | Path) -> dict[str, Any]:
    return load_summary_payload(summary_path, label="Tasked module summary")


def load_summary_payload(summary_path: str | Path, *, label: str) -> dict[str, Any]:
    summary_file = Path(summary_path).expanduser().resolve()
    if not summary_file.exists():
        raise RuntimeError(f"{label} file not found: {summary_file}")

    try:
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {summary_file}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} has unexpected shape: {summary_file}")
    return payload


def _merge_tasked_and_submitted_items(
    tasked_summary_payload: dict[str, Any],
    submitted_summary_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    tasked_items = tasked_summary_payload.get("items")
    if not isinstance(tasked_items, list):
        raise RuntimeError("Tasked module summary does not include an items list.")

    merged_items: list[dict[str, Any]] = []
    seen_item_dirs: set[str] = set()

    for item in tasked_items:
        if not isinstance(item, dict):
            continue
        item_dir = item.get("item_dir")
        if not isinstance(item_dir, str) or not item_dir.strip():
            continue
        normalized_dir = str(Path(item_dir).expanduser().resolve())
        if normalized_dir in seen_item_dirs:
            continue
        seen_item_dirs.add(normalized_dir)
        merged_items.append({"item_dir": normalized_dir, "source_group": "tasked"})

    if isinstance(submitted_summary_payload, dict):
        submitted_rows = submitted_summary_payload.get("submitted_assignments")
        if isinstance(submitted_rows, list):
            for row in submitted_rows:
                if not isinstance(row, dict):
                    continue
                assignment_dir = row.get("assignment_dir")
                if not isinstance(assignment_dir, str) or not assignment_dir.strip():
                    continue
                normalized_dir = str(Path(assignment_dir).expanduser().resolve())
                if normalized_dir in seen_item_dirs:
                    continue
                seen_item_dirs.add(normalized_dir)
                merged_items.append({"item_dir": normalized_dir, "source_group": "submitted"})

    return {
        "items": merged_items,
    }


def _canonical_filename(filename: str) -> str:
    path = Path(filename)
    normalized_stem = re.sub(r"_[0-9]+$", "", path.stem.lower())
    return f"{normalized_stem}{path.suffix.lower()}"


def _normalize_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float):
        if value.is_integer() and value >= 1:
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            normalized = int(stripped)
            return normalized if normalized >= 1 else None
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_generated_text(text: str) -> str:
    cleaned = re.sub(
        r"\[(?:SRC|FIG_SRC|FIG_IMAGE|IMAGE_SRC)\s+[^\]]*\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def generate_targeted_remediation_questions(
    *,
    course_name: str,
    module_name: str,
    concept_label: str,
    difficulty_targets: list[str],
    source_file: str,
    source_page: int,
    source_text: str,
    evidence_lines: list[str] | None = None,
    misconception_id: str = "",
    misconception_summary: str = "",
    excluded_question_texts: list[str] | None = None,
    question_kind: str = "repair",
    model: str = "gemini-2.5-flash",
) -> list[dict[str, Any]]:
    requested_count = len([item for item in difficulty_targets if str(item).strip()])
    if requested_count <= 0:
        return []

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")

    trimmed_source_text = str(source_text or "").strip()
    if not trimmed_source_text:
        raise RuntimeError("Targeted remediation generation requires source_text.")
    if len(trimmed_source_text) > 12_000:
        trimmed_source_text = trimmed_source_text[:12_000].rsplit(" ", 1)[0].strip()

    cleaned_evidence = [str(line).strip() for line in (evidence_lines or []) if str(line).strip()]
    evidence_block = "\n".join(f"- {line}" for line in cleaned_evidence[:3]) or "- No scored evidence lines available."
    excluded_block = "\n".join(
        f"- {str(text).strip()}"
        for text in (excluded_question_texts or [])
        if str(text).strip()
    ) or "- none"

    tier_map = {
        "foundation": "rebuild definition / core relationship",
        "standard": "direct practice on the same concept",
        "challenge": "transfer / application to a new scenario",
    }
    target_lines = [
        f"{index + 1}. {difficulty}: {tier_map.get(str(difficulty).strip().lower(), 'direct practice')}"
        for index, difficulty in enumerate(difficulty_targets)
    ]

    prompt = (
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Concept: {concept_label}\n"
        f"Question kind: {question_kind}\n"
        f"Source file: {source_file}\n"
        f"Source page: {source_page}\n"
        f"Misconception ID: {misconception_id or 'none'}\n"
        f"Misconception summary: {misconception_summary or 'none'}\n\n"
        "Generate targeted remediation multiple-choice questions grounded only in the provided source text.\n"
        "Keep the questions focused on the same concept as the trigger question.\n"
        "Do not repeat or closely paraphrase excluded questions.\n"
        "Each question must have exactly four choices and one correct answer.\n"
        "Include a concise 1-2 sentence why explanation.\n"
        "Return JSON only with this shape:\n"
        "{\"questions\":[{\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"answer\":\"A\",\"why\":\"...\",\"source_file\":\"...\",\"source_page\":1,"
        "\"source_figure_id\":\"\",\"objective_label\":\"...\",\"objective_topic\":\"...\","
        "\"objective_skill\":\"...\",\"objective_difficulty\":\"foundation\",\"misconception_id\":\"...\"}]}\n\n"
        f"Difficulty targets ({requested_count} total):\n" + "\n".join(target_lines) + "\n\n"
        "Evidence lines:\n"
        f"{evidence_block}\n\n"
        "Excluded question texts:\n"
        f"{excluded_block}\n\n"
        "Source text:\n"
        f"{trimmed_source_text}"
    )

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You generate concise, source-grounded remediation MCQs. "
                    "Return JSON only and do not repeat excluded questions."
                ),
                response_mime_type="application/json",
                temperature=0.25,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Targeted remediation Gemini request failed: {exc}") from exc

    parsed_payload = response.parsed
    if parsed_payload is None:
        response_text = response.text if isinstance(response.text, str) else ""
        if not response_text.strip():
            raise RuntimeError("Targeted remediation generation returned an empty response.")
        try:
            parsed_payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            parsed_payload = _extract_json_object(response_text)
            if not isinstance(parsed_payload, dict):
                raise RuntimeError("Targeted remediation generation did not return valid JSON.") from exc

    rows = parsed_payload.get("questions") if isinstance(parsed_payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Targeted remediation generation returned no questions array.")

    normalized_excluded = {_normalize_text(text) for text in (excluded_question_texts or []) if str(text).strip()}
    seen_questions: set[str] = set()
    normalized_questions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question_text = _clean_generated_text(str(row.get("question") or "").strip())
        if not question_text:
            continue
        normalized_text = _normalize_text(question_text)
        if normalized_text in normalized_excluded or normalized_text in seen_questions:
            continue

        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            continue
        cleaned_choices = [_clean_generated_text(str(choice or "").strip()) for choice in choices[:4]]
        if any(not choice for choice in cleaned_choices):
            continue
        answer = str(row.get("answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue

        target_difficulty = (
            str(difficulty_targets[min(index, requested_count - 1)] or "").strip().lower() or "standard"
        )
        objective_skill = str(row.get("objective_skill") or "").strip() or (
            "conceptual understanding" if target_difficulty == "foundation"
            else "analysis" if target_difficulty == "challenge"
            else "application"
        )
        normalized_questions.append(
            {
                "question": question_text,
                "choices": cleaned_choices,
                "answer": answer,
                "why": _clean_generated_text(str(row.get("why") or "").strip()),
                "source_file": str(row.get("source_file") or source_file).strip() or source_file,
                "source_page": _normalize_positive_int(row.get("source_page")) or source_page,
                "source_figure_id": str(row.get("source_figure_id") or "").strip(),
                "objective_label": str(row.get("objective_label") or concept_label).strip() or concept_label,
                "objective_topic": str(row.get("objective_topic") or concept_label).strip() or concept_label,
                "objective_skill": objective_skill,
                "objective_difficulty": target_difficulty,
                "misconception_id": str(row.get("misconception_id") or misconception_id).strip(),
            }
        )
        seen_questions.add(normalized_text)
        if len(normalized_questions) >= requested_count:
            break

    if len(normalized_questions) < requested_count:
        raise RuntimeError(
            f"Targeted remediation generation returned {len(normalized_questions)} usable question(s); "
            f"needed {requested_count}."
        )

    return normalized_questions


def generate_batch_remediation_questions(
    *,
    course_name: str,
    module_name: str,
    remediation_specs: list[dict[str, Any]],
    question_kind: str = "repair_test",
    model: str = "gemini-2.5-flash",
) -> list[dict[str, Any]]:
    normalized_specs = [spec for spec in remediation_specs if isinstance(spec, dict)]
    if not normalized_specs:
        return []

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")

    spec_blocks: list[str] = []
    for index, spec in enumerate(normalized_specs, start=1):
        source_text = str(spec.get("source_text") or "").strip()
        if len(source_text) > 6000:
            source_text = source_text[:6000].rsplit(" ", 1)[0].strip()
        evidence_lines = [
            str(line).strip()
            for line in (spec.get("evidence_lines") or [])
            if str(line).strip()
        ][:3]
        excluded = [
            str(text).strip()
            for text in (spec.get("excluded_question_texts") or [])
            if str(text).strip()
        ][:12]
        spec_blocks.append(
            "\n".join(
                [
                    f"SPEC {index}",
                    f"- concept_label: {str(spec.get('concept_label') or '').strip()}",
                    f"- difficulty_target: {str(spec.get('difficulty_target') or 'standard').strip()}",
                    f"- source_file: {str(spec.get('source_file') or 'Unknown source').strip()}",
                    f"- source_page: {int(spec.get('source_page') or 1)}",
                    f"- misconception_id: {str(spec.get('misconception_id') or '').strip() or 'none'}",
                    f"- misconception_summary: {str(spec.get('misconception_summary') or '').strip() or 'none'}",
                    "- evidence_lines:",
                    *([f"  - {line}" for line in evidence_lines] or ["  - none"]),
                    "- excluded_question_texts:",
                    *([f"  - {text}" for text in excluded] or ["  - none"]),
                    "- source_text:",
                    source_text or "No source text available.",
                ]
            )
        )

    prompt = (
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Question kind: {question_kind}\n"
        f"Generate exactly {len(normalized_specs)} remediation multiple-choice questions.\n"
        "Produce exactly one question for each numbered spec below.\n"
        "Return JSON only with this shape:\n"
        "{\"questions\":[{\"spec_index\":1,\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],"
        "\"answer\":\"A\",\"why\":\"...\",\"source_file\":\"...\",\"source_page\":1,"
        "\"source_figure_id\":\"\",\"objective_label\":\"...\",\"objective_topic\":\"...\","
        "\"objective_skill\":\"...\",\"objective_difficulty\":\"standard\",\"misconception_id\":\"...\"}]}\n"
        "Use each spec's source text only. Do not repeat or closely paraphrase excluded question texts.\n"
        "Keep each question grounded in its spec's concept and difficulty target.\n\n"
        + "\n\n".join(spec_blocks)
    )

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You generate one source-grounded remediation MCQ per spec and return JSON only."
                ),
                response_mime_type="application/json",
                temperature=0.25,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Batch remediation Gemini request failed: {exc}") from exc

    parsed_payload = response.parsed
    if parsed_payload is None:
        response_text = response.text if isinstance(response.text, str) else ""
        if not response_text.strip():
            raise RuntimeError("Batch remediation generation returned an empty response.")
        try:
            parsed_payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            parsed_payload = _extract_json_object(response_text)
            if not isinstance(parsed_payload, dict):
                raise RuntimeError("Batch remediation generation did not return valid JSON.") from exc

    rows = parsed_payload.get("questions") if isinstance(parsed_payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Batch remediation generation returned no questions array.")

    results_by_spec: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        spec_index = _normalize_positive_int(row.get("spec_index"))
        if spec_index is None or spec_index < 1 or spec_index > len(normalized_specs):
            continue
        question_text = _clean_generated_text(str(row.get("question") or "").strip())
        choices = row.get("choices")
        if not question_text or not isinstance(choices, list) or len(choices) != 4:
            continue
        answer = str(row.get("answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        cleaned_choices = [_clean_generated_text(str(choice or "").strip()) for choice in choices[:4]]
        if any(not choice for choice in cleaned_choices):
            continue
        spec = normalized_specs[spec_index - 1]
        results_by_spec[spec_index] = {
            "spec_index": spec_index,
            "question": question_text,
            "choices": cleaned_choices,
            "answer": answer,
            "why": _clean_generated_text(str(row.get("why") or "").strip()),
            "source_file": str(row.get("source_file") or spec.get("source_file") or "Unknown source").strip(),
            "source_page": _normalize_positive_int(row.get("source_page")) or int(spec.get("source_page") or 1),
            "source_figure_id": str(row.get("source_figure_id") or "").strip(),
            "objective_label": str(row.get("objective_label") or spec.get("concept_label") or "").strip(),
            "objective_topic": str(row.get("objective_topic") or spec.get("concept_label") or "").strip(),
            "objective_skill": str(row.get("objective_skill") or "application").strip(),
            "objective_difficulty": str(row.get("objective_difficulty") or spec.get("difficulty_target") or "standard").strip(),
            "misconception_id": str(row.get("misconception_id") or spec.get("misconception_id") or "").strip(),
        }

    if len(results_by_spec) < len(normalized_specs):
        raise RuntimeError(
            f"Batch remediation generation returned {len(results_by_spec)} usable question(s); "
            f"needed {len(normalized_specs)}."
        )

    return [results_by_spec[index] for index in range(1, len(normalized_specs) + 1)]


def _clean_choice_text(text: str) -> str:
    cleaned = _clean_generated_text(text)
    cleaned = re.sub(r"^\s*\(?[A-D]\)?\s*[\)\.\:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[A-D]\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _sanitize_path_part(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return sanitized[:80] if sanitized else fallback


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    b_area = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    *,
    iou_threshold: float,
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            idx
            for idx in order
            if _bbox_iou(boxes[current], boxes[idx]) < iou_threshold
        ]
    return keep


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


def _build_text_mask(text_regions: list[TextRegion], image_shape: tuple[int, int, int]) -> Any:
    if np is None or cv2 is None:
        raise RuntimeError("numpy and opencv-python are required for text masking.")

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in text_regions:
        x0, y0, x1, y1 = region.bbox
        xi0 = max(0, min(width - 1, int(round(x0))))
        yi0 = max(0, min(height - 1, int(round(y0))))
        xi1 = max(0, min(width, int(round(x1))))
        yi1 = max(0, min(height, int(round(y1))))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        cv2.rectangle(mask, (xi0, yi0), (xi1, yi1), 255, thickness=-1)

    return mask


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
    return model, normalized_names, target_class_ids


def _detect_figure_boxes(
    image_bgr: Any,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    if cv2 is None or np is None:
        return []

    model, names, target_class_ids = _load_yolo_layout_detector()
    try:
        predictions = model.predict(
            source=image_bgr,
            conf=YOLO_LAYOUT_CONFIDENCE,
            iou=YOLO_LAYOUT_IOU,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"YOLO detection failed on rendered PDF page: {exc}") from exc

    if not isinstance(predictions, list) or not predictions:
        return []

    height, width = image_bgr.shape[:2]
    detections: list[tuple[tuple[int, int, int, int], str, float]] = []
    for prediction in predictions:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue

        xyxy_attr = getattr(boxes, "xyxy", None)
        cls_attr = getattr(boxes, "cls", None)
        conf_attr = getattr(boxes, "conf", None)
        if xyxy_attr is None or cls_attr is None or conf_attr is None:
            continue

        xyxy_values = xyxy_attr.cpu().numpy() if hasattr(xyxy_attr, "cpu") else xyxy_attr
        cls_values = cls_attr.cpu().numpy() if hasattr(cls_attr, "cpu") else cls_attr
        conf_values = conf_attr.cpu().numpy() if hasattr(conf_attr, "cpu") else conf_attr

        for box_raw, class_raw, confidence_raw in zip(xyxy_values, cls_values, conf_values):
            try:
                class_id = int(class_raw)
            except (ValueError, TypeError):
                continue
            if class_id not in target_class_ids:
                continue

            if not isinstance(box_raw, (list, tuple)) and not hasattr(box_raw, "__iter__"):
                continue
            box_values = [float(value) for value in box_raw]
            if len(box_values) < 4:
                continue
            x0 = max(0, min(width - 1, int(round(box_values[0]))))
            y0 = max(0, min(height - 1, int(round(box_values[1]))))
            x1 = max(0, min(width, int(round(box_values[2]))))
            y1 = max(0, min(height, int(round(box_values[3]))))
            if x1 <= x0 or y1 <= y0:
                continue

            box_width = x1 - x0
            box_height = y1 - y0
            if box_width < FIGURE_MIN_WIDTH or box_height < FIGURE_MIN_HEIGHT:
                continue

            confidence = float(confidence_raw)
            label = names.get(class_id, f"class_{class_id}").strip()
            detections.append(((x0, y0, x1, y1), label, confidence))

    detections.sort(
        key=lambda item: (
            (item[0][1] + item[0][3]) / 2.0,
            (item[0][0] + item[0][2]) / 2.0,
            -item[2],
        )
    )
    return detections


def _extract_page_text_chunk(text_regions: list[TextRegion], *, max_chars: int) -> str:
    merged = "\n".join(region.text for region in text_regions if region.text.strip()).strip()
    if not merged:
        return ""
    return merged[:max_chars].strip()


def _associate_text_to_figure(
    figure_bbox: tuple[int, int, int, int],
    text_regions: list[TextRegion],
    *,
    page_width: int,
    page_height: int,
) -> tuple[str, float]:
    if not text_regions:
        return "", 0.0

    fx0, fy0, fx1, fy1 = tuple(float(v) for v in figure_bbox)
    figure_rect = (fx0, fy0, fx1, fy1)
    f_center = _rect_center(figure_rect)
    page_diag = max(1.0, math.hypot(page_width, page_height))

    scored_rows: list[tuple[float, TextRegion]] = []
    for region in text_regions:
        rx0, ry0, rx1, ry1 = region.bbox
        region_rect = (rx0, ry0, rx1, ry1)

        overlap_x_ratio = _horizontal_overlap_ratio(figure_rect, region_rect)

        vertical_gap = 0.0
        if ry1 < fy0:
            vertical_gap = fy0 - ry1
        elif ry0 > fy1:
            vertical_gap = ry0 - fy1

        caption_like = overlap_x_ratio >= 0.20 and vertical_gap <= CAPTION_VERTICAL_GAP_MAX

        r_center = _rect_center(region_rect)
        center_distance_ratio = math.hypot(r_center[0] - f_center[0], r_center[1] - f_center[1]) / page_diag

        intersection_w = max(0.0, min(fx1, rx1) - max(fx0, rx0))
        intersection_h = max(0.0, min(fy1, ry1) - max(fy0, ry0))
        intersects = intersection_w > 0 and intersection_h > 0

        if center_distance_ratio > NEAR_TEXT_DISTANCE_RATIO_MAX and not caption_like and not intersects:
            continue

        score = 0.0
        score += overlap_x_ratio * 2.0
        score += max(0.0, 1.0 - center_distance_ratio)
        if caption_like:
            score += 2.0
        if intersects:
            score += 0.7
        if region.source_kind == "native":
            score += 0.2

        if score <= 0.15:
            continue
        scored_rows.append((score, region))

    if not scored_rows:
        return "", 0.0

    scored_rows.sort(key=lambda current: current[0], reverse=True)

    picked: list[str] = []
    seen: set[str] = set()
    best_score = scored_rows[0][0]
    total_chars = 0
    for score, region in scored_rows:
        text = region.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if len(picked) >= ASSOCIATION_MAX_BLOCKS:
            break
        remaining = ASSOCIATION_MAX_CHARS - total_chars
        if remaining <= 0:
            break
        text_part = text[:remaining].strip()
        if not text_part:
            continue
        picked.append(text_part)
        total_chars += len(text_part)

    return " ".join(picked).strip(), best_score


def _build_figure_regions_from_detections(
    *,
    image_bgr: Any,
    source_file: str,
    source_page: int,
    detected_rows: list[tuple[tuple[int, int, int, int], str, float]],
    text_regions: list[TextRegion],
) -> list[FigureRegion]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for figure extraction.")

    if not detected_rows:
        return []

    figure_regions: list[FigureRegion] = []
    page_height, page_width = image_bgr.shape[:2]
    page_figure_counter = 0

    for detection in detected_rows:
        if page_figure_counter >= MAX_FIGURES_PER_FILE:
            break

        bbox, detected_label, detected_confidence = detection
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        crop_height, crop_width = crop.shape[:2]
        max_dim = max(crop_height, crop_width)
        if max_dim > FIGURE_MAX_IMAGE_DIM:
            scale = FIGURE_MAX_IMAGE_DIM / float(max_dim)
            resized_width = max(1, int(round(crop_width * scale)))
            resized_height = max(1, int(round(crop_height * scale)))
            crop = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        encoded_ok, encoded_png = cv2.imencode(".png", crop)
        if not encoded_ok:
            continue

        page_figure_counter += 1
        figure_id = f"fig#{page_figure_counter}"

        associated_text, association_score = _associate_text_to_figure(
            bbox,
            text_regions,
            page_width=page_width,
            page_height=page_height,
        )
        if not associated_text:
            associated_text = "No nearby text extracted."

        figure_regions.append(
            FigureRegion(
                source_file=source_file,
                source_page=source_page,
                source_figure_id=figure_id,
                bbox=bbox,
                detection_label=detected_label,
                detection_confidence=float(detected_confidence),
                associated_text=associated_text,
                association_score=association_score,
                image_bytes=encoded_png.tobytes(),
                importance_score=0.0,
                importance_level="unknown",
                importance_type="unknown",
                importance_reasons=[],
                importance_metrics={},
            )
        )

    return figure_regions


def _extract_pdf_layout_evidence(
    file_path: Path,
    *,
    max_chars_for_file: int,
) -> tuple[list[dict[str, Any]], list[FigureRegion]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF layout detection.")

    doc = fitz.open(str(file_path))
    source_chunks: list[dict[str, Any]] = []
    figure_regions: list[FigureRegion] = []

    total_chars = 0
    try:
        for page_index in range(min(doc.page_count, MAX_PDF_PAGES_PER_FILE)):
            page = doc.load_page(page_index)
            source_page = page_index + 1

            image_bgr, scale_x, scale_y = _render_page_to_bgr(page, scale=PDF_RENDER_SCALE)
            native_regions = _pdf_block_regions(
                page,
                source_file=file_path.name,
                source_page=source_page,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            native_char_count = sum(len(region.text) for region in native_regions)

            ocr_regions: list[TextRegion] = []
            if native_char_count < NATIVE_TEXT_CHAR_THRESHOLD:
                ocr_regions = _parse_ocr_line_regions(
                    image_bgr,
                    source_file=file_path.name,
                    source_page=source_page,
                )

            chosen_text_regions = native_regions
            if native_char_count < NATIVE_TEXT_CHAR_THRESHOLD and ocr_regions:
                chosen_text_regions = ocr_regions

            if total_chars < max_chars_for_file and chosen_text_regions:
                remaining = max_chars_for_file - total_chars
                page_text = _extract_page_text_chunk(chosen_text_regions, max_chars=remaining)
                if page_text:
                    source_chunks.append(
                        {
                            "source_file": file_path.name,
                            "source_page": source_page,
                            "text": page_text,
                        }
                    )
                    total_chars += len(page_text)

            text_association_regions = chosen_text_regions if chosen_text_regions else native_regions
            detected_rows = _detect_figure_boxes(image_bgr)
            figure_regions.extend(
                _build_figure_regions_from_detections(
                    image_bgr=image_bgr,
                    source_file=file_path.name,
                    source_page=source_page,
                    detected_rows=detected_rows,
                    text_regions=text_association_regions,
                )
            )

    finally:
        doc.close()

    return source_chunks, figure_regions


def _load_image_file_bgr(file_path: Path) -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("numpy and opencv-python are required for image layout detection.")

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
        raise RuntimeError(f"Could not load image file: {file_path}")
    return image_bgr


def _extract_image_layout_evidence(
    file_path: Path,
    *,
    max_chars_for_file: int,
) -> tuple[list[dict[str, Any]], list[FigureRegion]]:
    image_bgr = _load_image_file_bgr(file_path)
    source_page = 1

    ocr_regions = _parse_ocr_line_regions(
        image_bgr,
        source_file=file_path.name,
        source_page=source_page,
    )

    source_chunks: list[dict[str, Any]] = []
    if ocr_regions and max_chars_for_file > 0:
        page_text = _extract_page_text_chunk(ocr_regions, max_chars=max_chars_for_file)
        if page_text:
            source_chunks.append(
                {
                    "source_file": file_path.name,
                    "source_page": source_page,
                    "text": page_text,
                }
            )

    detected_rows = _detect_figure_boxes(image_bgr)
    figure_regions = _build_figure_regions_from_detections(
        image_bgr=image_bgr,
        source_file=file_path.name,
        source_page=source_page,
        detected_rows=detected_rows,
        text_regions=ocr_regions,
    )
    return source_chunks, figure_regions


def _extract_text_file(file_path: Path, max_chars: int) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _extract_docx_text(file_path: Path, max_chars: int) -> str:
    try:
        document = Document(str(file_path))
    except Exception:  # noqa: BLE001
        return ""

    lines = [line.text for line in document.paragraphs if line.text and line.text.strip()]
    return "\n".join(lines)[:max_chars]


def _extract_non_pdf_source_chunks(file_path: Path, max_chars: int) -> list[dict[str, Any]]:
    extension = file_path.suffix.lower()
    if extension in {".txt", ".md"}:
        text = _extract_text_file(file_path, max_chars=max_chars).strip()
    elif extension == ".docx":
        text = _extract_docx_text(file_path, max_chars=max_chars).strip()
    else:
        text = ""

    if not text:
        return []

    return [
        {
            "source_file": file_path.name,
            "source_page": 1,
            "text": text,
        }
    ]


def _build_tagged_study_text(
    source_chunks: list[dict[str, Any]],
    figure_regions: list[FigureRegion],
) -> str:
    lines: list[str] = []

    for chunk in source_chunks:
        source_file = str(chunk.get("source_file") or "").strip()
        source_page = _normalize_positive_int(chunk.get("source_page"))
        source_text = str(chunk.get("text") or "").strip()
        if not source_file or source_page is None or not source_text:
            continue

        safe_file = source_file.replace('"', "'")
        lines.append(f'[SRC file="{safe_file}" page={source_page}] {source_text}')

    for figure in figure_regions:
        safe_file = figure.source_file.replace('"', "'")
        assoc_text = figure.associated_text.strip() or "No nearby text extracted."
        importance_hint = (
            f"(detected={figure.detection_label}; det_conf={figure.detection_confidence:.2f}; "
            f"figure_type={figure.importance_type}; importance={figure.importance_level}; "
            f"score={figure.importance_score:.2f})"
        )
        lines.append(
            f'[FIG_SRC file="{safe_file}" page={figure.source_page} fig="{figure.source_figure_id}"] '
            f"{importance_hint} {assoc_text}"
        )

    return "\n\n".join(lines).strip()


def _layout_cache_dir_from_summary_path(tasked_items_summary_path: Path) -> Path:
    summary_parent = tasked_items_summary_path.parent
    if summary_parent.name == "tasked":
        module_dir = summary_parent.parent
    else:
        module_dir = summary_parent
    return module_dir / LAYOUT_CACHE_DIRNAME


def _mcq_output_dir_from_summary_path(tasked_items_summary_path: Path) -> Path:
    summary_parent = tasked_items_summary_path.parent
    if summary_parent.name == "tasked":
        module_dir = summary_parent.parent
    else:
        module_dir = summary_parent
    return module_dir / MCQ_OUTPUT_DIRNAME


def _load_figure_image_map(tasked_items_summary_path: Path) -> dict[tuple[str, int, str], str]:
    layout_dir = _layout_cache_dir_from_summary_path(tasked_items_summary_path)
    summary_path = layout_dir / "layout_summary.json"
    if not summary_path.exists():
        return {}

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    if not isinstance(payload, dict):
        return {}

    figures = payload.get("figures")
    if not isinstance(figures, list):
        return {}

    mapping: dict[tuple[str, int, str], str] = {}
    for row in figures:
        if not isinstance(row, dict):
            continue

        source_file = str(row.get("source_file") or "").strip()
        source_page = _normalize_positive_int(row.get("source_page"))
        source_figure_id = str(row.get("source_figure_id") or "").strip()
        image_path_raw = str(row.get("image_path") or "").strip()
        if not source_file or source_page is None or not source_figure_id or not image_path_raw:
            continue

        image_path = Path(image_path_raw)
        if not image_path.is_absolute():
            image_path = (summary_path.parent / image_path).resolve()

        mapping[(source_file, source_page, source_figure_id)] = str(image_path)

    return mapping


def _write_layout_cache(
    *,
    layout_dir: Path,
    summary_payload: dict[str, Any],
    source_chunks: list[dict[str, Any]],
    figure_regions: list[FigureRegion],
) -> None:
    layout_dir.mkdir(parents=True, exist_ok=True)
    figures_root = layout_dir / "figures"
    if figures_root.exists():
        shutil.rmtree(figures_root)
    figures_root.mkdir(parents=True, exist_ok=True)

    figure_rows: list[dict[str, Any]] = []
    for figure in figure_regions:
        file_slug = _sanitize_path_part(Path(figure.source_file).stem, "file")
        file_dir = figures_root / file_slug
        file_dir.mkdir(parents=True, exist_ok=True)

        figure_num_match = re.search(r"(\d+)$", figure.source_figure_id)
        figure_num = figure_num_match.group(1) if figure_num_match else figure.source_figure_id
        image_name = f"p{figure.source_page}_fig{figure_num}.png"
        image_path = file_dir / image_name
        image_path.write_bytes(figure.image_bytes)

        row = {
            "source_file": figure.source_file,
            "source_page": figure.source_page,
            "source_figure_id": figure.source_figure_id,
            "bbox": [int(v) for v in figure.bbox],
            "detection_label": figure.detection_label,
            "detection_confidence": round(float(figure.detection_confidence), 4),
            "association_score": round(float(figure.association_score), 4),
            "associated_text": figure.associated_text[:800],
            "importance_score": round(float(figure.importance_score), 4),
            "importance_level": figure.importance_level,
            "importance_type": figure.importance_type,
            "importance_reasons": figure.importance_reasons[:5],
            "importance_metrics": figure.importance_metrics,
            "importance_model": figure.importance_model,
            "importance_error": figure.importance_error,
            "selected_for_prompt": figure.selected_for_prompt,
            "image_path": str(image_path),
        }
        figure_rows.append(row)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": summary_payload.get("course"),
        "module": summary_payload.get("module"),
        "thresholds": {
            "image_grounded_ratio": IMAGE_GROUNDED_RATIO,
            "pdf_render_scale": PDF_RENDER_SCALE,
            "figure_min_area": FIGURE_MIN_AREA,
            "figure_min_width": FIGURE_MIN_WIDTH,
            "figure_min_height": FIGURE_MIN_HEIGHT,
            "figure_min_fill_ratio": FIGURE_MIN_FILL_RATIO,
            "figure_max_page_coverage": FIGURE_MAX_PAGE_COVERAGE,
            "figure_max_image_dim": FIGURE_MAX_IMAGE_DIM,
            "yolo_layout_model_path": str(YOLO_LAYOUT_MODEL_PATH),
            "yolo_layout_confidence": YOLO_LAYOUT_CONFIDENCE,
            "yolo_layout_iou": YOLO_LAYOUT_IOU,
            "yolo_layout_target_labels": sorted(YOLO_LAYOUT_TARGET_LABELS),
            "importance_provider": "google",
            "importance_model": GOOGLE_IMPORTANCE_MODEL,
            "ocr_word_confidence_min": OCR_WORD_CONFIDENCE_MIN,
            "native_text_char_threshold": NATIVE_TEXT_CHAR_THRESHOLD,
        },
        "text_chunk_count": len(source_chunks),
        "figure_count": len(figure_rows),
        "selected_figure_count": sum(1 for figure in figure_rows if figure.get("selected_for_prompt") is True),
        "figures": figure_rows,
    }
    summary_path = layout_dir / "layout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def collect_tasked_module_text(
    summary_payload: dict[str, Any],
    *,
    max_chars_per_file: int = 6000,
    max_total_chars: int = 50_000,
) -> tuple[list[dict[str, Any]], list[str], list[FigureRegion]]:
    items = summary_payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Tasked module summary does not include an items list.")

    seen_canonical_files: set[str] = set()
    selected_sources: list[str] = []
    source_chunks: list[dict[str, Any]] = []
    figure_regions: list[FigureRegion] = []
    total_chars = 0

    for item in items:
        if not isinstance(item, dict):
            continue

        item_dir_raw = item.get("item_dir")
        if not isinstance(item_dir_raw, str) or not item_dir_raw.strip():
            continue

        item_dir = Path(item_dir_raw)
        if not item_dir.exists() or not item_dir.is_dir():
            continue

        for candidate in sorted(item_dir.iterdir(), key=lambda path: path.name.lower()):
            if not candidate.is_file():
                continue
            if candidate.name.lower() in IGNORED_SOURCE_FILENAMES:
                continue
            if candidate.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
                continue

            canonical_name = _canonical_filename(candidate.name)
            if canonical_name in seen_canonical_files:
                continue

            remaining_chars = max_total_chars - total_chars
            if remaining_chars <= 0:
                break

            allowed_chars = min(max_chars_per_file, remaining_chars)

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


def _build_questions_schema(question_count: int) -> dict[str, Any]:
    diagnostic_option_schema = {
        "type": "object",
        "properties": {
            "misconception_id": {"type": "string"},
            "misconception_label": {"type": "string"},
            "why_student_might_pick": {"type": "string"},
            "why_wrong": {"type": "string"},
        },
        "required": [
            "misconception_id",
            "misconception_label",
            "why_student_might_pick",
            "why_wrong",
        ],
    }
    diagnostic_nullable_schema = {
        "anyOf": [
            diagnostic_option_schema,
            {"type": "null"},
        ]
    }

    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": question_count,
                "maxItems": question_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "blueprint_slot_id": {"type": "string"},
                        "objective_id": {"type": "string"},
                        "objective_label": {"type": "string"},
                        "objective_topic": {"type": "string"},
                        "objective_skill": {"type": "string"},
                        "objective_difficulty": {"type": "string"},
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                        },
                        "source_file": {"type": "string"},
                        "source_page": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "source_figure_id": {"type": "string"},
                        "is_image_grounded": {"type": "boolean"},
                        "distractor_diagnostics": {
                            "type": "object",
                            "properties": {
                                "A": diagnostic_nullable_schema,
                                "B": diagnostic_nullable_schema,
                                "C": diagnostic_nullable_schema,
                                "D": diagnostic_nullable_schema,
                            },
                            "required": ["A", "B", "C", "D"],
                        },
                    },
                    "required": [
                        "question",
                        "choices",
                        "blueprint_slot_id",
                        "objective_id",
                        "objective_label",
                        "objective_topic",
                        "objective_skill",
                        "objective_difficulty",
                        "answer",
                        "source_file",
                        "source_page",
                        "source_figure_id",
                        "is_image_grounded",
                        "distractor_diagnostics",
                    ],
                },
            }
        },
        "required": ["questions"],
    }


def _difficulty_guidance(difficulty_profile: str) -> tuple[str, str]:
    normalized = difficulty_profile.strip().lower()
    if normalized == "calculation_heavy":
        return (
            "calculation_heavy",
            (
                "Bias toward quantitative and graph/data interpretation questions when evidence supports it. "
                "Require multi-step setup and reasoning, not one-step arithmetic."
            ),
        )
    if normalized == "concept_synthesis":
        return (
            "concept_synthesis",
            (
                "Bias toward cross-concept synthesis, policy reasoning, and comparative analysis. "
                "Questions should connect multiple ideas from the module."
            ),
        )

    return (
        "exam_mixed",
        (
            "Use AP exam-level rigor: mostly multi-step concept+application questions, "
            "with a small minority of foundational checks. Avoid definition-only recall."
        ),
    )


def _normalize_blueprint_enum(value: Any, aliases: dict[str, str], fallback: str = "") -> str:
    cleaned = _normalize_text(str(value or "")).lower()
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return fallback
    return aliases.get(cleaned, fallback)


_BLUEPRINT_SKILL_ALIASES = {
    "conceptual": "conceptual understanding",
    "conceptual understanding": "conceptual understanding",
    "conceptual reasoning": "conceptual understanding",
    "application": "application",
    "applied reasoning": "application",
    "analysis": "analysis",
    "analytical reasoning": "analysis",
    "calculation": "calculation",
    "quantitative": "calculation",
    "graph interpretation": "graph interpretation",
    "graphing": "graph interpretation",
    "graph": "graph interpretation",
    "comparison": "comparison",
    "compare": "comparison",
    "policy reasoning": "policy reasoning",
    "policy": "policy reasoning",
    "error analysis": "error analysis",
    "misconception analysis": "error analysis",
}

_BLUEPRINT_DIFFICULTY_ALIASES = {
    "foundation": "foundation",
    "foundational": "foundation",
    "basic": "foundation",
    "standard": "standard",
    "core": "standard",
    "medium": "standard",
    "challenge": "challenge",
    "challenging": "challenge",
    "advanced": "challenge",
    "hard": "challenge",
}


def _normalize_blueprint_skill(value: Any) -> str:
    return _normalize_blueprint_enum(value, _BLUEPRINT_SKILL_ALIASES)


def _normalize_blueprint_difficulty(value: Any) -> str:
    return _normalize_blueprint_enum(value, _BLUEPRINT_DIFFICULTY_ALIASES)


def _derive_blueprint_key_terms(objective_label: str, topic: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.split(r"[,;/]| and | or ", f"{objective_label}; {topic}", flags=re.IGNORECASE):
        term = _normalize_text(raw_term).strip(" -")
        if len(term) < 4:
            continue
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(term)
        if len(terms) >= 6:
            break
    return terms


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _load_difficulty_dials(
    calibration_path: str | Path = DIFFICULTY_CALIBRATION_PATH,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dials = {
        difficulty: {
            "target_correct_rate": float(config["target_correct_rate"]),
            "steps": list(config["steps"]),
            "concepts": list(config["concepts"]),
            "computation": list(config["computation"]),
            "inference": list(config["inference"]),
            "reading_load": list(config["reading_load"]),
        }
        for difficulty, config in DEFAULT_DIFFICULTY_DIALS.items()
    }
    metadata: dict[str, Any] = {
        "source": "default",
        "path": str(Path(calibration_path).expanduser().resolve()),
        "observed_rates": {},
        "sample_counts": {},
    }

    path = Path(calibration_path).expanduser().resolve()
    if not path.exists():
        return dials, metadata

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return dials, metadata
    if not isinstance(payload, dict):
        return dials, metadata

    target_correct_rates = payload.get("target_correct_rates")
    applied_override = False
    if isinstance(target_correct_rates, dict):
        for difficulty, raw_target in target_correct_rates.items():
            normalized = _normalize_blueprint_difficulty(difficulty)
            if not normalized or normalized not in dials:
                continue
            try:
                target = float(raw_target)
            except Exception:  # noqa: BLE001
                continue
            if 0.05 <= target <= 0.95:
                if abs(target - float(DEFAULT_DIFFICULTY_DIALS[normalized]["target_correct_rate"])) > 0.0001:
                    applied_override = True
                dials[normalized]["target_correct_rate"] = round(target, 3)

    pilot_results = payload.get("pilot_results")
    observed_rates: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    if isinstance(pilot_results, list):
        grouped: dict[str, list[int]] = {difficulty: [] for difficulty in dials}
        for row in pilot_results:
            if not isinstance(row, dict):
                continue
            difficulty = _normalize_blueprint_difficulty(row.get("difficulty"))
            if difficulty not in grouped:
                continue
            correct_value = row.get("correct")
            if isinstance(correct_value, bool):
                grouped[difficulty].append(1 if correct_value else 0)
                continue
            if isinstance(correct_value, (int, float)):
                grouped[difficulty].append(1 if float(correct_value) >= 0.5 else 0)
        for difficulty, values in grouped.items():
            if not values:
                continue
            sample_counts[difficulty] = len(values)
            observed_rates[difficulty] = round(sum(values) / len(values), 3)
            if len(values) < MIN_DIFFICULTY_PILOT_SAMPLES:
                continue
            delta = observed_rates[difficulty] - float(dials[difficulty]["target_correct_rate"])
            if abs(delta) < DIFFICULTY_CALIBRATION_MARGIN:
                continue
            shift = 1 if delta > 0 else -1
            for feature_name in ("steps", "concepts", "computation", "inference", "reading_load"):
                minimum, maximum = dials[difficulty][feature_name]
                adjusted_min = _clamp_int(int(minimum) + shift, 0, 4)
                adjusted_max = _clamp_int(int(maximum) + shift, adjusted_min, 4)
                dials[difficulty][feature_name] = [adjusted_min, adjusted_max]

    metadata["source"] = "pilot_file" if observed_rates or applied_override else "default"
    metadata["observed_rates"] = observed_rates
    metadata["sample_counts"] = sample_counts
    return dials, metadata


def _difficulty_budget_text(dial: dict[str, Any]) -> str:
    return (
        f"target_correct_rate={int(round(float(dial.get('target_correct_rate') or 0.0) * 100))}%"
        f"; steps={dial.get('steps', [0, 0])[0]}-{dial.get('steps', [0, 0])[1]}"
        f"; concepts={dial.get('concepts', [0, 0])[0]}-{dial.get('concepts', [0, 0])[1]}"
        f"; computation={dial.get('computation', [0, 0])[0]}-{dial.get('computation', [0, 0])[1]}"
        f"; inference={dial.get('inference', [0, 0])[0]}-{dial.get('inference', [0, 0])[1]}"
        f"; reading_load={dial.get('reading_load', [0, 0])[0]}-{dial.get('reading_load', [0, 0])[1]}"
    )


def _difficulty_dial_guidance_text(
    difficulty_dials: dict[str, dict[str, Any]],
    calibration_metadata: dict[str, Any],
) -> str:
    lines = ["Difficulty dials (use these budgets instead of vibes):"]
    for difficulty in BLUEPRINT_DIFFICULTIES:
        dial = difficulty_dials.get(difficulty)
        if not isinstance(dial, dict):
            continue
        lines.append(f"- {difficulty}: {_difficulty_budget_text(dial)}")
    source_label = str(calibration_metadata.get("source") or "default")
    lines.append(f"Calibration source: {source_label}")
    observed_rates = calibration_metadata.get("observed_rates")
    sample_counts = calibration_metadata.get("sample_counts")
    if isinstance(observed_rates, dict) and isinstance(sample_counts, dict):
        observed_parts = []
        for difficulty in BLUEPRINT_DIFFICULTIES:
            if difficulty not in observed_rates:
                continue
            observed_parts.append(
                f"{difficulty}={int(round(float(observed_rates[difficulty]) * 100))}%"
                f" over {int(sample_counts.get(difficulty) or 0)} responses"
            )
        if observed_parts:
            lines.append("Pilot results: " + "; ".join(observed_parts))
    return "\n".join(lines)


def _attach_difficulty_dials_to_blueprint_slots(
    slots: list[dict[str, Any]],
    difficulty_dials: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_slots: list[dict[str, Any]] = []
    for slot in slots:
        current = dict(slot)
        difficulty = str(current.get("difficulty") or "standard")
        dial = difficulty_dials.get(difficulty, difficulty_dials.get("standard", DEFAULT_DIFFICULTY_DIALS["standard"]))
        current["difficulty_target_correct_rate"] = float(dial.get("target_correct_rate") or 0.0)
        current["difficulty_feature_budget"] = {
            feature_name: list(dial.get(feature_name) or [0, 0])
            for feature_name in ("steps", "concepts", "computation", "inference", "reading_load")
        }
        enriched_slots.append(current)
    return enriched_slots


def _clean_module_blueprint_topic(module_name: str) -> str:
    topic = _normalize_text(module_name)
    topic = re.sub(r"^(unit|chapter)\s*\d+\s*[:\-]\s*", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"^(unit|chapter)\s*\d+\s*", "", topic, flags=re.IGNORECASE)
    return topic.strip(" -:") or _normalize_text(module_name)


def _extract_blueprint_topic_candidates(module_name: str, study_text: str, limit: int) -> list[str]:
    cleaned_text = re.sub(r"\[(?:SRC|FIG_SRC|FIG_IMAGE)[^\]]*\]", " ", study_text)
    cleaned_text = _normalize_text(cleaned_text)
    base_topic = _clean_module_blueprint_topic(module_name)

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        candidate = _normalize_text(value).strip(" -:;,")
        if len(candidate) < 6:
            return
        lowered = candidate.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        candidates.append(candidate)

    if base_topic:
        add_candidate(base_topic)

    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", cleaned_text.lower())
        if token not in _BLUEPRINT_STOPWORDS
    ]
    ngram_counts: Counter[str] = Counter()
    for n in (3, 2):
        for index in range(0, max(0, len(tokens) - n + 1)):
            phrase_tokens = tokens[index:index + n]
            if len(set(phrase_tokens)) < n:
                continue
            phrase = " ".join(phrase_tokens)
            ngram_counts[phrase] += 1

    for phrase, count in ngram_counts.most_common(limit * 4):
        if count < 2 and len(candidates) >= max(2, limit // 2):
            continue
        add_candidate(" ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in phrase.split()))
        if len(candidates) >= limit:
            break

    if len(candidates) < max(2, limit // 2):
        unigram_counts = Counter(tokens)
        for token, _ in unigram_counts.most_common(limit * 3):
            add_candidate(token.capitalize())
            if len(candidates) >= limit:
                break

    return candidates[:limit]


def _objective_label_for_blueprint_topic(topic: str, skill: str) -> str:
    templates = {
        "conceptual understanding": "Explain the economic logic behind {topic}",
        "application": "Apply {topic} to a new scenario",
        "analysis": "Analyze how a change in {topic} alters outcomes",
        "calculation": "Calculate and interpret {topic} in context",
        "graph interpretation": "Interpret a graph or curve involving {topic}",
        "comparison": "Compare alternative outcomes involving {topic}",
        "policy reasoning": "Evaluate a policy or shock affecting {topic}",
        "error analysis": "Diagnose a misconception about {topic}",
    }
    template = templates.get(skill, "Analyze {topic} in context")
    return template.format(topic=topic)


def _key_terms_for_blueprint_slot(topic: str, objective_label: str, skill: str) -> list[str]:
    seed_terms = _derive_blueprint_key_terms(objective_label, topic)
    extras_by_skill = {
        "conceptual understanding": ["logic", "relationship"],
        "application": ["scenario", "market change"],
        "analysis": ["outcome", "shift"],
        "calculation": ["compute", "interpret"],
        "graph interpretation": ["curve", "graph"],
        "comparison": ["compare", "difference"],
        "policy reasoning": ["policy", "incentive"],
        "error analysis": ["mistake", "reasoning"],
    }
    for extra in extras_by_skill.get(skill, []):
        lowered = extra.lower()
        if lowered not in {term.lower() for term in seed_terms}:
            seed_terms.append(extra)
        if len(seed_terms) >= 6:
            break
    return seed_terms[:6]


def _build_deterministic_blueprint(
    *,
    module_name: str,
    question_count: int,
    study_text: str,
    difficulty_dials: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topic_count = min(max(2, math.ceil(question_count / 2)), max(3, question_count))
    topics = _extract_blueprint_topic_candidates(module_name, study_text, limit=topic_count)
    required_topics = 1 if question_count == 1 else min(question_count, max(3, int(math.ceil(question_count * 0.4))))
    if not topics:
        topics = [_clean_module_blueprint_topic(module_name) or "Current module"]
    if len(topics) < required_topics:
        base_topic = _clean_module_blueprint_topic(module_name) or "Current module"
        fallback_suffixes = [
            "core reasoning",
            "quantitative analysis",
            "market outcomes",
            "graph analysis",
            "common errors",
        ]
        for suffix in fallback_suffixes:
            candidate = f"{base_topic} {suffix}"
            if candidate.lower() in {topic.lower() for topic in topics}:
                continue
            topics.append(candidate)
            if len(topics) >= required_topics:
                break

    cleaned_text = study_text.lower()
    preferred_skills: list[str] = ["application", "analysis", "comparison", "conceptual understanding"]
    if "graph" in cleaned_text or "curve" in cleaned_text:
        preferred_skills.insert(2, "graph interpretation")
    if re.search(r"\b(price|wage|cost|revenue|marginal|percent|quantity)\b", cleaned_text):
        preferred_skills.insert(2, "calculation")
    preferred_skills.append("error analysis")
    if "policy" in cleaned_text or "tax" in cleaned_text or "subsid" in cleaned_text:
        preferred_skills.append("policy reasoning")

    skill_cycle: list[str] = []
    for skill in preferred_skills + list(BLUEPRINT_SKILLS):
        if skill not in skill_cycle:
            skill_cycle.append(skill)

    difficulty_cycle = ["standard", "challenge", "foundation"] if question_count >= 4 else ["standard", "challenge"]

    slots: list[dict[str, Any]] = []
    for index in range(question_count):
        topic = topics[index % len(topics)]
        skill = skill_cycle[index % len(skill_cycle)]
        difficulty = difficulty_cycle[index % len(difficulty_cycle)]
        objective_label = _objective_label_for_blueprint_topic(topic, skill)
        objective_id = f"obj_{abs(hash((topic.lower(), skill, difficulty, index))) % 10**8:08d}"
        slots.append(
            {
                "slot_id": f"bp{index + 1:02d}",
                "objective_id": objective_id,
                "objective_label": objective_label,
                "topic": topic,
                "skill": skill,
                "difficulty": difficulty,
                "key_terms": _key_terms_for_blueprint_slot(topic, objective_label, skill),
            }
        )

    slots = _attach_difficulty_dials_to_blueprint_slots(slots, difficulty_dials)
    return slots, _evaluate_blueprint_coverage(slots)


def _build_blueprint_schema(slot_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "blueprint": {
                "type": "array",
                "minItems": slot_count,
                "maxItems": slot_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "objective_id": {"type": "string"},
                        "objective_label": {"type": "string"},
                        "topic": {"type": "string"},
                        "skill": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "key_terms": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "slot_id",
                        "objective_id",
                        "objective_label",
                        "topic",
                        "skill",
                        "difficulty",
                        "key_terms",
                    ],
                },
            }
        },
        "required": ["blueprint"],
    }


def _normalize_blueprint_slots(payload: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Blueprint JSON had an unexpected shape.")

    raw_slots = payload.get("blueprint")
    if not isinstance(raw_slots, list):
        raise RuntimeError("Blueprint response does not include a valid 'blueprint' list.")

    normalized_slots: list[dict[str, Any]] = []
    seen_slot_ids: set[str] = set()
    seen_objective_ids: set[str] = set()
    for index, row in enumerate(raw_slots, start=1):
        if not isinstance(row, dict):
            continue

        slot_id = _normalize_text(str(row.get("slot_id") or "")).lower() or f"bp{index:02d}"
        objective_label = _normalize_text(str(row.get("objective_label") or ""))
        topic = _normalize_text(str(row.get("topic") or ""))
        objective_id = _normalize_text(str(row.get("objective_id") or ""))
        if not objective_id and objective_label and topic:
            objective_id = f"obj_{abs(hash((objective_label.lower(), topic.lower()))) % 10**8:08d}"

        skill = _normalize_blueprint_skill(row.get("skill"))
        difficulty = _normalize_blueprint_difficulty(row.get("difficulty"))

        key_terms_raw = row.get("key_terms")
        key_terms: list[str] = []
        seen_terms: set[str] = set()
        if isinstance(key_terms_raw, list):
            for term in key_terms_raw:
                cleaned_term = _normalize_text(str(term or "")).strip(" -")
                if len(cleaned_term) < 3:
                    continue
                lowered_term = cleaned_term.lower()
                if lowered_term in seen_terms:
                    continue
                seen_terms.add(lowered_term)
                key_terms.append(cleaned_term)
                if len(key_terms) >= 6:
                    break
        if len(key_terms) < 3:
            for term in _derive_blueprint_key_terms(objective_label, topic):
                lowered_term = term.lower()
                if lowered_term in seen_terms:
                    continue
                seen_terms.add(lowered_term)
                key_terms.append(term)
                if len(key_terms) >= 6:
                    break

        if (
            not slot_id
            or not objective_id
            or not objective_label
            or not topic
            or not skill
            or not difficulty
            or len(objective_label) < 16
            or len(topic) < 6
            or len(key_terms) < 3
            or slot_id in seen_slot_ids
            or objective_id in seen_objective_ids
        ):
            continue

        seen_slot_ids.add(slot_id)
        seen_objective_ids.add(objective_id)
        normalized_slots.append(
            {
                "slot_id": slot_id,
                "objective_id": objective_id,
                "objective_label": objective_label,
                "topic": topic,
                "skill": skill,
                "difficulty": difficulty,
                "key_terms": key_terms[:6],
            }
        )

    if len(normalized_slots) != expected_count:
        raise RuntimeError(
            f"Blueprint returned {len(normalized_slots)} valid slots; expected {expected_count}."
        )

    return normalized_slots


def _evaluate_blueprint_coverage(slots: list[dict[str, Any]]) -> dict[str, Any]:
    if not slots:
        return {
            "passed": False,
            "quality_score": float("-inf"),
            "feedback": "Blueprint did not produce any slots.",
            "unique_topics": 0,
            "unique_skills": 0,
            "unique_difficulties": 0,
            "unique_cells": 0,
        }

    total = len(slots)
    unique_topics = {str(slot.get("topic") or "").strip().lower() for slot in slots if str(slot.get("topic") or "").strip()}
    unique_skills = {str(slot.get("skill") or "").strip().lower() for slot in slots if str(slot.get("skill") or "").strip()}
    unique_difficulties = {
        str(slot.get("difficulty") or "").strip().lower()
        for slot in slots
        if str(slot.get("difficulty") or "").strip()
    }
    cell_counts: dict[tuple[str, str, str], int] = {}
    for slot in slots:
        key = (
            str(slot.get("topic") or "").strip().lower(),
            str(slot.get("skill") or "").strip().lower(),
            str(slot.get("difficulty") or "").strip().lower(),
        )
        cell_counts[key] = cell_counts.get(key, 0) + 1

    unique_cells = sum(1 for count in cell_counts.values() if count >= 1)
    duplicate_excess = sum(max(0, count - 1) for count in cell_counts.values())

    min_topics = 1 if total == 1 else min(total, max(2, int(math.ceil(total * 0.4))))
    min_skills = 1 if total == 1 else min(total, max(2, int(math.ceil(total * 0.35))))
    min_difficulties = 1 if total < 4 else 2

    passed = (
        len(unique_topics) >= min_topics
        and len(unique_skills) >= min_skills
        and len(unique_difficulties) >= min_difficulties
        and duplicate_excess <= max(0, total // 3)
    )

    feedback_parts: list[str] = []
    if len(unique_topics) < min_topics:
        feedback_parts.append(
            f"Increase topic coverage: {len(unique_topics)}/{total} distinct topics, need at least {min_topics}."
        )
    if len(unique_skills) < min_skills:
        feedback_parts.append(
            f"Increase skill coverage: {len(unique_skills)}/{total} distinct skills, need at least {min_skills}."
        )
    if len(unique_difficulties) < min_difficulties:
        feedback_parts.append(
            f"Use more than one difficulty level: {len(unique_difficulties)} found, need at least {min_difficulties}."
        )
    if duplicate_excess > max(0, total // 3):
        feedback_parts.append("Reduce repeated topic/skill/difficulty cells.")

    feedback = " ".join(feedback_parts).strip() or "Blueprint coverage passed."
    quality_score = (
        len(unique_topics) * 2.0
        + len(unique_skills) * 1.5
        + len(unique_difficulties) * 1.5
        + unique_cells
        - duplicate_excess
    )
    return {
        "passed": passed,
        "quality_score": quality_score,
        "feedback": feedback,
        "unique_topics": len(unique_topics),
        "unique_skills": len(unique_skills),
        "unique_difficulties": len(unique_difficulties),
        "unique_cells": unique_cells,
    }


def _build_blueprint_text(blueprint_slots: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for slot in blueprint_slots:
        slot_id = str(slot.get("slot_id") or "").strip()
        objective_id = str(slot.get("objective_id") or "").strip()
        objective_label = str(slot.get("objective_label") or "").strip()
        topic = str(slot.get("topic") or "").strip()
        skill = str(slot.get("skill") or "").strip()
        difficulty = str(slot.get("difficulty") or "").strip()
        difficulty_budget = slot.get("difficulty_feature_budget") if isinstance(slot.get("difficulty_feature_budget"), dict) else {}
        target_correct_rate = float(slot.get("difficulty_target_correct_rate") or 0.0)
        key_terms = [
            str(term).strip()
            for term in (slot.get("key_terms") or [])
            if isinstance(term, str) and str(term).strip()
        ]
        if not slot_id or not objective_id or not objective_label:
            continue
        lines.append(
            f'[BLUEPRINT slot="{slot_id}" objective="{objective_id}"] '
            f"label={objective_label}; topic={topic}; skill={skill}; difficulty={difficulty}; "
            f"target_correct_rate={int(round(target_correct_rate * 100))}%; "
            f"steps={difficulty_budget.get('steps', [0, 0])[0]}-{difficulty_budget.get('steps', [0, 0])[1]}; "
            f"concepts={difficulty_budget.get('concepts', [0, 0])[0]}-{difficulty_budget.get('concepts', [0, 0])[1]}; "
            f"computation={difficulty_budget.get('computation', [0, 0])[0]}-{difficulty_budget.get('computation', [0, 0])[1]}; "
            f"inference={difficulty_budget.get('inference', [0, 0])[0]}-{difficulty_budget.get('inference', [0, 0])[1]}; "
            f"reading_load={difficulty_budget.get('reading_load', [0, 0])[0]}-{difficulty_budget.get('reading_load', [0, 0])[1]}; "
            f"key_terms={', '.join(key_terms[:6])}"
        )
    return "\n".join(lines).strip()


def _generate_question_blueprint(
    *,
    client: genai.Client,
    model: str,
    course_name: str,
    module_name: str,
    question_count: int,
    difficulty_profile: str,
    difficulty_guidance: str,
    difficulty_dial_guidance: str,
    difficulty_dials: dict[str, dict[str, Any]],
    source_scope_line: str,
    study_text: str,
    status_callback: Any,
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_slots: list[dict[str, Any]] | None = None
    best_eval: dict[str, Any] | None = None
    revision_feedback = ""

    skill_list = ", ".join(BLUEPRINT_SKILLS)
    difficulty_list = ", ".join(BLUEPRINT_DIFFICULTIES)
    study_excerpt = study_text[:40_000]

    for attempt_index in range(1, max_attempts + 1):
        if status_callback:
            status_callback(f"Building question blueprint (attempt {attempt_index}/{max_attempts})...")

        prompt = (
            f"Course: {course_name}\n"
            f"Module: {module_name}\n"
            f"Question count: {question_count}\n"
            f"Difficulty profile: {difficulty_profile}\n"
            f"Difficulty guidance: {difficulty_guidance}\n"
            f"{difficulty_dial_guidance}\n"
            f"{source_scope_line}\n\n"
            "Create a test blueprint that improves alignment and coverage.\n"
            "Use a topic x skill x difficulty blueprint and return exactly one slot per requested question.\n"
            "Every slot must map to one concrete assessable objective from the study evidence.\n"
            "Reject vague objectives and avoid overfocusing on a single easy topic.\n"
            "Control difficulty using the explicit feature budgets above, not vague labels.\n"
            f"Allowed skills: {skill_list}\n"
            f"Allowed difficulties: {difficulty_list}\n"
            "Return JSON only with this shape:\n"
            "{\"blueprint\":[{\"slot_id\":\"bp01\",\"objective_id\":\"obj_...\",\"objective_label\":\"...\","
            "\"topic\":\"...\",\"skill\":\"application\",\"difficulty\":\"challenge\","
            "\"key_terms\":[\"...\",\"...\",\"...\"]}]}\n"
            "Rules:\n"
            "- Create exactly one slot per question.\n"
            "- Spread slots across distinct topics and skills when the evidence supports it.\n"
            "- Use at least two difficulty levels when the requested count is 4 or more.\n"
            "- objective_label must be concrete and standard-like, not generic.\n"
            "- key_terms must be 3-6 short terms or phrases that a well-aligned question would naturally use.\n"
        )
        if revision_feedback:
            prompt += f"\nRevision requirements from the previous attempt:\n{revision_feedback}\n"
        prompt += f"\nStudy evidence:\n{study_excerpt}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You design rigorous blueprint slots for assessment coverage.",
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Blueprint generation request failed: {exc}") from exc

        parsed_payload = response.parsed
        if parsed_payload is None:
            response_text = response.text if isinstance(response.text, str) else ""
            if not response_text.strip():
                raise RuntimeError("Blueprint generation returned an empty response.")
            try:
                parsed_payload = json.loads(response_text)
            except json.JSONDecodeError as exc:
                parsed_payload = _extract_json_object(response_text)
                if not isinstance(parsed_payload, dict):
                    raise RuntimeError("Blueprint generation did not return valid JSON.") from exc

        try:
            slots = _normalize_blueprint_slots(parsed_payload, question_count)
            slots = _attach_difficulty_dials_to_blueprint_slots(slots, difficulty_dials)
        except Exception as exc:  # noqa: BLE001
            revision_feedback = f"Blueprint structure was invalid: {exc}"
            continue

        coverage_eval = _evaluate_blueprint_coverage(slots)
        if best_eval is None or float(coverage_eval.get("quality_score") or float("-inf")) > float(
            best_eval.get("quality_score") or float("-inf")
        ):
            best_slots = slots
            best_eval = coverage_eval

        if bool(coverage_eval.get("passed")):
            return slots, coverage_eval

        revision_feedback = str(coverage_eval.get("feedback") or "").strip()

    if status_callback:
        status_callback("Model blueprint generation failed; falling back to deterministic local blueprint.")
    fallback_slots, fallback_eval = _build_deterministic_blueprint(
        module_name=module_name,
        question_count=question_count,
        study_text=study_text,
        difficulty_dials=difficulty_dials,
    )
    if bool(fallback_eval.get("passed")):
        return fallback_slots, fallback_eval

    if best_slots is None or best_eval is None:
        raise RuntimeError("Unable to build a valid test blueprint.")
    raise RuntimeError(f"Unable to build a valid test blueprint: {best_eval.get('feedback')}")


def _question_blueprint_text(question: dict[str, Any]) -> str:
    parts = [str(question.get("question") or "").strip()]
    choices = question.get("choices")
    if isinstance(choices, list):
        parts.extend(str(choice).strip() for choice in choices[:4])
    return " ".join(part for part in parts if part).lower()


def _evaluate_question_blueprint_alignment(
    question: dict[str, Any],
    *,
    blueprint_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slot_id = _normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
    slot = blueprint_lookup.get(slot_id)
    if slot is None:
        return {"passed": False, "score": -5.0, "feedback": "Missing or unknown blueprint slot."}

    if _normalize_text(str(question.get("objective_id") or "")) != str(slot.get("objective_id") or ""):
        return {"passed": False, "score": -4.0, "feedback": "Question objective_id did not match its blueprint slot."}
    if _normalize_text(str(question.get("objective_label") or "")).lower() != str(slot.get("objective_label") or "").lower():
        return {"passed": False, "score": -4.0, "feedback": "Question objective_label did not match its blueprint slot."}
    if _normalize_text(str(question.get("objective_topic") or "")).lower() != str(slot.get("topic") or "").lower():
        return {"passed": False, "score": -4.0, "feedback": "Question topic did not match its blueprint slot."}
    if _normalize_blueprint_skill(question.get("objective_skill")) != str(slot.get("skill") or ""):
        return {"passed": False, "score": -4.0, "feedback": "Question skill did not match its blueprint slot."}
    if _normalize_blueprint_difficulty(question.get("objective_difficulty")) != str(slot.get("difficulty") or ""):
        return {"passed": False, "score": -4.0, "feedback": "Question difficulty did not match its blueprint slot."}

    text_blob = _question_blueprint_text(question)
    key_terms = [
        str(term).strip().lower()
        for term in (slot.get("key_terms") or [])
        if isinstance(term, str) and str(term).strip()
    ]
    phrase_hits = sum(1 for term in key_terms if len(term) >= 4 and term in text_blob)

    blueprint_tokens: set[str] = set()
    for value in [slot.get("topic"), slot.get("objective_label"), *(slot.get("key_terms") or [])]:
        blueprint_tokens.update(
            token
            for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(value or "").lower())
            if len(token) >= 4
        )
    question_tokens = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text_blob)
        if len(token) >= 4
    }
    token_hits = len(blueprint_tokens.intersection(question_tokens))
    passed = phrase_hits >= BLUEPRINT_ALIGNMENT_MIN_PHRASE_HITS or token_hits >= BLUEPRINT_ALIGNMENT_MIN_TOKEN_HITS
    if not passed:
        return {
            "passed": False,
            "score": -2.5,
            "feedback": "Question content did not align clearly with its blueprint objective.",
        }

    score = float(phrase_hits * 2.0 + token_hits * 0.5)
    return {"passed": True, "score": score, "feedback": "Blueprint alignment passed."}


def _evaluate_set_blueprint_alignment(
    questions: list[dict[str, Any]],
    *,
    blueprint_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        return {
            "passed": False,
            "quality_score": float("-inf"),
            "feedback": "No questions were generated.",
        }

    blueprint_lookup = {
        str(slot.get("slot_id") or "").strip().lower(): slot
        for slot in blueprint_slots
        if str(slot.get("slot_id") or "").strip()
    }
    used_slots: dict[str, int] = {}
    evaluations: list[dict[str, Any]] = []
    failed_indices: list[int] = []
    for index, question in enumerate(questions, start=1):
        slot_id = _normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
        if slot_id:
            used_slots[slot_id] = used_slots.get(slot_id, 0) + 1
        evaluation = _evaluate_question_blueprint_alignment(question, blueprint_lookup=blueprint_lookup)
        evaluations.append(evaluation)
        if not bool(evaluation.get("passed")):
            failed_indices.append(index)

    missing_slots = [
        slot_id
        for slot_id in blueprint_lookup
        if used_slots.get(slot_id, 0) == 0
    ]
    duplicate_slots = [
        slot_id
        for slot_id, count in used_slots.items()
        if count > 1
    ]

    passed = not failed_indices and not missing_slots and not duplicate_slots and len(questions) == len(blueprint_slots)
    quality_score = sum(float(evaluation.get("score") or 0.0) for evaluation in evaluations) / max(1, len(evaluations))

    feedback_parts: list[str] = []
    if failed_indices:
        feedback_parts.append(
            "Questions failed blueprint alignment: " + ", ".join(str(index) for index in failed_indices[:6]) + "."
        )
    if missing_slots:
        feedback_parts.append("Missing blueprint slots: " + ", ".join(missing_slots[:6]) + ".")
    if duplicate_slots:
        feedback_parts.append("Duplicate blueprint slots: " + ", ".join(duplicate_slots[:6]) + ".")
    if len(questions) != len(blueprint_slots):
        feedback_parts.append(
            f"Question count did not match blueprint slots ({len(questions)} vs {len(blueprint_slots)})."
        )

    feedback = " ".join(feedback_parts).strip() or "Blueprint alignment passed."
    return {
        "passed": passed,
        "quality_score": quality_score,
        "feedback": feedback,
        "failed_indices": failed_indices,
        "missing_slots": missing_slots,
        "duplicate_slots": duplicate_slots,
        "unique_topics": len({str(slot.get("topic") or "").lower() for slot in blueprint_slots}),
        "unique_skills": len({str(slot.get("skill") or "").lower() for slot in blueprint_slots}),
        "unique_difficulties": len({str(slot.get("difficulty") or "").lower() for slot in blueprint_slots}),
    }


def _difficulty_level_from_count(count: int, thresholds: tuple[int, int, int]) -> int:
    if count <= thresholds[0]:
        return 1
    if count <= thresholds[1]:
        return 2
    if count <= thresholds[2]:
        return 3
    return 4


def _estimate_question_difficulty_features(
    question: dict[str, Any],
    *,
    blueprint_slot: dict[str, Any],
) -> dict[str, int]:
    stem = str(question.get("question") or "")
    choices = question.get("choices")
    choice_text = " ".join(str(choice) for choice in choices[:4]) if isinstance(choices, list) else ""
    text_blob = _normalize_text(f"{stem} {choice_text}")
    lowered_text = text_blob.lower()

    stem_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'%-]*", stem)
    choice_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'%-]*", choice_text)
    reading_load = _difficulty_level_from_count(len(stem_words) + max(0, len(choice_words) // 2), (24, 40, 58))

    numeric_signals = len(re.findall(r"\d", text_blob))
    numeric_signals += len(
        re.findall(r"\b(calculate|compute|solve|price|wage|cost|revenue|percent|ratio|quantity|marginal)\b", lowered_text)
    )
    computation = _difficulty_level_from_count(numeric_signals, (1, 3, 6))

    inference_signals = len(
        re.findall(
            r"\b(if|suppose|given|after|before|because|therefore|implies|most likely|best explains|would happen|consistent with|as a result)\b",
            lowered_text,
        )
    )
    inference_signals += 1 if any(token in lowered_text for token in ("compare", "evaluate", "justify", "predict")) else 0
    inference = _difficulty_level_from_count(inference_signals, (1, 3, 5))

    key_terms = [
        str(term).strip().lower()
        for term in (blueprint_slot.get("key_terms") or [])
        if isinstance(term, str) and str(term).strip()
    ]
    term_hits = sum(1 for term in key_terms if len(term) >= 4 and term in lowered_text)
    objective_terms = {
        token
        for token in re.findall(
            r"[A-Za-z][A-Za-z\-]{2,}",
            f"{blueprint_slot.get('objective_label') or ''} {blueprint_slot.get('topic') or ''}".lower(),
        )
        if token not in _BLUEPRINT_STOPWORDS
    }
    question_terms = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", lowered_text)
        if token not in _BLUEPRINT_STOPWORDS
    }
    concept_signals = term_hits + len(objective_terms.intersection(question_terms))
    concepts = _difficulty_level_from_count(concept_signals, (2, 4, 6))

    step_signals = 1
    if computation >= 2:
        step_signals += 1
    if inference >= 2:
        step_signals += 1
    if concepts >= 3:
        step_signals += 1
    if any(token in lowered_text for token in ("compare", "evaluate", "justify", "which combination", "best explains")):
        step_signals += 1
    steps = _clamp_int(step_signals, 1, 4)

    return {
        "steps": steps,
        "concepts": concepts,
        "computation": computation,
        "inference": inference,
        "reading_load": reading_load,
    }


def _difficulty_budget_score(actual_value: int, budget: Any) -> float:
    if not isinstance(budget, list) or len(budget) != 2:
        return 0.0
    try:
        minimum = int(budget[0])
        maximum = int(budget[1])
    except Exception:  # noqa: BLE001
        return 0.0
    if minimum <= actual_value <= maximum:
        return 1.0
    distance = min(abs(actual_value - minimum), abs(actual_value - maximum))
    if distance == 1:
        return 0.5
    return 0.0


def _score_actual_features_against_dial(
    actual_features: dict[str, int],
    dial: dict[str, Any],
) -> tuple[float, dict[str, float], float]:
    feature_scores = {
        feature_name: _difficulty_budget_score(actual_features.get(feature_name, 0), dial.get(feature_name))
        for feature_name in ("steps", "concepts", "computation", "inference", "reading_load")
    }
    average_score = sum(feature_scores.values()) / max(1, len(feature_scores))
    distance = 0.0
    for feature_name in ("steps", "concepts", "computation", "inference", "reading_load"):
        budget = dial.get(feature_name)
        if not isinstance(budget, list) or len(budget) != 2:
            distance += 5.0
            continue
        try:
            minimum = int(budget[0])
            maximum = int(budget[1])
        except Exception:  # noqa: BLE001
            distance += 5.0
            continue
        midpoint = (minimum + maximum) / 2.0
        distance += abs(float(actual_features.get(feature_name, 0)) - midpoint)
    return average_score, feature_scores, distance


def _estimate_actual_difficulty_label(
    actual_features: dict[str, int],
    *,
    difficulty_dials: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    best_label = "standard"
    best_target = float(DEFAULT_DIFFICULTY_DIALS["standard"]["target_correct_rate"])
    best_score = float("-inf")
    best_distance = float("inf")
    best_feature_scores: dict[str, float] = {}

    for difficulty in BLUEPRINT_DIFFICULTIES:
        dial = difficulty_dials.get(difficulty)
        if not isinstance(dial, dict):
            continue
        average_score, feature_scores, distance = _score_actual_features_against_dial(actual_features, dial)
        if (
            average_score > best_score
            or (math.isclose(average_score, best_score) and distance < best_distance)
        ):
            best_label = difficulty
            best_target = float(dial.get("target_correct_rate") or 0.0)
            best_score = average_score
            best_distance = distance
            best_feature_scores = feature_scores

    return {
        "difficulty": best_label,
        "target_correct_rate": best_target,
        "fit_score": best_score,
        "distance": best_distance,
        "feature_scores": best_feature_scores,
    }


def _evaluate_question_difficulty_control(
    question: dict[str, Any],
    *,
    blueprint_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slot_id = _normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
    blueprint_slot = blueprint_lookup.get(slot_id)
    if blueprint_slot is None:
        return {"passed": False, "score": -5.0, "feedback": "Missing blueprint slot for difficulty control."}

    budget = blueprint_slot.get("difficulty_feature_budget")
    if not isinstance(budget, dict):
        return {"passed": False, "score": -4.0, "feedback": "Blueprint slot is missing difficulty feature budget."}

    actual = _estimate_question_difficulty_features(question, blueprint_slot=blueprint_slot)
    feature_scores = {
        feature_name: _difficulty_budget_score(actual.get(feature_name, 0), budget.get(feature_name))
        for feature_name in ("steps", "concepts", "computation", "inference", "reading_load")
    }
    average_score = sum(feature_scores.values()) / max(1, len(feature_scores))
    far_off = [feature_name for feature_name, score in feature_scores.items() if score <= 0.0]
    required_off = [feature_name for feature_name in ("steps", "concepts", "inference") if feature_scores.get(feature_name, 0.0) <= 0.0]
    passed = average_score >= 0.7 and not required_off and len(far_off) <= 1
    if passed:
        feedback = "Difficulty control passed."
    else:
        feedback = (
            "Difficulty control missed the requested budget for: "
            + ", ".join(required_off or far_off[:3] or ["unknown"])
            + "."
        )
    return {
        "passed": passed,
        "score": average_score * 4.0,
        "feedback": feedback,
        "actual_features": actual,
        "feature_scores": feature_scores,
        "target_correct_rate": float(blueprint_slot.get("difficulty_target_correct_rate") or 0.0),
    }


def _evaluate_set_difficulty_control(
    questions: list[dict[str, Any]],
    *,
    blueprint_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        return {
            "passed": False,
            "quality_score": float("-inf"),
            "feedback": "No questions were generated.",
        }

    blueprint_lookup = {
        str(slot.get("slot_id") or "").strip().lower(): slot
        for slot in blueprint_slots
        if str(slot.get("slot_id") or "").strip()
    }
    evaluations = [
        _evaluate_question_difficulty_control(question, blueprint_lookup=blueprint_lookup)
        for question in questions
    ]
    failed_indices = [
        index + 1
        for index, evaluation in enumerate(evaluations)
        if not bool(evaluation.get("passed"))
    ]
    required_pass_count = max(1, int(math.ceil(len(questions) * 0.8)))
    passed_count = len(questions) - len(failed_indices)
    passed = passed_count >= required_pass_count
    quality_score = sum(float(evaluation.get("score") or 0.0) for evaluation in evaluations) / len(evaluations)
    feedback = (
        "Difficulty control passed."
        if passed
        else (
            f"Difficulty control passed for {passed_count}/{len(questions)} questions; "
            f"need at least {required_pass_count}. Failed indices: "
            + ", ".join(str(index) for index in failed_indices[:6])
            + "."
        )
    )
    return {
        "passed": passed,
        "quality_score": quality_score,
        "feedback": feedback,
        "failed_indices": failed_indices,
        "passed_count": passed_count,
        "required_pass_count": required_pass_count,
        "evaluations": evaluations,
    }


def _attach_difficulty_review_to_questions(
    questions: list[dict[str, Any]],
    *,
    blueprint_slots: list[dict[str, Any]],
    difficulty_dials: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    difficulty_eval = _evaluate_set_difficulty_control(
        questions,
        blueprint_slots=blueprint_slots,
    )
    evaluations = difficulty_eval.get("evaluations")
    evaluation_rows = evaluations if isinstance(evaluations, list) else []
    for question, evaluation in zip(questions, evaluation_rows):
        if not isinstance(evaluation, dict):
            continue
        question["difficulty_passed"] = bool(evaluation.get("passed"))
        question["difficulty_feedback"] = str(evaluation.get("feedback") or "")
        actual_features = evaluation.get("actual_features")
        question["difficulty_actual_features"] = (
            dict(actual_features) if isinstance(actual_features, dict) else {}
        )
        feature_scores = evaluation.get("feature_scores")
        question["difficulty_feature_scores"] = (
            dict(feature_scores) if isinstance(feature_scores, dict) else {}
        )
        actual_difficulty = _estimate_actual_difficulty_label(
            question["difficulty_actual_features"],
            difficulty_dials=difficulty_dials,
        )
        question["difficulty_actual_label"] = str(actual_difficulty.get("difficulty") or "").strip()
        try:
            question["difficulty_actual_target_correct_rate"] = float(
                actual_difficulty.get("target_correct_rate") or 0.0
            )
        except Exception:  # noqa: BLE001
            question["difficulty_actual_target_correct_rate"] = 0.0
        try:
            question["difficulty_actual_fit_score"] = float(actual_difficulty.get("fit_score") or 0.0)
        except Exception:  # noqa: BLE001
            question["difficulty_actual_fit_score"] = 0.0
        if "target_correct_rate" in evaluation:
            try:
                question["difficulty_target_correct_rate"] = float(
                    evaluation.get("target_correct_rate") or 0.0
                )
            except Exception:  # noqa: BLE001
                question["difficulty_target_correct_rate"] = float(
                    question.get("difficulty_target_correct_rate") or 0.0
                )
    return difficulty_eval


def _combine_warning_messages(*messages: str | None) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    for raw_message in messages:
        message = _normalize_text(str(raw_message or ""))
        if not message or message in seen:
            continue
        seen.add(message)
        parts.append(message)
    if not parts:
        return None
    return " | ".join(parts)


def _evaluate_question_depth(question: dict[str, Any]) -> dict[str, Any]:
    stem = str(question.get("question") or "").strip()
    choices_raw = question.get("choices")
    choices = choices_raw if isinstance(choices_raw, list) else []
    stem_lower = stem.lower()
    score = 0

    recall_patterns = [
        r"^\s*what is\b",
        r"^\s*which factor\b",
        r"^\s*define\b",
        r"^\s*who\b",
        r"^\s*when\b",
        r"^\s*where\b",
    ]
    recall_like = any(re.search(pattern, stem_lower) for pattern in recall_patterns)
    if recall_like:
        score -= 3

    if len(stem) >= 90:
        score += 1
    if len(stem) >= 140:
        score += 1

    scenario_keywords = (
        "if ",
        "suppose",
        "given ",
        "consider",
        "assume",
        "scenario",
        "policy",
        "market",
        "shift",
        "equilibrium",
        "graph",
        "table",
        "data",
        "most likely",
        "best explains",
        "what happens",
    )
    if any(keyword in stem_lower for keyword in scenario_keywords):
        score += 2

    if re.search(r"\b\d+(\.\d+)?\b", stem_lower) or "%" in stem_lower:
        score += 1

    choice_lengths = [len(str(choice).strip()) for choice in choices[:4]]
    if choice_lengths:
        avg_choice_len = sum(choice_lengths) / len(choice_lengths)
        if avg_choice_len >= 28:
            score += 1
        if avg_choice_len >= 45:
            score += 1

    distractor_signal_terms = (
        "because",
        "therefore",
        "however",
        "depends",
        "only if",
        "increase",
        "decrease",
    )
    rich_distractor_count = 0
    for choice in choices[:4]:
        choice_lower = str(choice).strip().lower()
        if any(term in choice_lower for term in distractor_signal_terms):
            rich_distractor_count += 1
    if rich_distractor_count >= 2:
        score += 1

    is_depth_pass = score >= DEPTH_PASS_SCORE and not recall_like
    return {
        "score": score,
        "is_recall_like": recall_like,
        "is_depth_pass": is_depth_pass,
    }


def _evaluate_set_depth(questions: list[dict[str, Any]]) -> dict[str, Any]:
    if not questions:
        return {
            "passed": False,
            "quality_score": float("-inf"),
            "deep_count": 0,
            "recall_count": 0,
            "required_deep": 0,
            "allowed_recall": 0,
            "feedback": "No questions were generated.",
        }

    evaluations = [_evaluate_question_depth(question) for question in questions]
    deep_count = sum(1 for evaluation in evaluations if evaluation["is_depth_pass"])
    recall_count = sum(1 for evaluation in evaluations if evaluation["is_recall_like"])
    total = len(evaluations)

    required_deep = max(1, int(math.ceil(total * DEPTH_MIN_PASS_RATIO)))
    allowed_recall = int(math.floor(total * RECALL_MAX_RATIO))
    passed = deep_count >= required_deep and recall_count <= allowed_recall

    avg_score = sum(float(evaluation["score"]) for evaluation in evaluations) / max(1, total)
    quality_score = (deep_count * 3.0) - (recall_count * 2.0) + avg_score

    feedback_parts: list[str] = []
    if deep_count < required_deep:
        feedback_parts.append(
            f"Increase depth: only {deep_count}/{total} met the depth bar, need at least {required_deep}/{total}."
        )
    if recall_count > allowed_recall:
        feedback_parts.append(
            f"Reduce recall-style stems: currently {recall_count}/{total}, max allowed is {allowed_recall}/{total}."
        )

    weak_indices = [
        idx + 1
        for idx, evaluation in enumerate(evaluations)
        if not evaluation["is_depth_pass"]
    ][:3]
    if weak_indices:
        feedback_parts.append(f"Rework weak questions: {', '.join(str(index) for index in weak_indices)}.")

    feedback = " ".join(feedback_parts).strip()
    if not feedback:
        feedback = "Depth profile passed."

    return {
        "passed": passed,
        "quality_score": quality_score,
        "deep_count": deep_count,
        "recall_count": recall_count,
        "required_deep": required_deep,
        "allowed_recall": allowed_recall,
        "feedback": feedback,
    }


def _build_misconception_bank_text(misconceptions: list[dict[str, Any]], *, max_records: int = 24) -> str:
    lines: list[str] = []
    for record in misconceptions[:max_records]:
        misconception_id = str(record.get("misconception_id") or "").strip()
        if not misconception_id:
            continue
        topic = str(record.get("topic") or "").strip()
        misconception_label = str(record.get("misconception_label") or "").strip()
        misconception_text = str(record.get("misconception") or "").strip()
        correct_idea = str(record.get("correct_idea") or "").strip()
        tags = [
            str(tag).strip()
            for tag in (record.get("tags") or [])
            if isinstance(tag, str) and str(tag).strip()
        ]
        tags_label = ", ".join(tags[:8])
        safe_topic = topic.replace('"', "'")
        line = (
            f'[MISCONCEPTION id="{misconception_id}" topic="{safe_topic}"] '
            f"label={misconception_label}; tags={tags_label}; misconception={misconception_text}; correct_idea={correct_idea}"
        )
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_distractor_diagnostics(
    diagnostics_raw: Any,
    *,
    answer: str,
    valid_misconception_ids: set[str],
) -> dict[str, dict[str, str] | None] | None:
    if not isinstance(diagnostics_raw, dict):
        return None

    normalized: dict[str, dict[str, str] | None] = {}
    seen_ids: set[str] = set()
    for option in ("A", "B", "C", "D"):
        row = diagnostics_raw.get(option)
        if option == answer:
            normalized[option] = None
            continue

        if not isinstance(row, dict):
            return None

        misconception_id = str(row.get("misconception_id") or "").strip()
        misconception_label = str(row.get("misconception_label") or "").strip()
        why_pick = str(row.get("why_student_might_pick") or "").strip()
        why_wrong = str(row.get("why_wrong") or "").strip()

        if not misconception_id or not misconception_label or not why_pick or not why_wrong:
            return None
        if misconception_id not in valid_misconception_ids:
            return None
        if misconception_id in seen_ids:
            return None

        seen_ids.add(misconception_id)
        normalized[option] = {
            "misconception_id": misconception_id,
            "misconception_label": misconception_label,
            "why_student_might_pick": why_pick,
            "why_wrong": why_wrong,
        }

    return normalized


def _is_generic_diagnostic_text(text: str) -> bool:
    cleaned = text.strip().lower()
    if not cleaned:
        return True
    generic_fragments = (
        "familiar but incorrect reasoning pattern",
        "this option conflicts with the correct concept",
        "does not match the evidence required",
        "common mistake",
        "not correct",
    )
    return any(fragment in cleaned for fragment in generic_fragments)


def _validate_distractor_diagnostics_specificity(
    diagnostics: dict[str, dict[str, str] | None] | None,
    *,
    answer: str,
    valid_misconception_ids: set[str],
) -> bool:
    if not isinstance(diagnostics, dict):
        return False

    seen_ids: set[str] = set()
    why_pick_seen: set[str] = set()
    why_wrong_seen: set[str] = set()
    for option in ("A", "B", "C", "D"):
        if option == answer:
            if diagnostics.get(option) is not None:
                return False
            continue

        row = diagnostics.get(option)
        if not isinstance(row, dict):
            return False

        misconception_id = str(row.get("misconception_id") or "").strip()
        misconception_label = str(row.get("misconception_label") or "").strip()
        why_pick = str(row.get("why_student_might_pick") or "").strip()
        why_wrong = str(row.get("why_wrong") or "").strip()

        if not misconception_id or misconception_id not in valid_misconception_ids:
            return False
        if misconception_id in seen_ids:
            return False
        seen_ids.add(misconception_id)

        if len(misconception_label) < 18 or _is_generic_diagnostic_text(misconception_label):
            return False
        if len(why_pick) < 28 or len(why_wrong) < 28:
            return False
        if _is_generic_diagnostic_text(why_pick) or _is_generic_diagnostic_text(why_wrong):
            return False

        pick_key = why_pick.lower()
        wrong_key = why_wrong.lower()
        if pick_key in why_pick_seen or wrong_key in why_wrong_seen:
            return False
        why_pick_seen.add(pick_key)
        why_wrong_seen.add(wrong_key)

    return True


def _repair_diagnostics_for_question(
    *,
    client: genai.Client,
    model: str,
    question: dict[str, Any],
    misconception_lookup: dict[str, dict[str, Any]],
    valid_misconception_ids: set[str],
) -> dict[str, dict[str, str] | None] | None:
    answer = str(question.get("answer") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return None

    choice_lines: list[str] = []
    choices = question.get("choices")
    if not isinstance(choices, list) or len(choices) < 4:
        return None
    for label, choice in zip(("A", "B", "C", "D"), choices[:4]):
        choice_lines.append(f"{label}) {str(choice).strip()}")

    misconception_lines: list[str] = []
    for misconception_id in sorted(valid_misconception_ids):
        source = misconception_lookup.get(misconception_id) or {}
        label = str(source.get("misconception_label") or source.get("misconception") or "").strip()
        misconception_text = str(source.get("misconception") or "").strip()
        correct_idea = str(source.get("correct_idea") or "").strip()
        if not label or not misconception_text:
            continue
        misconception_lines.append(
            f"- {misconception_id}: label={label}; misconception={misconception_text}; correct={correct_idea}"
        )

    if not misconception_lines:
        return None

    prompt = (
        "Generate distractor_diagnostics JSON for one MCQ.\n"
        "Return JSON only with keys A,B,C,D.\n"
        f"Correct answer is {answer}; its value must be null.\n"
        "For each wrong option, return an object with: misconception_id, misconception_label, "
        "why_student_might_pick, why_wrong.\n"
        "Rules:\n"
        "- Use distinct misconception_id values for each wrong option.\n"
        "- Use only misconception_id values from the provided misconception bank.\n"
        "- Explanations must be option-specific and non-generic.\n\n"
        f"Question: {str(question.get('question') or '').strip()}\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + "\n\nMisconception bank:\n"
        + "\n".join(misconception_lines[:32])
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except Exception:
        return None

    response_text = response.text if isinstance(response.text, str) else ""
    payload = _extract_json_object(response_text)
    diagnostics = _normalize_distractor_diagnostics(
        payload,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    )
    if diagnostics is None:
        return None
    if not _validate_distractor_diagnostics_specificity(
        diagnostics,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    ):
        return None
    return diagnostics


def _repair_diagnostics_for_questions(
    *,
    client: genai.Client,
    model: str,
    questions: list[dict[str, Any]],
    misconception_lookup: dict[str, dict[str, Any]],
    valid_misconception_ids: set[str],
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for question in questions:
        current = dict(question)
        answer = str(current.get("answer") or "").strip().upper()
        diagnostics = current.get("distractor_diagnostics")
        if _validate_distractor_diagnostics_specificity(
            diagnostics if isinstance(diagnostics, dict) else None,
            answer=answer,
            valid_misconception_ids=valid_misconception_ids,
        ):
            repaired.append(current)
            continue

        repaired_diag = _repair_diagnostics_for_question(
            client=client,
            model=model,
            question=current,
            misconception_lookup=misconception_lookup,
            valid_misconception_ids=valid_misconception_ids,
        )
        if repaired_diag is not None:
            current["distractor_diagnostics"] = repaired_diag
        repaired.append(current)
    return repaired


def _evaluate_distractor_diagnostics(
    question: dict[str, Any],
    *,
    valid_misconception_ids: set[str],
) -> dict[str, Any]:
    diagnostics_raw = question.get("distractor_diagnostics")
    diagnostics = diagnostics_raw if isinstance(diagnostics_raw, dict) else {}
    answer = str(question.get("answer") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return {
            "passed": False,
            "score": float("-inf"),
            "feedback": "Invalid answer key in question diagnostics.",
        }

    passed = _validate_distractor_diagnostics_specificity(
        diagnostics,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    )
    if not passed:
        return {
            "passed": False,
            "score": -4.0,
            "feedback": "Distractor diagnostics are missing, generic, or not option-specific.",
            "missing_count": 1,
            "generic_count": 1,
        }

    score = 0.0
    for option in ("A", "B", "C", "D"):
        if option == answer:
            continue
        row = diagnostics.get(option)
        if not isinstance(row, dict):
            continue
        misconception_label = str(row.get("misconception_label") or "").strip()
        why_pick = str(row.get("why_student_might_pick") or "").strip()
        why_wrong = str(row.get("why_wrong") or "").strip()
        score += min(2.0, len(misconception_label) / 40.0)
        score += min(2.0, len(why_pick) / 80.0)
        score += min(2.0, len(why_wrong) / 80.0)

    return {
        "passed": True,
        "score": score,
        "feedback": "Distractor diagnostics passed.",
        "missing_count": 0,
        "generic_count": 0,
    }


def _evaluate_set_distractor_diagnostics(
    questions: list[dict[str, Any]],
    *,
    valid_misconception_ids: set[str],
) -> dict[str, Any]:
    if not questions:
        return {
            "passed": False,
            "quality_score": float("-inf"),
            "feedback": "No questions were generated.",
        }

    evaluations = [
        _evaluate_distractor_diagnostics(
            question,
            valid_misconception_ids=valid_misconception_ids,
        )
        for question in questions
    ]
    failed_indices = [
        idx + 1
        for idx, evaluation in enumerate(evaluations)
        if not bool(evaluation.get("passed"))
    ]
    quality_score = sum(float(evaluation.get("score") or 0.0) for evaluation in evaluations) / len(evaluations)
    passed = not failed_indices
    if passed:
        feedback = "Distractor diagnostics passed."
    else:
        feedback = (
            "Distractor diagnostics failed for question(s): "
            + ", ".join(str(index) for index in failed_indices[:6])
            + ". Ensure every wrong option has non-generic misconception metadata."
        )

    return {
        "passed": passed,
        "quality_score": quality_score,
        "failed_indices": failed_indices,
        "feedback": feedback,
    }


def _attach_figure_image_paths(
    questions: list[dict[str, Any]],
    figure_image_map: dict[tuple[str, int, str], str],
) -> None:
    for question in questions:
        source_file = str(question.get("source_file") or "").strip()
        source_page = _normalize_positive_int(question.get("source_page"))
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
            _bbox_area(tuple(float(v) for v in figure.bbox)),
        ),
        reverse=True,
    )

    deleted_count = len(figure_regions) - len(kept)
    return kept, deleted_count


def _normalize_generated_questions(
    payload: Any,
    expected_count: int,
    *,
    blueprint_slots: list[dict[str, Any]],
    target_image_questions: int,
    valid_sources: set[tuple[str, int]],
    valid_figures: set[tuple[str, int, str]],
    fallback_source: tuple[str, int],
    fallback_figure: tuple[str, int, str] | None,
    valid_misconception_ids: set[str],
    figure_image_map: dict[tuple[str, int, str], str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini response JSON had an unexpected shape.")

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise RuntimeError("Gemini response does not include a valid 'questions' list.")

    blueprint_lookup = {
        str(slot.get("slot_id") or "").strip().lower(): slot
        for slot in blueprint_slots
        if str(slot.get("slot_id") or "").strip()
    }
    blueprint_order = {
        str(slot.get("slot_id") or "").strip().lower(): index
        for index, slot in enumerate(blueprint_slots)
        if str(slot.get("slot_id") or "").strip()
    }

    normalized_questions: list[dict[str, Any]] = []
    for row in raw_questions:
        if not isinstance(row, dict):
            continue

        question_text = _clean_generated_text(str(row.get("question") or ""))
        choices_raw = row.get("choices")
        if not question_text or not isinstance(choices_raw, list):
            continue

        normalized_choices: list[str] = []
        for choice in choices_raw:
            cleaned_choice = _clean_choice_text(str(choice))
            if cleaned_choice:
                normalized_choices.append(cleaned_choice)
        if len(normalized_choices) < 4:
            continue

        answer = str(row.get("answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            answer_match = re.search(r"\b([ABCD])\b", answer)
            answer = answer_match.group(1) if answer_match else ""
        if answer not in {"A", "B", "C", "D"}:
            continue

        blueprint_slot_id = _normalize_text(str(row.get("blueprint_slot_id") or "")).lower()
        blueprint_slot = blueprint_lookup.get(blueprint_slot_id)
        if blueprint_slot is None:
            continue

        objective_id = _normalize_text(str(row.get("objective_id") or ""))
        objective_label = _normalize_text(str(row.get("objective_label") or ""))
        objective_topic = _normalize_text(str(row.get("objective_topic") or ""))
        objective_skill = _normalize_blueprint_skill(row.get("objective_skill"))
        objective_difficulty = _normalize_blueprint_difficulty(row.get("objective_difficulty"))

        if (
            objective_id != str(blueprint_slot.get("objective_id") or "")
            or objective_label.lower() != str(blueprint_slot.get("objective_label") or "").lower()
            or objective_topic.lower() != str(blueprint_slot.get("topic") or "").lower()
            or objective_skill != str(blueprint_slot.get("skill") or "")
            or objective_difficulty != str(blueprint_slot.get("difficulty") or "")
        ):
            continue

        diagnostics = _normalize_distractor_diagnostics(
            row.get("distractor_diagnostics"),
            answer=answer,
            valid_misconception_ids=valid_misconception_ids,
        )

        source_file = str(row.get("source_file") or "").strip()
        source_page = _normalize_positive_int(row.get("source_page"))
        resolved_source = (
            source_file,
            source_page,
        ) if source_file and source_page is not None else fallback_source
        if resolved_source not in valid_sources:
            resolved_source = fallback_source

        raw_figure_id = str(row.get("source_figure_id") or "").strip()
        raw_is_image_grounded = bool(row.get("is_image_grounded"))
        is_image_grounded = raw_is_image_grounded or bool(raw_figure_id)

        resolved_figure_id: str | None = None
        if is_image_grounded:
            candidate_figure = (resolved_source[0], resolved_source[1], raw_figure_id)
            if raw_figure_id and candidate_figure in valid_figures:
                resolved_figure_id = raw_figure_id
            elif fallback_figure is not None:
                resolved_source = (fallback_figure[0], fallback_figure[1])
                resolved_figure_id = fallback_figure[2]
            else:
                is_image_grounded = False

        normalized_questions.append(
            {
                "question": question_text,
                "choices": normalized_choices[:4],
                "blueprint_slot_id": blueprint_slot_id,
                "objective_id": objective_id,
                "objective_label": str(blueprint_slot.get("objective_label") or objective_label),
                "objective_topic": str(blueprint_slot.get("topic") or objective_topic),
                "objective_skill": str(blueprint_slot.get("skill") or objective_skill),
                "objective_difficulty": str(blueprint_slot.get("difficulty") or objective_difficulty),
                "difficulty_target_correct_rate": float(blueprint_slot.get("difficulty_target_correct_rate") or 0.0),
                "difficulty_feature_budget": dict(
                    blueprint_slot.get("difficulty_feature_budget")
                    if isinstance(blueprint_slot.get("difficulty_feature_budget"), dict)
                    else {}
                ),
                "answer": answer,
                "source_file": resolved_source[0],
                "source_page": resolved_source[1],
                "source_figure_id": resolved_figure_id,
                "is_image_grounded": bool(resolved_figure_id),
                "source_figure_image_path": None,
                "distractor_diagnostics": diagnostics,
            }
        )

    if len(normalized_questions) != expected_count:
        raise RuntimeError(
            f"Gemini returned {len(normalized_questions)} valid questions; expected {expected_count}."
        )

    normalized_questions.sort(
        key=lambda question: blueprint_order.get(str(question.get("blueprint_slot_id") or "").strip().lower(), expected_count)
    )

    if target_image_questions > 0:
        image_indices = [
            idx
            for idx, question in enumerate(normalized_questions)
            if question.get("is_image_grounded") is True
        ]

        if len(image_indices) > target_image_questions:
            for idx in image_indices[target_image_questions:]:
                normalized_questions[idx]["is_image_grounded"] = False
                normalized_questions[idx]["source_figure_id"] = None

        elif len(image_indices) < target_image_questions and fallback_figure is not None:
            needed = target_image_questions - len(image_indices)
            for idx, question in enumerate(normalized_questions):
                if needed <= 0:
                    break
                if question.get("is_image_grounded") is True:
                    continue
                question["is_image_grounded"] = True
                question["source_file"] = fallback_figure[0]
                question["source_page"] = fallback_figure[1]
                question["source_figure_id"] = fallback_figure[2]
                needed -= 1

    _attach_figure_image_paths(normalized_questions, figure_image_map or {})

    return normalized_questions


def generate_mcqs_from_tasked_module(
    tasked_items_summary_path: str | Path,
    question_count: int,
    *,
    submitted_items_summary_path: str | Path | None = None,
    model: str = "gemini-2.5-flash",
    max_chars_per_file: int = 6000,
    max_total_chars: int = 50_000,
    difficulty_profile: str = "exam_mixed",
    max_quality_attempts: int = 3,
    misconception_cache_ttl_hours: int = 168,
    misconception_search_model: str = "gemini-2.5-flash",
    misconception_embedding_model: str = "gemini-embedding-001",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    global LAST_GENERATION_REPORT
    if question_count < 1 or question_count > 30:
        raise ValueError("question_count must be between 1 and 30.")
    if max_quality_attempts < 1:
        raise ValueError("max_quality_attempts must be >= 1.")

    LAST_GENERATION_REPORT = {
        "attempts_used": 0,
        "max_quality_attempts": max_quality_attempts,
        "depth_passed": False,
        "diagnostics_passed": False,
        "blueprint_passed": False,
        "difficulty_control_passed": False,
        "difficulty_control_feedback": "",
        "difficulty_failed_indices": [],
        "difficulty_passed_count": 0,
        "difficulty_required_pass_count": 0,
        "warning": None,
        "using_submitted_sources": bool(submitted_items_summary_path),
        "figure_candidates_count": 0,
        "figure_selected_count": 0,
        "figure_deleted_count": 0,
        "importance_provider": "google",
        "importance_model": GOOGLE_IMPORTANCE_MODEL,
        "misconception_cache_hit": False,
        "misconception_raw_count": 0,
        "misconception_canonical_count": 0,
        "misconception_selected_count": 0,
        "misconception_sources_used": 0,
        "misconception_storage_root": "",
        "misconception_search_model": misconception_search_model,
        "misconception_embedding_model": misconception_embedding_model,
        "blueprint_slots_count": 0,
        "blueprint_topics_count": 0,
        "blueprint_skills_count": 0,
        "blueprint_difficulties_count": 0,
        "difficulty_calibration_source": "default",
        "difficulty_calibration_path": str(DIFFICULTY_CALIBRATION_PATH),
    }

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")

    def _status(message: str) -> None:
        if verbose:
            print(f"[MCQ] {message}")

    _status("Loading module cache and source content...")

    summary_path = Path(tasked_items_summary_path).expanduser().resolve()
    tasked_summary_payload = load_tasked_items_summary(summary_path)
    submitted_summary_payload: dict[str, Any] | None = None
    if submitted_items_summary_path:
        submitted_summary_payload = load_summary_payload(
            submitted_items_summary_path,
            label="Submitted module summary",
        )

    combined_summary_payload = _merge_tasked_and_submitted_items(
        tasked_summary_payload,
        submitted_summary_payload,
    )

    source_chunks, source_files, figure_regions = collect_tasked_module_text(
        combined_summary_payload,
        max_chars_per_file=max_chars_per_file,
        max_total_chars=max_total_chars,
    )
    _status(
        f"Parsed source material from {len(source_files)} files "
        f"({len(source_chunks)} text chunks, {len(figure_regions)} figure candidates)."
    )
    figure_candidates_count = len(figure_regions)

    layout_dir = _layout_cache_dir_from_summary_path(summary_path)
    importance_cache_path = layout_dir / "importance_cache_google_models_gemma-3-27b-it.json"
    selected_figure_regions, deleted_figure_count = _classify_figures_with_gemma(
        figure_regions,
        api_key=api_key,
        cache_path=importance_cache_path,
        model=GOOGLE_IMPORTANCE_MODEL,
    )
    _status(
        f"Classified figure importance with {GOOGLE_IMPORTANCE_MODEL}: "
        f"kept {len(selected_figure_regions)} / {figure_candidates_count}."
    )
    figure_selected_count = len(selected_figure_regions)

    if LAYOUT_CACHE_ENABLED:
        _write_layout_cache(
            layout_dir=layout_dir,
            summary_payload=tasked_summary_payload,
            source_chunks=source_chunks,
            figure_regions=selected_figure_regions,
        )
    figure_image_map = _load_figure_image_map(summary_path)

    study_text = _build_tagged_study_text(source_chunks, selected_figure_regions)
    if not study_text and not selected_figure_regions:
        raise RuntimeError("No parseable study content found in tasked module files.")

    valid_sources = {
        (
            str(chunk.get("source_file") or "").strip(),
            int(chunk.get("source_page")),
        )
        for chunk in source_chunks
        if isinstance(chunk.get("source_file"), str)
        and str(chunk.get("source_file")).strip()
        and isinstance(chunk.get("source_page"), int)
        and int(chunk.get("source_page")) >= 1
    }
    for figure in selected_figure_regions:
        valid_sources.add((figure.source_file, figure.source_page))
    if not valid_sources:
        raise RuntimeError("Could not build source references from the selected module files.")

    fallback_source = next(iter(valid_sources))

    valid_figures = {
        (figure.source_file, figure.source_page, figure.source_figure_id)
        for figure in selected_figure_regions
    }
    fallback_figure = next(iter(valid_figures)) if valid_figures else None

    target_image_questions = 0
    if valid_figures:
        target_image_questions = int(math.floor(question_count * IMAGE_GROUNDED_RATIO))
        if question_count > 0 and target_image_questions < 1:
            target_image_questions = 1
        target_image_questions = min(target_image_questions, question_count, len(valid_figures))

    course_payload = tasked_summary_payload.get("course")
    module_payload = tasked_summary_payload.get("module")
    course_name = course_payload.get("name") if isinstance(course_payload, dict) else "Unknown course"
    module_name = module_payload.get("name") if isinstance(module_payload, dict) else "Unknown module"
    course_id = _normalize_positive_int(course_payload.get("id")) if isinstance(course_payload, dict) else None
    module_id = _normalize_positive_int(module_payload.get("id")) if isinstance(module_payload, dict) else None
    source_listing = ", ".join(source_files)
    resolved_difficulty, difficulty_guidance = _difficulty_guidance(difficulty_profile)
    difficulty_dials, difficulty_calibration = _load_difficulty_dials()
    difficulty_dial_guidance = _difficulty_dial_guidance_text(difficulty_dials, difficulty_calibration)
    source_scope_line = (
        "Sources include both tasked module files and submitted assignment artifacts."
        if submitted_summary_payload is not None
        else "Sources include tasked module files."
    )
    _status(
        "Loaded difficulty calibration from "
        f"{difficulty_calibration.get('source') or 'default'}."
    )
    LAST_GENERATION_REPORT["difficulty_calibration_source"] = str(difficulty_calibration.get("source") or "default")
    LAST_GENERATION_REPORT["difficulty_calibration_path"] = str(difficulty_calibration.get("path") or DIFFICULTY_CALIBRATION_PATH)

    misconception_payload = build_or_load_module_misconceptions(
        api_key=api_key,
        course_name=str(course_name),
        module_name=str(module_name),
        course_id=course_id,
        module_id=module_id,
        source_files=source_files,
        source_chunks=source_chunks,
        cache_ttl_hours=misconception_cache_ttl_hours,
        search_model=misconception_search_model,
        embedding_model=misconception_embedding_model,
        status_callback=_status,
    )
    selected_misconceptions_raw = misconception_payload.get("selected_misconceptions")
    selected_misconceptions = (
        selected_misconceptions_raw
        if isinstance(selected_misconceptions_raw, list)
        else []
    )
    if not selected_misconceptions:
        raise RuntimeError("Misconception pipeline did not return any selected misconceptions.")

    valid_misconception_ids = {
        str(record.get("misconception_id") or "").strip()
        for record in selected_misconceptions
        if isinstance(record, dict) and str(record.get("misconception_id") or "").strip()
    }
    misconception_lookup = {
        str(record.get("misconception_id") or "").strip(): record
        for record in selected_misconceptions
        if isinstance(record, dict) and str(record.get("misconception_id") or "").strip()
    }
    if not valid_misconception_ids:
        raise RuntimeError("Misconception pipeline returned records without valid misconception_id values.")

    misconception_bank_text = _build_misconception_bank_text(selected_misconceptions, max_records=24)
    if not misconception_bank_text:
        raise RuntimeError("Misconception bank is empty after filtering.")
    _status(f"Prepared misconception bank with {len(valid_misconception_ids)} records.")

    LAST_GENERATION_REPORT["misconception_cache_hit"] = bool(misconception_payload.get("cache_hit"))
    LAST_GENERATION_REPORT["misconception_raw_count"] = int(misconception_payload.get("raw_count") or 0)
    LAST_GENERATION_REPORT["misconception_canonical_count"] = int(
        misconception_payload.get("canonical_count") or 0
    )
    LAST_GENERATION_REPORT["misconception_selected_count"] = len(valid_misconception_ids)
    sources_used = misconception_payload.get("sources_used")
    LAST_GENERATION_REPORT["misconception_sources_used"] = (
        len(sources_used) if isinstance(sources_used, list) else 0
    )
    LAST_GENERATION_REPORT["misconception_storage_root"] = str(
        misconception_payload.get("storage_root") or ""
    )
    misconception_warning = str(misconception_payload.get("warning") or "").strip()
    if misconception_warning:
        LAST_GENERATION_REPORT["warning"] = misconception_warning
    base_warning = misconception_warning or None

    figure_parts: list[Any] = []
    for figure in selected_figure_regions[:MAX_FIGURE_IMAGE_PARTS_FOR_PROMPT]:
        safe_file = figure.source_file.replace('"', "'")
        figure_parts.append(
            types.Part.from_text(
                text=(
                    f'[FIG_IMAGE file="{safe_file}" page={figure.source_page} '
                    f'fig="{figure.source_figure_id}"]'
                )
            )
        )
        figure_parts.append(types.Part.from_bytes(data=figure.image_bytes, mime_type=figure.mime_type))

    client = genai.Client(api_key=api_key)
    blueprint_slots, blueprint_coverage_eval = _generate_question_blueprint(
        client=client,
        model=model,
        course_name=str(course_name),
        module_name=str(module_name),
        question_count=question_count,
        difficulty_profile=resolved_difficulty,
        difficulty_guidance=difficulty_guidance,
        difficulty_dial_guidance=difficulty_dial_guidance,
        difficulty_dials=difficulty_dials,
        source_scope_line=source_scope_line,
        study_text=study_text,
        status_callback=_status,
        max_attempts=min(3, max_quality_attempts),
    )
    blueprint_text = _build_blueprint_text(blueprint_slots)
    if not blueprint_text:
        raise RuntimeError("Blueprint generation returned no usable slots.")
    _status(
        "Prepared test blueprint with "
        f"{len(blueprint_slots)} slots, {int(blueprint_coverage_eval.get('unique_topics') or 0)} topics, "
        f"{int(blueprint_coverage_eval.get('unique_skills') or 0)} skills."
    )
    LAST_GENERATION_REPORT["blueprint_slots_count"] = len(blueprint_slots)
    LAST_GENERATION_REPORT["blueprint_topics_count"] = int(blueprint_coverage_eval.get("unique_topics") or 0)
    LAST_GENERATION_REPORT["blueprint_skills_count"] = int(blueprint_coverage_eval.get("unique_skills") or 0)
    LAST_GENERATION_REPORT["blueprint_difficulties_count"] = int(
        blueprint_coverage_eval.get("unique_difficulties") or 0
    )
    best_questions: list[dict[str, Any]] | None = None
    best_depth_eval: dict[str, Any] | None = None
    best_diagnostics_eval: dict[str, Any] | None = None
    best_blueprint_eval: dict[str, Any] | None = None
    best_quality_score: float | None = None
    last_error: str | None = None
    revision_feedback = ""
    for attempt_index in range(1, max_quality_attempts + 1):
        _status(f"Generating questions (attempt {attempt_index}/{max_quality_attempts})...")
        LAST_GENERATION_REPORT["attempts_used"] = attempt_index
        LAST_GENERATION_REPORT["figure_candidates_count"] = figure_candidates_count
        LAST_GENERATION_REPORT["figure_selected_count"] = figure_selected_count
        LAST_GENERATION_REPORT["figure_deleted_count"] = deleted_figure_count
        prompt = (
            f"Course: {course_name}\n"
            f"Module: {module_name}\n"
            f"Source files: {source_listing}\n"
            f"{source_scope_line}\n"
            f"Difficulty profile: {resolved_difficulty}\n"
            f"Difficulty guidance: {difficulty_guidance}\n\n"
            f"{difficulty_dial_guidance}\n\n"
            "Generate in-depth multiple choice study questions from the provided content.\n"
            "Questions must mostly require concept understanding plus application.\n"
            "Avoid definition-only or one-step recall stems.\n"
            "Distractors must be plausible and test common misconceptions.\n"
            "Use the misconception bank and simulate wrong reasoning to produce distractors.\n"
            "Prefer scenario-based, graph/data interpretation, or policy reasoning when evidence supports it.\n"
            "Follow the blueprint exactly: generate exactly one question per blueprint slot.\n"
            "Every question must map cleanly to its assigned objective.\n"
            "Control difficulty using the slot's explicit feature budget: steps, concepts, computation, "
            "inference distance, and reading load.\n"
            "Only use figures that passed the Google Gemma importance filtering.\n"
            f"Return exactly {question_count} questions.\n"
            "Each question must have exactly 4 choices.\n"
            "Use answer letters A, B, C, or D only.\n"
            f"Exactly {target_image_questions} questions must be image-grounded (if figure tags are available).\n"
            "Image-grounded questions must use valid [FIG_SRC ...] evidence and provide source_figure_id.\n"
            "Text-grounded questions should keep source_figure_id as an empty string.\n"
            "Use only source_file/source_page/source_figure_id values present in tags.\n"
            "Each question must include blueprint_slot_id, objective_id, objective_label, objective_topic, "
            "objective_skill, objective_difficulty copied exactly from the blueprint slot it fills.\n"
            "Every question must include distractor_diagnostics with keys A/B/C/D.\n"
            "For the correct option, set distractor_diagnostics to null.\n"
            "For each incorrect option, provide misconception_id, misconception_label, why_student_might_pick, why_wrong.\n"
            "misconception_id values must match IDs from the misconception bank.\n"
            "misconception_label must be specific and must not be broad labels like 'Factor Markets'.\n"
            "why_student_might_pick and why_wrong must be specific to that exact option and distinct across options.\n"
            "Return a JSON object only with this top-level shape:\n"
            "{\"questions\": [{\"question\":\"...\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],"
            "\"blueprint_slot_id\":\"bp01\",\"objective_id\":\"obj_...\",\"objective_label\":\"...\","
            "\"objective_topic\":\"...\",\"objective_skill\":\"application\",\"objective_difficulty\":\"challenge\","
            "\"answer\":\"A|B|C|D\","
            "\"source_file\":\"...\",\"source_page\":1,\"source_figure_id\":\"\","
            "\"is_image_grounded\":false,\"distractor_diagnostics\":{\"A\":null,\"B\":{...},\"C\":{...},\"D\":{...}}}]}\n"
            "Never include [SRC], [FIG_SRC], or [FIG_IMAGE] tags in questions or choices.\n"
            "Do not include explanations.\n"
        )
        if revision_feedback:
            prompt += f"\nRevision requirements from previous attempt:\n{revision_feedback}\n"
        prompt += (
            f"\nBlueprint:\n{blueprint_text}\n\n"
            f"Misconception bank:\n{misconception_bank_text}\n\nStudy evidence:\n{study_text}"
        )

        contents: list[Any] = [prompt, *figure_parts]

        try:
            _status("Calling Gemini to generate questions...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You generate rigorous AP-level study MCQs grounded in provided text and figure evidence."
                    ),
                    response_mime_type="application/json",
                    temperature=0.35,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        parsed_payload = response.parsed
        if parsed_payload is None:
            response_text = response.text if isinstance(response.text, str) else ""
            if not response_text.strip():
                raise RuntimeError("Gemini returned an empty response.")
            try:
                parsed_payload = json.loads(response_text)
            except json.JSONDecodeError as exc:
                parsed_payload = _extract_json_object(response_text)
                if not isinstance(parsed_payload, dict):
                    raise RuntimeError("Gemini did not return valid JSON output.") from exc

        try:
            questions = _normalize_generated_questions(
                parsed_payload,
                expected_count=question_count,
                blueprint_slots=blueprint_slots,
                target_image_questions=target_image_questions,
                valid_sources=valid_sources,
                valid_figures=valid_figures,
                fallback_source=fallback_source,
                fallback_figure=fallback_figure,
                valid_misconception_ids=valid_misconception_ids,
                figure_image_map=figure_image_map,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            _status(f"Attempt {attempt_index} returned invalid structure; retrying.")
            revision_feedback = (
                "JSON output was structurally invalid. Return exactly the requested schema and count."
            )
            continue

        _status("Repairing/validating distractor diagnostics...")
        questions = _repair_diagnostics_for_questions(
            client=client,
            model=model,
            questions=questions,
            misconception_lookup=misconception_lookup,
            valid_misconception_ids=valid_misconception_ids,
        )

        depth_eval = _evaluate_set_depth(questions)
        diagnostics_eval = _evaluate_set_distractor_diagnostics(
            questions,
            valid_misconception_ids=valid_misconception_ids,
        )
        difficulty_control_eval = _attach_difficulty_review_to_questions(
            questions,
            blueprint_slots=blueprint_slots,
            difficulty_dials=difficulty_dials,
        )
        blueprint_eval = _evaluate_set_blueprint_alignment(
            questions,
            blueprint_slots=blueprint_slots,
        )
        combined_quality_score = (
            float(depth_eval["quality_score"])
            + float(diagnostics_eval["quality_score"])
            + float(difficulty_control_eval["quality_score"])
            + float(blueprint_eval["quality_score"])
        )
        if (
            best_questions is None
            or best_depth_eval is None
            or best_diagnostics_eval is None
            or best_blueprint_eval is None
            or best_quality_score is None
            or combined_quality_score > best_quality_score
        ):
            best_questions = questions
            best_depth_eval = depth_eval
            best_diagnostics_eval = diagnostics_eval
            best_blueprint_eval = blueprint_eval
            best_quality_score = combined_quality_score

        LAST_GENERATION_REPORT = {
            "attempts_used": attempt_index,
            "max_quality_attempts": max_quality_attempts,
            "difficulty_profile": resolved_difficulty,
            "using_submitted_sources": submitted_summary_payload is not None,
            "figure_candidates_count": figure_candidates_count,
            "figure_selected_count": figure_selected_count,
            "figure_deleted_count": deleted_figure_count,
            "importance_provider": "google",
            "importance_model": GOOGLE_IMPORTANCE_MODEL,
            "depth_passed": bool(depth_eval["passed"]),
            "diagnostics_passed": bool(diagnostics_eval["passed"]),
            "blueprint_passed": bool(blueprint_eval["passed"]),
            "difficulty_control_passed": bool(difficulty_control_eval["passed"]),
            "depth_feedback": str(depth_eval["feedback"]),
            "diagnostics_feedback": str(diagnostics_eval["feedback"]),
            "blueprint_feedback": str(blueprint_eval["feedback"]),
            "difficulty_control_feedback": str(difficulty_control_eval["feedback"]),
            "difficulty_failed_indices": list(difficulty_control_eval.get("failed_indices") or []),
            "difficulty_passed_count": int(difficulty_control_eval.get("passed_count") or 0),
            "difficulty_required_pass_count": int(difficulty_control_eval.get("required_pass_count") or 0),
            "deep_count": int(depth_eval["deep_count"]),
            "required_deep": int(depth_eval["required_deep"]),
            "recall_count": int(depth_eval["recall_count"]),
            "allowed_recall": int(depth_eval["allowed_recall"]),
            "blueprint_slots_count": len(blueprint_slots),
            "blueprint_topics_count": int(blueprint_eval.get("unique_topics") or 0),
            "blueprint_skills_count": int(blueprint_eval.get("unique_skills") or 0),
            "blueprint_difficulties_count": int(blueprint_eval.get("unique_difficulties") or 0),
            "misconception_cache_hit": bool(misconception_payload.get("cache_hit")),
            "misconception_raw_count": int(misconception_payload.get("raw_count") or 0),
            "misconception_canonical_count": int(misconception_payload.get("canonical_count") or 0),
            "misconception_selected_count": len(valid_misconception_ids),
            "misconception_sources_used": len(sources_used) if isinstance(sources_used, list) else 0,
            "misconception_storage_root": str(misconception_payload.get("storage_root") or ""),
            "misconception_search_model": misconception_search_model,
            "misconception_embedding_model": misconception_embedding_model,
            "difficulty_calibration_source": str(difficulty_calibration.get("source") or "default"),
            "difficulty_calibration_path": str(difficulty_calibration.get("path") or DIFFICULTY_CALIBRATION_PATH),
            "warning": base_warning,
        }

        if depth_eval["passed"] and diagnostics_eval["passed"] and difficulty_control_eval["passed"] and blueprint_eval["passed"]:
            _status("Generation complete: quality checks passed.")
            return questions

        _status("Quality checks not met; preparing targeted retry.")
        revision_feedback = (
            f"{str(depth_eval['feedback'])} "
            f"{str(diagnostics_eval['feedback'])} "
            f"{str(difficulty_control_eval['feedback'])} "
            f"{str(blueprint_eval['feedback'])}"
        ).strip()

    if best_questions is None:
        if last_error:
            raise RuntimeError(
                f"Unable to produce valid MCQs after {max_quality_attempts} attempt(s): {last_error}"
            )
        raise RuntimeError(f"Unable to produce valid MCQs after {max_quality_attempts} attempt(s).")

    if not isinstance(best_diagnostics_eval, dict) or not bool(best_diagnostics_eval.get("passed")):
        diagnostics_feedback = (
            str(best_diagnostics_eval.get("feedback"))
            if isinstance(best_diagnostics_eval, dict)
            else "Distractor diagnostic validation failed."
        )
        raise RuntimeError(
            "Failed to produce MCQs with valid distractor diagnostics after "
            f"{max_quality_attempts} attempt(s): {diagnostics_feedback}"
        )

    best_difficulty_control_eval = _attach_difficulty_review_to_questions(
        best_questions,
        blueprint_slots=blueprint_slots,
        difficulty_dials=difficulty_dials,
    )

    if not isinstance(best_blueprint_eval, dict) or not bool(best_blueprint_eval.get("passed")):
        blueprint_feedback = (
            str(best_blueprint_eval.get("feedback"))
            if isinstance(best_blueprint_eval, dict)
            else "Blueprint alignment validation failed."
        )
        raise RuntimeError(
            "Failed to produce MCQs with valid blueprint alignment after "
            f"{max_quality_attempts} attempt(s): {blueprint_feedback}"
        )

    warning_parts: list[str] = []
    if not isinstance(best_depth_eval, dict) or not bool(best_depth_eval.get("passed")):
        warning_parts.append(
            f"Depth quality target not met after {max_quality_attempts} attempt(s). Using best available question set."
        )
    if not bool(best_difficulty_control_eval.get("passed")):
        warning_parts.append(
            "Difficulty control target not met after "
            f"{max_quality_attempts} attempt(s): {str(best_difficulty_control_eval.get('feedback') or '').strip()}"
        )
    warning = _combine_warning_messages(
        base_warning,
        " ".join(part for part in warning_parts if part).strip(),
    )
    _status("Returning best available question set after retries.")
    LAST_GENERATION_REPORT = {
        "attempts_used": max_quality_attempts,
        "max_quality_attempts": max_quality_attempts,
        "difficulty_profile": resolved_difficulty,
        "using_submitted_sources": submitted_summary_payload is not None,
        "figure_candidates_count": figure_candidates_count,
        "figure_selected_count": figure_selected_count,
        "figure_deleted_count": deleted_figure_count,
        "importance_provider": "google",
        "importance_model": GOOGLE_IMPORTANCE_MODEL,
        "depth_passed": bool(best_depth_eval["passed"]) if isinstance(best_depth_eval, dict) else False,
        "diagnostics_passed": True,
        "blueprint_passed": True,
        "difficulty_control_passed": bool(best_difficulty_control_eval.get("passed")),
        "depth_feedback": (
            str(best_depth_eval["feedback"]) if isinstance(best_depth_eval, dict) else "Depth gate not met."
        ),
        "diagnostics_feedback": str(best_diagnostics_eval["feedback"]),
        "blueprint_feedback": str(best_blueprint_eval["feedback"]) if isinstance(best_blueprint_eval, dict) else "",
        "difficulty_control_feedback": str(best_difficulty_control_eval.get("feedback") or ""),
        "difficulty_failed_indices": list(best_difficulty_control_eval.get("failed_indices") or []),
        "difficulty_passed_count": int(best_difficulty_control_eval.get("passed_count") or 0),
        "difficulty_required_pass_count": int(best_difficulty_control_eval.get("required_pass_count") or 0),
        "deep_count": int(best_depth_eval["deep_count"]) if isinstance(best_depth_eval, dict) else 0,
        "required_deep": int(best_depth_eval["required_deep"]) if isinstance(best_depth_eval, dict) else 0,
        "recall_count": int(best_depth_eval["recall_count"]) if isinstance(best_depth_eval, dict) else 0,
        "allowed_recall": int(best_depth_eval["allowed_recall"]) if isinstance(best_depth_eval, dict) else 0,
        "blueprint_slots_count": len(blueprint_slots),
        "blueprint_topics_count": int(best_blueprint_eval.get("unique_topics") or 0) if isinstance(best_blueprint_eval, dict) else 0,
        "blueprint_skills_count": int(best_blueprint_eval.get("unique_skills") or 0) if isinstance(best_blueprint_eval, dict) else 0,
        "blueprint_difficulties_count": int(best_blueprint_eval.get("unique_difficulties") or 0) if isinstance(best_blueprint_eval, dict) else 0,
        "misconception_cache_hit": bool(misconception_payload.get("cache_hit")),
        "misconception_raw_count": int(misconception_payload.get("raw_count") or 0),
        "misconception_canonical_count": int(misconception_payload.get("canonical_count") or 0),
        "misconception_selected_count": len(valid_misconception_ids),
        "misconception_sources_used": len(sources_used) if isinstance(sources_used, list) else 0,
        "misconception_storage_root": str(misconception_payload.get("storage_root") or ""),
        "misconception_search_model": misconception_search_model,
        "misconception_embedding_model": misconception_embedding_model,
        "difficulty_calibration_source": str(difficulty_calibration.get("source") or "default"),
        "difficulty_calibration_path": str(difficulty_calibration.get("path") or DIFFICULTY_CALIBRATION_PATH),
        "warning": warning,
    }
    return best_questions


def render_mcqs_markdown(
    questions: list[dict[str, Any]],
    *,
    course_name: str,
    module_name: str,
    generated_at: str,
) -> str:
    report = LAST_GENERATION_REPORT if isinstance(LAST_GENERATION_REPORT, dict) else {}
    blueprint_rows: list[dict[str, str]] = []
    seen_blueprint_slots: set[str] = set()
    for question in questions:
        slot_id = str(question.get("blueprint_slot_id") or "").strip()
        if not slot_id or slot_id in seen_blueprint_slots:
            continue
        seen_blueprint_slots.add(slot_id)
        blueprint_rows.append(
            {
                "slot_id": slot_id,
                "objective_label": str(question.get("objective_label") or "").strip(),
                "objective_topic": str(question.get("objective_topic") or "").strip(),
                "objective_skill": str(question.get("objective_skill") or "").strip(),
                "objective_difficulty": str(question.get("objective_difficulty") or "").strip(),
                "difficulty_target_correct_rate": float(question.get("difficulty_target_correct_rate") or 0.0),
                "difficulty_feature_budget": dict(
                    question.get("difficulty_feature_budget")
                    if isinstance(question.get("difficulty_feature_budget"), dict)
                    else {}
                ),
            }
        )

    lines: list[str] = [
        "# Multiple-Choice Questions",
        "",
        f"- Course: {course_name}",
        f"- Module: {module_name}",
        f"- Generated: {generated_at}",
        f"- Question Count: {len(questions)}",
        "",
    ]

    if blueprint_rows:
        lines.extend(["## Blueprint", ""])
        for row in blueprint_rows:
            lines.append(
                "- "
                f"{row['slot_id']}: {row['objective_label']} "
                f"(Topic: {row['objective_topic']} | Skill: {row['objective_skill']} | "
                f"Difficulty: {row['objective_difficulty']})"
            )
            budget = row.get("difficulty_feature_budget") if isinstance(row.get("difficulty_feature_budget"), dict) else {}
            lines.append(
                "  Difficulty Dial: "
                f"target {int(round(float(row.get('difficulty_target_correct_rate') or 0.0) * 100))}% correct"
                f" | steps {budget.get('steps', [0, 0])[0]}-{budget.get('steps', [0, 0])[1]}"
                f" | concepts {budget.get('concepts', [0, 0])[0]}-{budget.get('concepts', [0, 0])[1]}"
                f" | computation {budget.get('computation', [0, 0])[0]}-{budget.get('computation', [0, 0])[1]}"
                f" | inference {budget.get('inference', [0, 0])[0]}-{budget.get('inference', [0, 0])[1]}"
                f" | reading {budget.get('reading_load', [0, 0])[0]}-{budget.get('reading_load', [0, 0])[1]}"
            )
        lines.extend([""])

    difficulty_rows = [
        question
        for question in questions
        if isinstance(question.get("difficulty_passed"), bool)
    ]
    difficulty_failed_indices_report = report.get("difficulty_failed_indices")
    difficulty_failed_indices = (
        list(difficulty_failed_indices_report)
        if isinstance(difficulty_failed_indices_report, list)
        else [index for index, question in enumerate(difficulty_rows, start=1) if question.get("difficulty_passed") is False]
    )
    difficulty_passed_count = (
        int(report.get("difficulty_passed_count") or 0)
        if isinstance(report.get("difficulty_passed_count"), int)
        else sum(1 for question in difficulty_rows if question.get("difficulty_passed") is True)
    )
    difficulty_required_pass_count = (
        int(report.get("difficulty_required_pass_count") or 0)
        if isinstance(report.get("difficulty_required_pass_count"), int)
        else max(1, int(math.ceil(max(1, len(difficulty_rows)) * 0.8))) if difficulty_rows else 0
    )
    difficulty_calibration_source = str(report.get("difficulty_calibration_source") or "").strip() or "unknown"
    difficulty_control_feedback = str(report.get("difficulty_control_feedback") or "").strip()
    if difficulty_rows:
        lines.extend(
            [
                "## Difficulty Review",
                "",
                f"- Calibration Source: {difficulty_calibration_source}",
                f"- Passed Count: {difficulty_passed_count}/{len(difficulty_rows)}",
                f"- Required Pass Count: {difficulty_required_pass_count}",
                (
                    f"- Failed Questions: {', '.join(str(index) for index in difficulty_failed_indices)}"
                    if difficulty_failed_indices
                    else "- Failed Questions: none"
                ),
            ]
        )
        if difficulty_control_feedback:
            lines.append(f"- Review Summary: {difficulty_control_feedback}")
        lines.extend([""])

    lines.extend([
        "## Questions",
        "",
    ])

    answer_key: list[str] = []
    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question") or "").strip()
        choices = question.get("choices")
        if not question_text or not isinstance(choices, list) or len(choices) < 4:
            continue

        source_file = str(question.get("source_file") or "Unknown source").strip() or "Unknown source"
        source_page = _normalize_positive_int(question.get("source_page")) or 1
        source_figure_id = str(question.get("source_figure_id") or "").strip()
        if source_figure_id:
            source_label = f"[Source: {source_file} p.{source_page} {source_figure_id}]"
        else:
            source_label = f"[Source: {source_file} p.{source_page}]"

        lines.append(f"{index}. {question_text} {source_label}")
        objective_label = str(question.get("objective_label") or "").strip()
        objective_topic = str(question.get("objective_topic") or "").strip()
        objective_skill = str(question.get("objective_skill") or "").strip()
        objective_difficulty = str(question.get("objective_difficulty") or "").strip()
        blueprint_slot_id = str(question.get("blueprint_slot_id") or "").strip()
        if objective_label or objective_topic or objective_skill or objective_difficulty:
            lines.append(
                "   Objective: "
                f"{objective_label or 'Unknown objective'}"
                + (
                    f" [{blueprint_slot_id}]"
                    if blueprint_slot_id
                    else ""
                )
                + (
                    f" | Topic: {objective_topic}"
                    if objective_topic
                    else ""
                )
                + (
                    f" | Skill: {objective_skill}"
                    if objective_skill
                    else ""
                )
                + (
                    f" | Difficulty: {objective_difficulty}"
                    if objective_difficulty
                    else ""
                )
            )
        difficulty_budget = question.get("difficulty_feature_budget") if isinstance(question.get("difficulty_feature_budget"), dict) else {}
        target_correct_rate = float(question.get("difficulty_target_correct_rate") or 0.0)
        if difficulty_budget:
            lines.append(
                "   Difficulty Dial: "
                f"target {int(round(target_correct_rate * 100))}% correct"
                f" | steps {difficulty_budget.get('steps', [0, 0])[0]}-{difficulty_budget.get('steps', [0, 0])[1]}"
                f" | concepts {difficulty_budget.get('concepts', [0, 0])[0]}-{difficulty_budget.get('concepts', [0, 0])[1]}"
                f" | computation {difficulty_budget.get('computation', [0, 0])[0]}-{difficulty_budget.get('computation', [0, 0])[1]}"
                f" | inference {difficulty_budget.get('inference', [0, 0])[0]}-{difficulty_budget.get('inference', [0, 0])[1]}"
                f" | reading {difficulty_budget.get('reading_load', [0, 0])[0]}-{difficulty_budget.get('reading_load', [0, 0])[1]}"
            )
        actual_difficulty_label = str(question.get("difficulty_actual_label") or "").strip()
        actual_target_correct_rate = float(question.get("difficulty_actual_target_correct_rate") or 0.0)
        if actual_difficulty_label:
            lines.append(
                "   Actual Difficulty: "
                f"{actual_difficulty_label}"
                + (
                    f" (~{int(round(actual_target_correct_rate * 100))}% correct)"
                    if actual_target_correct_rate > 0
                    else ""
                )
            )
        difficulty_passed = question.get("difficulty_passed")
        if isinstance(difficulty_passed, bool):
            lines.append(
                f"   Difficulty Review: {'passed' if difficulty_passed else 'missed target'}"
            )
            difficulty_feedback = str(question.get("difficulty_feedback") or "").strip()
            if difficulty_feedback:
                lines.append(f"   Difficulty Feedback: {difficulty_feedback}")
            if not difficulty_passed:
                actual_features = question.get("difficulty_actual_features")
                if isinstance(actual_features, dict) and actual_features:
                    lines.append(
                        "   Actual Features: "
                        f"steps {actual_features.get('steps', '?')}"
                        f" | concepts {actual_features.get('concepts', '?')}"
                        f" | computation {actual_features.get('computation', '?')}"
                        f" | inference {actual_features.get('inference', '?')}"
                        f" | reading {actual_features.get('reading_load', '?')}"
                    )
                feature_scores = question.get("difficulty_feature_scores")
                if isinstance(feature_scores, dict) and feature_scores:
                    feature_labels = []
                    for feature_name in ("steps", "concepts", "computation", "inference", "reading_load"):
                        score = float(feature_scores.get(feature_name) or 0.0)
                        label = "ok" if score >= 1.0 else "near" if score >= 0.5 else "miss"
                        feature_labels.append(f"{feature_name}={label}")
                    lines.append("   Feature Fit: " + " | ".join(feature_labels))
        for label, choice in zip(("A", "B", "C", "D"), choices[:4]):
            lines.append(f"   {label}) {str(choice).strip()}")

        figure_image_path = str(question.get("source_figure_image_path") or "").strip()
        if figure_image_path:
            figure_link_name = Path(figure_image_path).name
            lines.append(f"   Figure: [{figure_link_name}]({figure_image_path})")

        diagnostics_raw = question.get("distractor_diagnostics")
        diagnostics = diagnostics_raw if isinstance(diagnostics_raw, dict) else {}
        answer = str(question.get("answer") or "").strip().upper()
        lines.append("   Distractor Diagnostics:")
        for label in ("A", "B", "C", "D"):
            if label == answer:
                continue
            diagnostic = diagnostics.get(label)
            if not isinstance(diagnostic, dict):
                lines.append(f"   - {label}: [missing diagnostic metadata]")
                continue
            misconception_label = str(diagnostic.get("misconception_label") or "").strip() or "Unknown misconception"
            misconception_id = str(diagnostic.get("misconception_id") or "").strip() or "unknown_id"
            why_pick = str(diagnostic.get("why_student_might_pick") or "").strip() or "No rationale provided."
            why_wrong = str(diagnostic.get("why_wrong") or "").strip() or "No correction provided."
            lines.append(f"   - {label}: {misconception_label} ({misconception_id})")
            lines.append(f"     Why chosen: {why_pick}")
            lines.append(f"     Why wrong: {why_wrong}")
        lines.append("")

        normalized_answer = answer if answer in {"A", "B", "C", "D"} else "?"
        answer_key.append(f"{index}:{normalized_answer}")

    if answer_key:
        lines.extend(["## Answer Key", "", f"Answer Key: {', '.join(answer_key)}", ""])

    return "\n".join(lines).strip() + "\n"


def save_mcqs_markdown(
    questions: list[dict[str, Any]],
    *,
    tasked_items_summary_path: str | Path,
    course_name: str,
    module_name: str,
) -> Path:
    summary_path = Path(tasked_items_summary_path).expanduser().resolve()
    output_dir = _mcq_output_dir_from_summary_path(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")
    generated_label = timestamp.isoformat()

    normalized_questions: list[dict[str, Any]] = []
    for row in questions:
        current = dict(row)
        figure_image_path = str(current.get("source_figure_image_path") or "").strip()
        if figure_image_path:
            image_path = Path(figure_image_path)
            if not image_path.is_absolute():
                image_path = (summary_path.parent / image_path).resolve()
            try:
                relative_path = image_path.relative_to(output_dir)
            except ValueError:
                relative_path = Path(os.path.relpath(str(image_path), start=str(output_dir)))
            current["source_figure_image_path"] = str(relative_path)
        else:
            current["source_figure_image_path"] = None
        normalized_questions.append(current)

    markdown_text = render_mcqs_markdown(
        normalized_questions,
        course_name=course_name,
        module_name=module_name,
        generated_at=generated_label,
    )

    timestamped_path = output_dir / f"mcq_{timestamp_label}.md"
    latest_path = output_dir / "mcq_latest.md"
    timestamped_path.write_text(markdown_text, encoding="utf-8")
    latest_path.write_text(markdown_text, encoding="utf-8")
    return timestamped_path


def print_mcqs(questions: list[dict[str, Any]]) -> None:
    answer_key: list[str] = []

    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question") or "").strip()
        choices = question.get("choices")
        if not question_text or not isinstance(choices, list) or len(choices) < 4:
            continue

        source_file = str(question.get("source_file") or "Unknown source").strip() or "Unknown source"
        source_page = _normalize_positive_int(question.get("source_page")) or 1
        source_figure_id = str(question.get("source_figure_id") or "").strip()

        if source_figure_id:
            source_label = f"[Source: {source_file} p.{source_page} {source_figure_id}]"
        else:
            source_label = f"[Source: {source_file} p.{source_page}]"

        print(f"{index}. {question_text} {source_label}")
        objective_label = str(question.get("objective_label") or "").strip()
        objective_topic = str(question.get("objective_topic") or "").strip()
        objective_skill = str(question.get("objective_skill") or "").strip()
        objective_difficulty = str(question.get("objective_difficulty") or "").strip()
        blueprint_slot_id = str(question.get("blueprint_slot_id") or "").strip()
        if objective_label or objective_topic or objective_skill or objective_difficulty:
            print(
                "Objective: "
                f"{objective_label or 'Unknown objective'}"
                + (f" [{blueprint_slot_id}]" if blueprint_slot_id else "")
                + (f" | Topic: {objective_topic}" if objective_topic else "")
                + (f" | Skill: {objective_skill}" if objective_skill else "")
                + (f" | Difficulty: {objective_difficulty}" if objective_difficulty else "")
            )
        for label, choice in zip(("A", "B", "C", "D"), choices[:4]):
            print(f"{label}) {str(choice).strip()}")
        print()

        answer = str(question.get("answer") or "").strip().upper()
        normalized_answer = answer if answer in {"A", "B", "C", "D"} else "?"
        answer_key.append(f"{index}:{normalized_answer}")

    if answer_key:
        print(f"Answer Key: {', '.join(answer_key)}")
