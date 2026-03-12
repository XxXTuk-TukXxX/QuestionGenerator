from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_SOURCE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx"}
IGNORED_SOURCE_FILENAMES = {"item.json", "item_links.txt", "submission.json"}

IMAGE_GROUNDED_RATIO = 0.25
LAYOUT_CACHE_ENABLED = True
LAYOUT_CACHE_DIRNAME = "layout"
MCQ_OUTPUT_DIRNAME = "mcq"

MAX_FIGURE_IMAGE_PARTS_FOR_PROMPT = 18
GOOGLE_IMPORTANCE_MODEL = "models/gemma-3-27b-it"

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
DIFFICULTY_CALIBRATION_PATH = PROJECT_ROOT / "User" / "DifficultyCalibration" / "calibration.json"
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


def merge_tasked_and_submitted_items(
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

    merged_payload = dict(tasked_summary_payload)
    merged_payload["items"] = merged_items
    return merged_payload


def canonical_filename(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalize_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_generated_text(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None

    decoder = json.JSONDecoder()
    # Gemini occasionally wraps the JSON object in prose or code fences,
    # so scan for the first decodable object instead of assuming a clean payload.
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


def clean_choice_text(value: str) -> str:
    text = clean_generated_text(value)
    text = re.sub(r"^\s*[A-D][\).:-]\s*", "", text)
    return text.strip()


def sanitize_path_part(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return sanitized[:80] if sanitized else fallback


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)
