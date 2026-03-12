from __future__ import annotations

import argparse
import contextlib
import json
import mimetypes
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_policy_default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import canvas
import gemma
import study_engine
from models.gemma_output import render_mcqs_markdown
from models.gemma_sources import _mcq_output_dir_from_summary_path

FRONTEND_ROOT = REPO_ROOT / "Frontend"
CACHE_ROOT = REPO_ROOT / "Cache"
SOURCE_FILE_SUFFIXES = {
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".md",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".txt",
    ".webp",
}
SOURCE_FILE_SKIP_NAMES = {"item_links.txt", "submitted_assignments.txt", "tasked_items.txt"}
NATURAL_CHUNK_RE = re.compile(r"(\d+)")
PAGE_SPLIT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<page>\d+)$")
QUIZ_METADATA_RE = re.compile(r"^- (?P<key>[^:]+): (?P<value>.+)$")
QUIZ_QUESTION_RE = re.compile(
    r"^(?P<number>\d+)\.\s+(?P<question>.+?)(?:\s+\[Source:\s+(?P<source_file>.+?)\s+p\.(?P<source_page>\d+)(?:\s+(?P<source_figure>[^\]]+))?\])?$"
)
QUIZ_CHOICE_RE = re.compile(r"^\s*(?P<label>[ABCD])\)\s+(?P<text>.+)$")
QUIZ_DIAGNOSTIC_RE = re.compile(r"^\s*-\s*(?P<label>[ABCD]):\s+(?P<body>.+)$")
QUIZ_DIAGNOSTIC_ID_RE = re.compile(r"^(?P<label>.+?)\s+\((?P<misconception_id>[^()]+)\)\s*$")
QUIZ_ANSWER_KEY_RE = re.compile(r"^Answer Key:\s*(?P<body>.+)$")
DEFAULT_GENERATION_QUESTION_COUNT = 20
GENERATION_LOCK = threading.Lock()
TASKED_SUMMARY_FILENAMES = {"tasked_items.json", "custom_items.json"}
DIFFICULTY_ORDER = {"foundation": 0, "standard": 1, "challenge": 2}


def _discover_markdown_files(cache_root: Path) -> list[dict[str, object]]:
    if not cache_root.exists():
        return []

    paths = [
        path
        for path in cache_root.rglob("mcq*.md")
        if path.is_file() and "mcq" in path.relative_to(cache_root).parts
    ]
    paths.sort(key=lambda path: (-path.stat().st_mtime, str(path.relative_to(cache_root)).lower()))

    files: list[dict[str, object]] = []
    for path in paths:
        files.append(_build_cache_file_payload(path))
    return files


def _resolve_cache_file(relative_path: str) -> Path | None:
    candidate = (CACHE_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(CACHE_ROOT.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _build_cache_file_payload(path: Path, *, directory_root: Path | None = None, bucket: str = "") -> dict[str, object]:
    stat = path.stat()
    relative_path = path.relative_to(CACHE_ROOT).as_posix()
    if directory_root is None:
        directory = path.parent.relative_to(CACHE_ROOT).as_posix()
    else:
        directory = path.parent.relative_to(directory_root).as_posix()

    payload: dict[str, object] = {
        "name": path.name,
        "relative_path": relative_path,
        "directory": directory,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "url": f"/cache/{quote(relative_path, safe='/')}",
    }
    if bucket:
        payload["bucket"] = bucket
    return payload


def _natural_key(text: str) -> list[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in NATURAL_CHUNK_RE.split(text)]


def _display_name(filename: str) -> str:
    return filename.replace("_", " ")


def _module_root_for_quiz(quiz_path: Path) -> Path | None:
    try:
        relative_parts = quiz_path.relative_to(CACHE_ROOT).parts
    except ValueError:
        return None

    if "mcq" not in relative_parts:
        return None

    mcq_index = relative_parts.index("mcq")
    if mcq_index < 2:
        return None
    return CACHE_ROOT.joinpath(*relative_parts[:mcq_index])


def _is_page_split_derivative(path: Path, sibling_names: set[str]) -> bool:
    match = PAGE_SPLIT_SUFFIX_RE.match(path.stem)
    if not match:
        return False
    original_name = f"{match.group('base')}{path.suffix}"
    return original_name in sibling_names


def _list_document_files(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SOURCE_FILE_SUFFIXES
        and path.name not in SOURCE_FILE_SKIP_NAMES
    ]
    if not files:
        return []

    sibling_names = {path.name for path in files}
    originals = [path for path in files if not _is_page_split_derivative(path, sibling_names)]
    if originals:
        return sorted(originals, key=lambda path: _natural_key(path.name))

    return sorted(files, key=lambda path: _natural_key(path.name))


def _document_group_key(name: str) -> str:
    return name.strip().lower()


def _discover_module_source_files(module_root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for bucket in ("tasked", "submitted"):
        bucket_root = module_root / bucket
        if not bucket_root.exists():
            continue

        for path in sorted(bucket_root.iterdir(), key=lambda item: _natural_key(item.name)):
            if path.is_dir():
                document_files = _list_document_files(path)
            elif path.is_file() and path.suffix.lower() in SOURCE_FILE_SUFFIXES and path.name not in SOURCE_FILE_SKIP_NAMES:
                document_files = [path]
            else:
                continue

            for document_path in document_files:
                payload = _build_cache_file_payload(document_path, directory_root=module_root, bucket=bucket)
                payload["raw_name"] = document_path.name
                payload["name"] = _display_name(document_path.name)
                files.append(payload)

    grouped: dict[str, dict[str, object]] = {}
    for file in files:
        key = _document_group_key(str(file.get("raw_name") or file.get("name") or ""))
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {
                **file,
                "bucket": str(file.get("bucket") or ""),
                "copy_count": 1,
                "locations": [
                    {
                        "bucket": str(file.get("bucket") or ""),
                        "directory": str(file.get("directory") or ""),
                        "relative_path": str(file.get("relative_path") or ""),
                        "url": str(file.get("url") or ""),
                        "size_bytes": int(file.get("size_bytes") or 0),
                        "modified_at": float(file.get("modified_at") or 0),
                    }
                ],
            }
            continue

        locations = list(entry.get("locations") or [])
        locations.append(
            {
                "bucket": str(file.get("bucket") or ""),
                "directory": str(file.get("directory") or ""),
                "relative_path": str(file.get("relative_path") or ""),
                "url": str(file.get("url") or ""),
                "size_bytes": int(file.get("size_bytes") or 0),
                "modified_at": float(file.get("modified_at") or 0),
            }
        )
        entry["locations"] = locations
        entry["copy_count"] = len(locations)

        existing_bucket = str(entry.get("bucket") or "")
        new_bucket = str(file.get("bucket") or "")
        if existing_bucket != new_bucket:
            entry["bucket"] = "both"

        # Prefer the submitted copy when both exist because it reflects the user's own source cache.
        if existing_bucket == "tasked" and new_bucket == "submitted":
            for field in ("relative_path", "url", "directory", "size_bytes", "modified_at"):
                entry[field] = file.get(field)

    deduped_files = list(grouped.values())
    for entry in deduped_files:
        locations = list(entry.get("locations") or [])
        locations.sort(
            key=lambda location: (
                0 if str(location.get("bucket") or "") == "submitted" else 1,
                _natural_key(str(location.get("directory") or "")),
                _natural_key(str(location.get("relative_path") or "")),
            )
        )
        entry["locations"] = locations
        entry["copy_count"] = len(locations)
        entry["directory"] = " + ".join(str(location.get("directory") or "") for location in locations if location.get("directory"))

    deduped_files.sort(
        key=lambda file: (
            0 if str(file.get("bucket") or "") == "both" else 1,
            _natural_key(str(file.get("name") or "")),
        )
    )
    return deduped_files


def _cache_relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(CACHE_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _cache_url_from_path(path: Path) -> str | None:
    relative_path = _cache_relative_path(path)
    if relative_path is None:
        return None
    return f"/cache/{quote(relative_path, safe='/')}"


def _build_module_source_lookup(module_root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for file in _discover_module_source_files(module_root):
        url = str(file.get("url") or "").strip()
        if not url:
            continue
        for candidate in (
            str(file.get("raw_name") or "").strip(),
            str(file.get("name") or "").strip(),
            Path(str(file.get("raw_name") or "")).stem,
            Path(str(file.get("name") or "")).stem,
        ):
            if candidate and candidate.lower() not in lookup:
                lookup[candidate.lower()] = url
    return lookup


def _parse_answer_key_map(lines: list[str]) -> dict[int, str]:
    answer_map: dict[int, str] = {}
    for line in lines:
        match = QUIZ_ANSWER_KEY_RE.match(line.strip())
        if not match:
            continue
        for chunk in match.group("body").split(","):
            number_text, _, answer_text = chunk.partition(":")
            try:
                number = int(number_text.strip())
            except ValueError:
                continue
            answer = answer_text.strip().upper()
            if answer in {"A", "B", "C", "D"}:
                answer_map[number] = answer
    return answer_map


def _parse_objective_metadata(objective_text: str) -> dict[str, str]:
    cleaned = objective_text.strip()
    if not cleaned:
        return {
            "objective": "",
            "objective_label": "",
            "blueprint_slot_id": "",
            "topic": "",
            "skill": "",
            "difficulty": "",
        }

    parts = [part.strip() for part in cleaned.split(" | ")]
    first = parts[0]
    slot_match = re.search(r"\[([^\]]+)\]\s*$", first)
    blueprint_slot_id = slot_match.group(1).strip() if slot_match else ""
    objective_label = re.sub(r"\s*\[[^\]]+\]\s*$", "", first).strip()

    metadata = {
        "objective": cleaned,
        "objective_label": objective_label,
        "blueprint_slot_id": blueprint_slot_id,
        "topic": "",
        "skill": "",
        "difficulty": "",
    }
    for part in parts[1:]:
        if part.startswith("Topic:"):
            metadata["topic"] = part.removeprefix("Topic:").strip()
        elif part.startswith("Skill:"):
            metadata["skill"] = part.removeprefix("Skill:").strip()
        elif part.startswith("Difficulty:"):
            metadata["difficulty"] = part.removeprefix("Difficulty:").strip()
    return metadata


def _finalize_quiz_question(question: dict[str, object] | None, *, answer_map: dict[int, str]) -> dict[str, object] | None:
    if not isinstance(question, dict):
        return None
    choices = question.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return None
    number = question.get("number")
    if not isinstance(number, int):
        return None

    answer = answer_map.get(number, "")
    question["answer"] = answer if answer in {"A", "B", "C", "D"} else ""
    source_file = str(question.get("source_file") or "").strip()
    source_page = question.get("source_page")
    source_figure = str(question.get("source_figure_id") or "").strip()
    if source_file and isinstance(source_page, int) and source_page >= 1:
        source_label = f"Source: {source_file} p.{source_page}"
        if source_figure:
            source_label += f" {source_figure}"
        question["source_label"] = source_label
    else:
        question["source_label"] = ""
    return question


def _parse_quiz_markdown(quiz_path: Path) -> dict[str, object]:
    lines = quiz_path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    questions_start_index: int | None = None
    answer_key_start_index: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "## Questions":
            questions_start_index = index
        elif stripped == "## Answer Key" and answer_key_start_index is None:
            answer_key_start_index = index

        if questions_start_index is None:
            metadata_match = QUIZ_METADATA_RE.match(stripped)
            if metadata_match:
                metadata[metadata_match.group("key").strip().lower().replace(" ", "_")] = metadata_match.group("value").strip()

    if questions_start_index is None:
        raise RuntimeError("Quiz markdown does not include a Questions section.")

    answer_key_map = _parse_answer_key_map(lines[questions_start_index + 1 :])
    question_lines = lines[questions_start_index + 1 : answer_key_start_index if answer_key_start_index is not None else None]

    module_root = _module_root_for_quiz(quiz_path)
    source_lookup = _build_module_source_lookup(module_root) if module_root is not None and module_root.exists() else {}

    questions: list[dict[str, object]] = []
    current_question: dict[str, object] | None = None
    current_diagnostic_label: str | None = None
    current_diagnostic_map: dict[str, dict[str, str] | None] | None = None

    for raw_line in question_lines:
        stripped = raw_line.strip()
        if not stripped:
            current_diagnostic_label = None
            continue

        question_match = QUIZ_QUESTION_RE.match(stripped)
        if question_match:
            finalized = _finalize_quiz_question(current_question, answer_map=answer_key_map)
            if finalized is not None:
                questions.append(finalized)

            question_number = int(question_match.group("number"))
            source_file = str(question_match.group("source_file") or "").strip()
            source_page_text = question_match.group("source_page")
            source_page = int(source_page_text) if source_page_text and source_page_text.isdigit() else None
            source_figure_id = str(question_match.group("source_figure") or "").strip()
            source_url = source_lookup.get(source_file.lower(), "") if source_file else ""

            current_question = {
                "number": question_number,
                "question": question_match.group("question").strip(),
                "choices": [],
                "answer": "",
                "source_file": source_file,
                "source_page": source_page,
                "source_figure_id": source_figure_id,
                "source_url": source_url,
                "objective": "",
                "objective_label": "",
                "objective_topic": "",
                "objective_skill": "",
                "objective_difficulty": "",
                "blueprint_slot_id": "",
                "difficulty_dial": "",
                "actual_difficulty": "",
                "difficulty_review": "",
                "difficulty_feedback": "",
                "figure_label": "",
                "figure_url": "",
                "distractor_diagnostics": {},
            }
            current_diagnostic_map = current_question["distractor_diagnostics"] if isinstance(current_question["distractor_diagnostics"], dict) else None
            current_diagnostic_label = None
            continue

        if current_question is None:
            continue

        choice_match = QUIZ_CHOICE_RE.match(stripped)
        if choice_match:
            choices = current_question.get("choices")
            if isinstance(choices, list):
                choices.append(choice_match.group("text").strip())
            current_diagnostic_label = None
            continue

        if stripped.startswith("Objective:"):
            objective_text = stripped.removeprefix("Objective:").strip()
            objective_metadata = _parse_objective_metadata(objective_text)
            current_question["objective"] = objective_metadata["objective"]
            current_question["objective_label"] = objective_metadata["objective_label"]
            current_question["objective_topic"] = objective_metadata["topic"]
            current_question["objective_skill"] = objective_metadata["skill"]
            current_question["objective_difficulty"] = objective_metadata["difficulty"]
            current_question["blueprint_slot_id"] = objective_metadata["blueprint_slot_id"]
            continue
        if stripped.startswith("Difficulty Dial:"):
            current_question["difficulty_dial"] = stripped.removeprefix("Difficulty Dial:").strip()
            continue
        if stripped.startswith("Actual Difficulty:"):
            current_question["actual_difficulty"] = stripped.removeprefix("Actual Difficulty:").strip()
            continue
        if stripped.startswith("Difficulty Review:"):
            current_question["difficulty_review"] = stripped.removeprefix("Difficulty Review:").strip()
            continue
        if stripped.startswith("Difficulty Feedback:"):
            current_question["difficulty_feedback"] = stripped.removeprefix("Difficulty Feedback:").strip()
            continue
        if stripped.startswith("Figure:"):
            figure_text = stripped.removeprefix("Figure:").strip()
            figure_match = re.match(r"^\[(?P<label>.+?)\]\((?P<path>.+)\)$", figure_text)
            if figure_match:
                figure_label = figure_match.group("label").strip()
                figure_path = figure_match.group("path").strip()
                resolved_figure_path = Path(figure_path)
                if not resolved_figure_path.is_absolute():
                    resolved_figure_path = (quiz_path.parent / figure_path).resolve()
                current_question["figure_label"] = figure_label
                current_question["figure_url"] = _cache_url_from_path(resolved_figure_path) or ""
            continue
        if stripped == "Distractor Diagnostics:":
            current_diagnostic_label = None
            continue

        diagnostic_match = QUIZ_DIAGNOSTIC_RE.match(stripped)
        if diagnostic_match and isinstance(current_diagnostic_map, dict):
            option_label = diagnostic_match.group("label").strip().upper()
            body_text = diagnostic_match.group("body").strip()
            label_match = QUIZ_DIAGNOSTIC_ID_RE.match(body_text)
            misconception_label = body_text
            misconception_id = ""
            if label_match:
                misconception_label = label_match.group("label").strip()
                misconception_id = label_match.group("misconception_id").strip()

            current_diagnostic_map[option_label] = {
                "misconception_label": misconception_label,
                "misconception_id": misconception_id,
                "why_student_might_pick": "",
                "why_wrong": "",
            }
            current_diagnostic_label = option_label
            continue

        if stripped.startswith("Why chosen:") and current_diagnostic_label and isinstance(current_diagnostic_map, dict):
            diagnostic_entry = current_diagnostic_map.get(current_diagnostic_label)
            if isinstance(diagnostic_entry, dict):
                diagnostic_entry["why_student_might_pick"] = stripped.removeprefix("Why chosen:").strip()
            continue

        if stripped.startswith("Why wrong:") and current_diagnostic_label and isinstance(current_diagnostic_map, dict):
            diagnostic_entry = current_diagnostic_map.get(current_diagnostic_label)
            if isinstance(diagnostic_entry, dict):
                diagnostic_entry["why_wrong"] = stripped.removeprefix("Why wrong:").strip()
            continue

    finalized = _finalize_quiz_question(current_question, answer_map=answer_key_map)
    if finalized is not None:
        questions.append(finalized)

    return {
        "quiz": _build_cache_file_payload(quiz_path),
        "metadata": metadata,
        "question_count": len(questions),
        "questions": questions,
        "answer_key": answer_key_map,
    }


def _build_json_body(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _parse_json_request_body(handler: SimpleHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length") or 0)
    if content_length <= 0:
        return {}

    raw_body = handler.rfile.read(content_length)
    if not raw_body:
        return {}

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _coerce_question_count(payload: dict[str, object]) -> int:
    raw_value = payload.get("question_count", DEFAULT_GENERATION_QUESTION_COUNT)
    if isinstance(raw_value, bool):
        raise ValueError("question_count must be a number between 1 and 30.")

    try:
        question_count = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("question_count must be a number between 1 and 30.") from exc

    if 1 <= question_count <= 30:
        return question_count
    raise ValueError("question_count must be a number between 1 and 30.")


def _normalize_answer_list(raw_answers: object, *, question_count: int) -> list[str | None]:
    answers = raw_answers if isinstance(raw_answers, list) else []
    normalized: list[str | None] = []
    for index in range(question_count):
        value = answers[index] if index < len(answers) else None
        normalized_value = str(value or "").strip().upper()
        normalized.append(normalized_value if normalized_value in {"A", "B", "C", "D"} else None)
    return normalized


def _difficulty_target_for_question(question: dict[str, object]) -> str:
    objective_difficulty = str(question.get("objective_difficulty") or "").strip().lower()
    if objective_difficulty in DIFFICULTY_ORDER:
        return objective_difficulty

    actual_difficulty = str(question.get("actual_difficulty") or "").strip().lower()
    if actual_difficulty:
        actual_label = actual_difficulty.split(" ", 1)[0]
        if actual_label in DIFFICULTY_ORDER:
            return actual_label

    return "standard"


def _find_tasked_summary_path(module_root: Path) -> Path | None:
    tasked_root = module_root / "tasked"
    for filename in TASKED_SUMMARY_FILENAMES:
        candidate = tasked_root / filename
        if candidate.is_file():
            return candidate
    return None


def _find_submitted_summary_path(tasked_summary_path: Path) -> Path | None:
    module_root = tasked_summary_path.parent.parent
    candidate = module_root / "submitted" / "submitted_assignments.json"
    return candidate if candidate.is_file() else None


def _diagnostic_namespace_map(question: dict[str, object]) -> dict[str, object]:
    diagnostics = question.get("distractor_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}

    mapped: dict[str, object] = {}
    for option, row in diagnostics.items():
        if not isinstance(row, dict):
            continue
        mapped[str(option).strip().upper()] = SimpleNamespace(
            summary=str(row.get("misconception_label") or "").strip(),
            misconception_id=str(row.get("misconception_id") or "").strip(),
            why_chosen=str(row.get("why_student_might_pick") or "").strip(),
            why_wrong=str(row.get("why_wrong") or "").strip(),
        )
    return mapped


def _question_namespace(question: dict[str, object]) -> SimpleNamespace:
    choices_raw = question.get("choices")
    choices = (
        {
            label: str(choices_raw[index] or "").strip()
            for index, label in enumerate(("A", "B", "C", "D"))
            if isinstance(choices_raw, list) and index < len(choices_raw)
        }
        if isinstance(choices_raw, list)
        else {}
    )
    return SimpleNamespace(
        number=int(question.get("number") or 0),
        text=str(question.get("question") or "").strip(),
        source=str(question.get("source_label") or "").strip(),
        source_file=str(question.get("source_file") or "").strip(),
        source_page=int(question.get("source_page") or 1),
        source_figure_id=str(question.get("source_figure_id") or "").strip(),
        correct_answer=str(question.get("answer") or "").strip().upper(),
        choices=choices,
        objective=str(question.get("objective") or "").strip(),
        objective_label=str(question.get("objective_label") or "").strip(),
        blueprint_slot_id=str(question.get("blueprint_slot_id") or "").strip(),
        topic=str(question.get("objective_topic") or "").strip(),
        skill=str(question.get("objective_skill") or "").strip(),
        difficulty=str(question.get("objective_difficulty") or "").strip(),
        actual_difficulty=str(question.get("actual_difficulty") or "").strip(),
        distractor_diagnostics=_diagnostic_namespace_map(question),
    )


def _build_adaptive_specs(
    *,
    quiz_payload: dict[str, object],
    answers: list[str | None],
    module_root: Path,
) -> list[dict[str, object]]:
    questions = quiz_payload.get("questions")
    if not isinstance(questions, list):
        return []

    excluded_question_texts = [
        str(question.get("question") or "").strip()
        for question in questions
        if isinstance(question, dict) and str(question.get("question") or "").strip()
    ]

    specs: list[dict[str, object]] = []
    for index, raw_question in enumerate(questions):
        if not isinstance(raw_question, dict):
            continue

        selected_option = answers[index] if index < len(answers) else None
        correct_answer = str(raw_question.get("answer") or "").strip().upper()
        if selected_option is None or selected_option == correct_answer:
            continue

        question_object = _question_namespace(raw_question)
        evidence = study_engine.extract_source_evidence(module_root, question_object)
        metadata = study_engine.question_metadata(question_object)
        diagnostics = raw_question.get("distractor_diagnostics")
        selected_diagnostic = (
            diagnostics.get(selected_option)
            if isinstance(diagnostics, dict) and isinstance(diagnostics.get(selected_option), dict)
            else {}
        )

        misconception_label = str(selected_diagnostic.get("misconception_label") or "").strip()
        why_wrong = str(selected_diagnostic.get("why_wrong") or "").strip()
        misconception_summary = misconception_label or why_wrong or metadata.get("concept_label") or ""

        specs.append(
            {
                "spec_index": len(specs) + 1,
                "concept_label": str(metadata.get("concept_label") or "").strip() or f"Question {index + 1}",
                "difficulty_target": _difficulty_target_for_question(raw_question),
                "source_file": str(evidence.get("source_file") or raw_question.get("source_file") or "Unknown source").strip(),
                "source_page": int(evidence.get("source_page") or raw_question.get("source_page") or 1),
                "source_text": str(evidence.get("page_text") or "").strip(),
                "evidence_lines": list(evidence.get("evidence_lines") or []),
                "misconception_id": str(selected_diagnostic.get("misconception_id") or "").strip(),
                "misconception_summary": misconception_summary,
                "excluded_question_texts": excluded_question_texts,
            }
        )
    return specs


def _inject_adaptive_metadata(
    markdown_text: str,
    *,
    source_quiz_relative_path: str,
    wrong_question_count: int,
    targeted_misconception_count: int,
) -> str:
    lines = markdown_text.splitlines()
    insert_index = next(
        (index + 1 for index, line in enumerate(lines) if line.startswith("- Question Count:")),
        6,
    )
    extra_lines = [
        "- Quiz Type: Adaptive Review",
        f"- Adaptive Source Quiz: {source_quiz_relative_path}",
        f"- Trigger Question Count: {wrong_question_count}",
        f"- Targeted Misconceptions: {targeted_misconception_count}",
    ]
    lines[insert_index:insert_index] = extra_lines
    return "\n".join(lines).strip() + "\n"


def _save_adaptive_markdown(
    *,
    questions: list[dict[str, object]],
    tasked_summary_path: Path,
    course_name: str,
    module_name: str,
    source_quiz_relative_path: str,
    wrong_question_count: int,
    targeted_misconception_count: int,
) -> Path:
    output_dir = _mcq_output_dir_from_summary_path(tasked_summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")
    generated_label = timestamp.isoformat()

    markdown_text = render_mcqs_markdown(
        questions,
        course_name=course_name,
        module_name=f"{module_name} Adaptive Review".strip(),
        generated_at=generated_label,
        report={},
    )
    markdown_text = _inject_adaptive_metadata(
        markdown_text,
        source_quiz_relative_path=source_quiz_relative_path,
        wrong_question_count=wrong_question_count,
        targeted_misconception_count=targeted_misconception_count,
    )

    timestamped_path = output_dir / f"mcq_adaptive_{timestamp_label}.md"
    latest_path = output_dir / "mcq_adaptive_latest.md"
    timestamped_path.write_text(markdown_text, encoding="utf-8")
    latest_path.write_text(markdown_text, encoding="utf-8")
    return timestamped_path


def _run_generation_process(*, question_count: int) -> tuple[int, dict[str, object]]:
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--run-latest-cache",
        "--question-count",
        str(question_count),
        "--json",
        "--progress-to-stderr",
    ]
    return _run_main_process(command)


def _run_main_process(command: list[str]) -> tuple[int, dict[str, object]]:
    started_at = time.perf_counter()
    print(f"[frontend] Starting generation command: {' '.join(command)}", file=sys.stderr, flush=True)
    completed = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_lines: list[str] = []

    def _pump_stderr() -> None:
        if completed.stderr is None:
            return
        for raw_line in completed.stderr:
            line = raw_line.rstrip()
            stderr_lines.append(line)
            if line:
                print(f"[generation] {line}", file=sys.stderr, flush=True)

    stderr_thread = threading.Thread(target=_pump_stderr, daemon=True)
    stderr_thread.start()

    stdout_text = completed.stdout.read().strip() if completed.stdout is not None else ""
    returncode = completed.wait()
    stderr_thread.join()
    elapsed_seconds = time.perf_counter() - started_at
    print(
        f"[frontend] Generation command finished with exit code {returncode} in {elapsed_seconds:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    stderr_text = "\n".join(line for line in stderr_lines if line).strip()
    result_payload: dict[str, object] = {}
    if stdout_text:
        try:
            parsed = json.loads(stdout_text)
            if isinstance(parsed, dict):
                result_payload = parsed
        except json.JSONDecodeError:
            result_payload = {}

    if returncode != 0:
        error_message = (
            str(result_payload.get("error") or "").strip()
            or stderr_text
            or stdout_text
            or "Quiz generation failed."
        )
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": False,
            "error": error_message,
        }

    if not result_payload:
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "success": False,
            "error": "Quiz generation completed but returned no JSON result.",
        }

    return HTTPStatus.OK, result_payload


def _coerce_course_id(raw_value: str) -> int:
    try:
        course_id = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("course_id must be a valid Canvas course ID.") from exc
    if course_id <= 0:
        raise ValueError("course_id must be a valid Canvas course ID.")
    return course_id


@contextlib.contextmanager
def _canvas_browser_context():
    canvas.load_dotenv()
    username = canvas.pick_env("LCDS_USERNAME", "CANVAS_USERNAME", "USERNAME")
    password = canvas.pick_env("LCDS_PASSWORD", "CANVAS_PASSWORD", "PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing Canvas credentials. Add LCDS_USERNAME/LCDS_PASSWORD or CANVAS_USERNAME/CANVAS_PASSWORD to .env."
        )

    resolved_headless = canvas.as_bool(canvas.os.getenv("HEADLESS"), default=True)
    with canvas.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=resolved_headless)
        try:
            context = browser.new_context()
            page = context.new_page()
            canvas.login_to_canvas(page, username, password)
            yield context
        finally:
            browser.close()


def _fetch_canvas_courses() -> list[dict[str, object]]:
    with _canvas_browser_context() as context:
        courses = canvas.fetch_active_courses(context)

    payload: list[dict[str, object]] = []
    for course in courses:
        payload.append(
            {
                "id": course.course_id,
                "name": course.name,
            }
        )
    return payload


def _fetch_canvas_modules(course_id: int) -> list[dict[str, object]]:
    with _canvas_browser_context() as context:
        module_rows = canvas.fetch_course_modules_with_items(context, course_id)
        modules = canvas.sort_modules_for_selection(module_rows)

    payload: list[dict[str, object]] = []
    for module in modules:
        module_id = module.get("id")
        if not isinstance(module_id, int):
            continue
        payload.append(
            {
                "id": module_id,
                "name": canvas.module_name_from_payload(module),
                "position": module.get("position"),
                "item_count": len(module.get("items") or []) if isinstance(module.get("items"), list) else 0,
            }
        )
    return payload


def _parse_multipart_form_data(handler: SimpleHTTPRequestHandler) -> tuple[dict[str, str], list[dict[str, object]]]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")

    content_length = int(handler.headers.get("Content-Length") or 0)
    if content_length <= 0:
        raise ValueError("Upload body is empty.")

    raw_body = handler.rfile.read(content_length)
    parser_input = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + raw_body
    )
    message = BytesParser(policy=email_policy_default).parsebytes(parser_input)
    if not message.is_multipart():
        raise ValueError("Upload body was not valid multipart/form-data.")

    fields: dict[str, str] = {}
    files: list[dict[str, object]] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        if not field_name:
            continue

        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append(
                {
                    "field_name": field_name,
                    "filename": str(filename),
                    "content": payload,
                }
            )
            continue

        charset = part.get_content_charset() or "utf-8"
        fields[field_name] = payload.decode(charset, errors="replace")

    return fields, files


def _write_uploaded_files(uploaded_files: list[dict[str, object]], temp_root: Path) -> int:
    saved_count = 0
    for item in uploaded_files:
        filename = str(item.get("filename") or "").strip()
        content = item.get("content")
        if not filename or not isinstance(content, (bytes, bytearray)):
            continue

        relative_name = Path(filename.replace("\\", "/"))
        safe_parts = [part for part in relative_name.parts if part not in {"", ".", ".."}]
        if not safe_parts:
            continue

        target_path = temp_root.joinpath(*safe_parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            handle.write(bytes(content))
        saved_count += 1

    return saved_count


class FrontendRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - inherited interface
        self._dispatch_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - inherited interface
        self._dispatch_request(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - inherited interface
        parsed = urlparse(self.path)
        if parsed.path == "/api/generation/run":
            self._run_generation(send_body=True)
            return
        if parsed.path == "/api/generation/local-upload":
            self._run_local_upload_generation(send_body=True)
            return
        if parsed.path == "/api/generation/adaptive":
            self._run_adaptive_generation(send_body=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unsupported POST endpoint")

    def _dispatch_request(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/cache/markdown-files":
            self._serve_cache_index(send_body=send_body)
            return

        if parsed.path == "/api/cache/quiz-sources":
            self._serve_quiz_sources(parse_qs(parsed.query), send_body=send_body)
            return

        if parsed.path == "/api/quiz":
            self._serve_quiz_detail(parse_qs(parsed.query), send_body=send_body)
            return

        if parsed.path == "/api/canvas/courses":
            self._serve_canvas_courses(send_body=send_body)
            return

        if parsed.path == "/api/canvas/modules":
            self._serve_canvas_modules(parse_qs(parsed.query), send_body=send_body)
            return

        if parsed.path.startswith("/cache/"):
            self._serve_cache_file(parsed.path.removeprefix("/cache/"), send_body=send_body)
            return

        if parsed.path == "/":
            self.path = "/home.html"
        else:
            self.path = parsed.path

        if send_body:
            super().do_GET()
            return

        super().do_HEAD()

    def _send_json_response(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
        *,
        send_body: bool,
    ) -> None:
        body = _build_json_body(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _serve_cache_index(self, *, send_body: bool) -> None:
        payload = {
            "cache_root": str(CACHE_ROOT),
            "count": 0,
            "files": [],
        }
        files = _discover_markdown_files(CACHE_ROOT)
        payload["count"] = len(files)
        payload["files"] = files

        self._send_json_response(HTTPStatus.OK, payload, send_body=send_body)

    def _serve_quiz_sources(self, query: dict[str, list[str]], *, send_body: bool) -> None:
        relative_path = (query.get("path") or [""])[0].strip()
        if not relative_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing quiz path")
            return

        quiz_path = _resolve_cache_file(relative_path)
        if quiz_path is None or quiz_path.suffix.lower() != ".md":
            self.send_error(HTTPStatus.NOT_FOUND, "Quiz file not found")
            return

        module_root = _module_root_for_quiz(quiz_path)
        if module_root is None or not module_root.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Module cache not found for quiz")
            return

        source_files = _discover_module_source_files(module_root)
        payload = {
            "quiz": _build_cache_file_payload(quiz_path),
            "module_directory": module_root.relative_to(CACHE_ROOT).as_posix(),
            "source_count": len(source_files),
            "source_files": source_files,
        }

        self._send_json_response(HTTPStatus.OK, payload, send_body=send_body)

    def _serve_quiz_detail(self, query: dict[str, list[str]], *, send_body: bool) -> None:
        relative_path = (query.get("path") or [""])[0].strip()
        if not relative_path:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": "Missing quiz path."},
                send_body=send_body,
            )
            return

        quiz_path = _resolve_cache_file(relative_path)
        if quiz_path is None or quiz_path.suffix.lower() != ".md":
            self._send_json_response(
                HTTPStatus.NOT_FOUND,
                {"success": False, "error": "Quiz file not found."},
                send_body=send_body,
            )
            return

        try:
            payload = _parse_quiz_markdown(quiz_path)
        except Exception as exc:  # noqa: BLE001
            self._send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return

        self._send_json_response(HTTPStatus.OK, payload, send_body=send_body)

    def _serve_canvas_courses(self, *, send_body: bool) -> None:
        try:
            payload = {
                "count": 0,
                "courses": _fetch_canvas_courses(),
            }
            payload["count"] = len(payload["courses"])
        except Exception as exc:  # noqa: BLE001
            self._send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return

        self._send_json_response(HTTPStatus.OK, payload, send_body=send_body)

    def _serve_canvas_modules(self, query: dict[str, list[str]], *, send_body: bool) -> None:
        raw_course_id = (query.get("course_id") or [""])[0].strip()
        if not raw_course_id:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": "Missing course_id query parameter."},
                send_body=send_body,
            )
            return

        try:
            course_id = _coerce_course_id(raw_course_id)
            payload = {
                "course_id": course_id,
                "count": 0,
                "modules": _fetch_canvas_modules(course_id),
            }
            payload["count"] = len(payload["modules"])
        except ValueError as exc:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return

        self._send_json_response(HTTPStatus.OK, payload, send_body=send_body)

    def _run_generation(self, *, send_body: bool) -> None:
        try:
            payload = _parse_json_request_body(self)
            question_count = _coerce_question_count(payload)
        except ValueError as exc:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return

        source_mode = str(payload.get("source_mode") or "cache").strip().lower()
        if source_mode not in {"cache", "canvas"}:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": "source_mode must be either 'cache' or 'canvas'."},
                send_body=send_body,
            )
            return

        if not GENERATION_LOCK.acquire(blocking=False):
            self._send_json_response(
                HTTPStatus.CONFLICT,
                {"success": False, "error": "A quiz generation run is already in progress."},
                send_body=send_body,
            )
            return

        try:
            if source_mode == "canvas":
                course_selection = str(payload.get("course_selection") or "").strip()
                module_selection = str(payload.get("module_selection") or "").strip()
                if not course_selection or not module_selection:
                    self._send_json_response(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "success": False,
                            "error": "course_selection and module_selection are required for Canvas generation.",
                        },
                        send_body=send_body,
                    )
                    return

                command = [
                    sys.executable,
                    str(REPO_ROOT / "main.py"),
                    "--source-mode",
                    "canvas",
                    "--course-selection",
                    course_selection,
                    "--module-selection",
                    module_selection,
                    "--question-count",
                    str(question_count),
                    "--json",
                    "--progress-to-stderr",
                ]
                status, result_payload = _run_main_process(command)
            else:
                status, result_payload = _run_generation_process(question_count=question_count)
        finally:
            GENERATION_LOCK.release()

        self._send_json_response(status, result_payload, send_body=send_body)

    def _run_local_upload_generation(self, *, send_body: bool) -> None:
        if not GENERATION_LOCK.acquire(blocking=False):
            self._send_json_response(
                HTTPStatus.CONFLICT,
                {"success": False, "error": "A quiz generation run is already in progress."},
                send_body=send_body,
            )
            return

        try:
            try:
                fields, uploaded_files = _parse_multipart_form_data(self)
            except ValueError as exc:
                self._send_json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"success": False, "error": str(exc)},
                    send_body=send_body,
                )
                return

            study_set_name = str(fields.get("study_set_name") or "").strip()
            if not study_set_name:
                self._send_json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"success": False, "error": "study_set_name is required."},
                    send_body=send_body,
                )
                return

            try:
                question_count = _coerce_question_count(
                    {"question_count": fields.get("question_count", DEFAULT_GENERATION_QUESTION_COUNT)}
                )
            except ValueError as exc:
                self._send_json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"success": False, "error": str(exc)},
                    send_body=send_body,
                )
                return

            with tempfile.TemporaryDirectory(prefix="sqg-upload-", dir=str(REPO_ROOT)) as temp_dir:
                upload_root = Path(temp_dir)
                saved_count = _write_uploaded_files(uploaded_files, upload_root)
                if saved_count < 1:
                    self._send_json_response(
                        HTTPStatus.BAD_REQUEST,
                        {"success": False, "error": "Upload at least one source file to generate a quiz."},
                        send_body=send_body,
                    )
                    return

                command = [
                    sys.executable,
                    str(REPO_ROOT / "main.py"),
                    "--source-mode",
                    "local",
                    "--source-dir",
                    str(upload_root),
                    "--study-set-name",
                    study_set_name,
                    "--question-count",
                    str(question_count),
                    "--json",
                    "--progress-to-stderr",
                ]
                status, result_payload = _run_main_process(command)
        finally:
            GENERATION_LOCK.release()

        self._send_json_response(status, result_payload, send_body=send_body)

    def _run_adaptive_generation(self, *, send_body: bool) -> None:
        try:
            payload = _parse_json_request_body(self)
        except ValueError as exc:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
            return

        relative_path = str(payload.get("quiz_path") or "").strip()
        if not relative_path:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"success": False, "error": "quiz_path is required."},
                send_body=send_body,
            )
            return

        quiz_path = _resolve_cache_file(relative_path)
        if quiz_path is None or quiz_path.suffix.lower() != ".md":
            self._send_json_response(
                HTTPStatus.NOT_FOUND,
                {"success": False, "error": "Quiz file not found."},
                send_body=send_body,
            )
            return

        if not GENERATION_LOCK.acquire(blocking=False):
            self._send_json_response(
                HTTPStatus.CONFLICT,
                {"success": False, "error": "A quiz generation run is already in progress."},
                send_body=send_body,
            )
            return

        started_at = time.perf_counter()
        print(
            f"[frontend] Starting adaptive quiz generation for {relative_path}",
            file=sys.stderr,
            flush=True,
        )

        try:
            quiz_payload = _parse_quiz_markdown(quiz_path)
            question_count = int(quiz_payload.get("question_count") or 0)
            answers = _normalize_answer_list(payload.get("answers"), question_count=question_count)
            module_root = _module_root_for_quiz(quiz_path)
            if module_root is None or not module_root.exists():
                raise RuntimeError("Module cache could not be resolved for the selected quiz.")

            print("[frontend] Building remediation specs from completed attempt...", file=sys.stderr, flush=True)
            remediation_specs = _build_adaptive_specs(
                quiz_payload=quiz_payload,
                answers=answers,
                module_root=module_root,
            )
            if not remediation_specs:
                raise RuntimeError(
                    "Adaptive quiz generation requires at least one incorrect answer in the completed attempt."
                )

            metadata = quiz_payload.get("metadata") if isinstance(quiz_payload.get("metadata"), dict) else {}
            course_name = str(metadata.get("course") or "").strip() or "Unknown course"
            module_name = str(metadata.get("module") or "").strip() or "Unknown module"
            targeted_misconceptions = {
                str(spec.get("misconception_id") or "").strip()
                for spec in remediation_specs
                if str(spec.get("misconception_id") or "").strip()
            }

            print(
                "[frontend] Generating adaptive remediation questions "
                f"for {len(remediation_specs)} missed question(s)...",
                file=sys.stderr,
                flush=True,
            )
            generated_rows = gemma.generate_batch_remediation_questions(
                course_name=course_name,
                module_name=module_name,
                remediation_specs=remediation_specs,
                question_kind="adaptive_review",
            )

            tasked_summary_path = _find_tasked_summary_path(module_root)
            if tasked_summary_path is None:
                raise RuntimeError("Tasked module summary could not be found for this quiz.")

            saved_path = _save_adaptive_markdown(
                questions=generated_rows,
                tasked_summary_path=tasked_summary_path,
                course_name=course_name,
                module_name=module_name,
                source_quiz_relative_path=relative_path,
                wrong_question_count=len(remediation_specs),
                targeted_misconception_count=len(targeted_misconceptions),
            )
            saved_relative_path = _cache_relative_path(saved_path)
            if saved_relative_path is None:
                raise RuntimeError("Adaptive quiz was saved outside the Cache directory.")

            elapsed_seconds = time.perf_counter() - started_at
            print(
                f"[frontend] Adaptive quiz generation finished in {elapsed_seconds:.1f}s: {saved_relative_path}",
                file=sys.stderr,
                flush=True,
            )
            self._send_json_response(
                HTTPStatus.OK,
                {
                    "success": True,
                    "question_count": len(generated_rows),
                    "markdown_path": str(saved_path),
                    "relative_markdown_path": saved_relative_path,
                    "source_quiz_relative_path": relative_path,
                    "targeted_misconception_count": len(targeted_misconceptions),
                    "trigger_question_count": len(remediation_specs),
                },
                send_body=send_body,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_seconds = time.perf_counter() - started_at
            print(
                f"[frontend] Adaptive quiz generation failed after {elapsed_seconds:.1f}s: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"success": False, "error": str(exc)},
                send_body=send_body,
            )
        finally:
            GENERATION_LOCK.release()

    def _serve_cache_file(self, relative_path: str, *, send_body: bool) -> None:
        target = _resolve_cache_file(unquote(relative_path))
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Cache file not found")
            return

        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        if content_type is None and target.suffix.lower() == ".md":
            content_type = "text/markdown"
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type.startswith("text/"):
            content_type = f"{content_type}; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the static frontend with live Cache markdown data.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FrontendRequestHandler)
    print(f"Serving frontend from {FRONTEND_ROOT} at http://{args.host}:{args.port}")
    print(f"Reading markdown files from {CACHE_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
