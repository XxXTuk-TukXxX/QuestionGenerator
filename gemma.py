from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from QG.Distractors.misconception_mining import build_or_load_module_misconceptions
from models.gemma_blueprint import (
    _attach_difficulty_review_to_questions,
    _build_blueprint_text,
    _combine_warning_messages,
    _difficulty_dial_guidance_text,
    _difficulty_guidance,
    _evaluate_set_blueprint_alignment,
    _evaluate_set_depth,
    _generate_question_blueprint,
    _load_difficulty_dials,
    _normalize_blueprint_difficulty,
    _normalize_blueprint_skill,
)
from models.gemma_common import (
    DIFFICULTY_CALIBRATION_PATH,
    GOOGLE_IMPORTANCE_MODEL,
    IMAGE_GROUNDED_RATIO,
    LAYOUT_CACHE_ENABLED,
    MAX_FIGURE_IMAGE_PARTS_FOR_PROMPT,
    clean_choice_text as _clean_choice_text,
    clean_generated_text as _clean_generated_text,
    extract_json_object as _extract_json_object,
    load_summary_payload,
    load_tasked_items_summary,
    merge_tasked_and_submitted_items as _merge_tasked_and_submitted_items,
    normalize_positive_int as _normalize_positive_int,
    normalize_text as _normalize_text,
)
from models.gemma_output import (
    render_mcqs_markdown as _render_mcqs_markdown,
    save_mcqs_markdown as _save_mcqs_markdown,
)
from models.gemma_sources import (
    _attach_figure_image_paths,
    _build_tagged_study_text,
    _classify_figures_with_gemma,
    _layout_cache_dir_from_summary_path,
    _load_figure_image_map,
    _write_layout_cache,
    collect_tasked_module_text,
)

load_dotenv()
DEFAULT_GEMINI_MODEL = str(os.getenv("MODEL") or "gemini-2.5-flash").strip() or "gemini-2.5-flash"
model = DEFAULT_GEMINI_MODEL
LAST_GENERATION_REPORT: dict[str, Any] = {}
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


def _resolve_env_setting(explicit_value: str | None, *env_names: str, fallback: str) -> str:
    resolved = str(explicit_value or "").strip()
    if resolved:
        return resolved

    for env_name in env_names:
        candidate = str(os.getenv(env_name) or "").strip()
        if candidate:
            return candidate
    return fallback


def _is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message or "not_found" in message or "not found" in message


def generate_batch_remediation_questions(
    *,
    course_name: str,
    module_name: str,
    remediation_specs: list[dict[str, Any]],
    question_kind: str = "repair_test",
    model: str | None = None,
) -> list[dict[str, Any]]:
    normalized_specs = [spec for spec in remediation_specs if isinstance(spec, dict)]
    if not normalized_specs:
        return []

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")
    resolved_model = _resolve_env_setting(
        model,
        "GEMINI_REMEDIATION_MODEL",
        "MODEL",
        fallback=DEFAULT_GEMINI_MODEL,
    )

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
    candidate_models: list[str] = []
    for candidate in (
        resolved_model,
        str(os.getenv("MODEL") or "").strip(),
        DEFAULT_GEMINI_MODEL,
        "gemini-2.5-flash",
    ):
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

    last_error: Exception | None = None
    response = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You generate one source-grounded remediation MCQ per spec and return JSON only."
                    ),
                    response_mime_type="application/json",
                    temperature=0.25,
                ),
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not _is_model_not_found_error(exc):
                raise RuntimeError(
                    f"Batch remediation Gemma request failed with model '{model_name}': {exc}"
                ) from exc

    if response is None:
        attempted = ", ".join(candidate_models) if candidate_models else "none"
        raise RuntimeError(
            "Batch remediation Gemma request failed after trying model(s): "
            f"{attempted}. Last error: {last_error}"
        ) from last_error

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


def _fallback_diagnostic_label(
    *,
    misconception_id: str,
    misconception_source: dict[str, Any],
) -> str:
    label = str(misconception_source.get("misconception_label") or "").strip()
    misconception_text = str(misconception_source.get("misconception") or "").strip()
    correct_idea = str(misconception_source.get("correct_idea") or "").strip()

    if len(label) >= 18 and not _is_generic_diagnostic_text(label):
        return label
    if len(misconception_text) >= 18 and not _is_generic_diagnostic_text(misconception_text):
        return misconception_text
    if correct_idea:
        return f"{misconception_id}: confusing the concept with {correct_idea}"
    return f"{misconception_id}: distractor-specific misconception pattern"


def _build_fallback_diagnostic_entry(
    *,
    option: str,
    choice_text: str,
    misconception_id: str,
    misconception_source: dict[str, Any],
) -> dict[str, str]:
    misconception_text = str(misconception_source.get("misconception") or "").strip()
    correct_idea = str(misconception_source.get("correct_idea") or "").strip()
    choice_fragment = choice_text.strip() or f"option {option}"
    misconception_fragment = misconception_text or "the misconception described in the evidence bank"
    correct_fragment = correct_idea or "the correct idea supported by the source evidence"

    return {
        "misconception_id": misconception_id,
        "misconception_label": _fallback_diagnostic_label(
            misconception_id=misconception_id,
            misconception_source=misconception_source,
        ),
        "why_student_might_pick": (
            f"A student might choose {option} because '{choice_fragment}' sounds consistent with "
            f"{misconception_fragment}, so the distractor appears to fit the topic at first glance."
        ),
        "why_wrong": (
            f"Option {option} is wrong because the source supports {correct_fragment}, whereas "
            f"'{choice_fragment}' reflects {misconception_fragment} instead of the correct reasoning."
        ),
    }


def _build_fallback_diagnostics_for_question(
    *,
    question: dict[str, Any],
    misconception_lookup: dict[str, dict[str, Any]],
    valid_misconception_ids: set[str],
) -> dict[str, dict[str, str] | None] | None:
    answer = str(question.get("answer") or "").strip().upper()
    choices = question.get("choices")
    diagnostics_raw = question.get("distractor_diagnostics")
    diagnostics_source = diagnostics_raw if isinstance(diagnostics_raw, dict) else {}

    if answer not in {"A", "B", "C", "D"}:
        return None
    if not isinstance(choices, list) or len(choices) < 4:
        return None

    available_ids = sorted(valid_misconception_ids)
    if len(available_ids) < 3:
        return None

    fallback: dict[str, dict[str, str] | None] = {}
    used_ids: set[str] = set()
    for option, choice_text in zip(("A", "B", "C", "D"), choices[:4]):
        if option == answer:
            fallback[option] = None
            continue

        existing_row = diagnostics_source.get(option) if isinstance(diagnostics_source.get(option), dict) else {}
        preferred_id = str(existing_row.get("misconception_id") or "").strip()
        if preferred_id not in valid_misconception_ids or preferred_id in used_ids:
            preferred_id = ""

        if not preferred_id:
            for candidate_id in available_ids:
                if candidate_id not in used_ids:
                    preferred_id = candidate_id
                    break

        if not preferred_id:
            return None

        used_ids.add(preferred_id)
        fallback[option] = _build_fallback_diagnostic_entry(
            option=option,
            choice_text=str(choice_text).strip(),
            misconception_id=preferred_id,
            misconception_source=misconception_lookup.get(preferred_id) or {},
        )

    if not _validate_distractor_diagnostics_specificity(
        fallback,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    ):
        return None
    return fallback


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
        return _build_fallback_diagnostics_for_question(
            question=question,
            misconception_lookup=misconception_lookup,
            valid_misconception_ids=valid_misconception_ids,
        )

    response_text = response.text if isinstance(response.text, str) else ""
    payload = _extract_json_object(response_text)
    diagnostics = _normalize_distractor_diagnostics(
        payload,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    )
    if diagnostics is None:
        return _build_fallback_diagnostics_for_question(
            question=question,
            misconception_lookup=misconception_lookup,
            valid_misconception_ids=valid_misconception_ids,
        )
    if not _validate_distractor_diagnostics_specificity(
        diagnostics,
        answer=answer,
        valid_misconception_ids=valid_misconception_ids,
    ):
        return _build_fallback_diagnostics_for_question(
            question=question,
            misconception_lookup=misconception_lookup,
            valid_misconception_ids=valid_misconception_ids,
        )
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
        raise RuntimeError("Gemma response JSON had an unexpected shape.")

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise RuntimeError("Gemma response does not include a valid 'questions' list.")

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

        # Clamp model-provided source references back onto the vetted source set
        # so every exported question still points at real local evidence.
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
            f"Gemma returned {len(normalized_questions)} valid questions; expected {expected_count}."
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
    model: str | None = None,
    max_chars_per_file: int = 6000,
    max_total_chars: int = 50_000,
    difficulty_profile: str = "exam_mixed",
    max_quality_attempts: int = 3,
    misconception_cache_ttl_hours: int = 168,
    misconception_search_model: str | None = None,
    misconception_embedding_model: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    global LAST_GENERATION_REPORT
    if question_count < 1 or question_count > 30:
        raise ValueError("question_count must be between 1 and 30.")
    if max_quality_attempts < 1:
        raise ValueError("max_quality_attempts must be >= 1.")

    load_dotenv()
    model = _resolve_env_setting(model, "MODEL", fallback=DEFAULT_GEMINI_MODEL)
    misconception_search_model = _resolve_env_setting(
        misconception_search_model,
        "GEMINI_SEARCH_MODEL",
        "MODEL",
        fallback=model,
    )
    misconception_embedding_model = _resolve_env_setting(
        misconception_embedding_model,
        "GEMINI_EMBEDDING_MODEL",
        fallback=DEFAULT_GEMINI_EMBEDDING_MODEL,
    )

    LAST_GENERATION_REPORT = {
        "attempts_used": 0,
        "max_quality_attempts": max_quality_attempts,
        "generation_model": model,
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

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env file.")

    def _status(message: str) -> None:
        if verbose:
            print(f"[MCQ] {message}")

    _status(f"Using generation model: Gemma:27B.")
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

    # Figure selection is a separate pass so diagrams/tables can be filtered for
    # prompt value before they consume limited multimodal context.
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
            _status("Calling Gemma to generate questions...")
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
            "generation_model": model,
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
        "generation_model": model,
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
    return _render_mcqs_markdown(
        questions,
        course_name=course_name,
        module_name=module_name,
        generated_at=generated_at,
        report=LAST_GENERATION_REPORT,
    )


def save_mcqs_markdown(
    questions: list[dict[str, Any]],
    *,
    tasked_items_summary_path: str | Path,
    course_name: str,
    module_name: str,
) -> Path:
    return _save_mcqs_markdown(
        questions,
        tasked_items_summary_path=tasked_items_summary_path,
        course_name=course_name,
        module_name=module_name,
        report=LAST_GENERATION_REPORT,
    )
