from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .gemma_common import normalize_positive_int
from .gemma_sources import _mcq_output_dir_from_summary_path


def render_mcqs_markdown(
    questions: list[dict[str, Any]],
    *,
    course_name: str,
    module_name: str,
    generated_at: str,
    report: dict[str, Any] | None = None,
) -> str:
    report = report if isinstance(report, dict) else {}
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
        source_page = normalize_positive_int(question.get("source_page")) or 1
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
                + (f" [{blueprint_slot_id}]" if blueprint_slot_id else "")
                + (f" | Topic: {objective_topic}" if objective_topic else "")
                + (f" | Skill: {objective_skill}" if objective_skill else "")
                + (f" | Difficulty: {objective_difficulty}" if objective_difficulty else "")
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
    report: dict[str, Any] | None = None,
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
                current["source_figure_image_path"] = str(image_path.relative_to(output_dir))
            except ValueError:
                current["source_figure_image_path"] = str(image_path)
        normalized_questions.append(current)

    markdown_text = render_mcqs_markdown(
        normalized_questions,
        course_name=course_name,
        module_name=module_name,
        generated_at=generated_label,
        report=report,
    )

    timestamped_path = output_dir / f"mcq_{timestamp_label}.md"
    latest_path = output_dir / "mcq_latest.md"
    timestamped_path.write_text(markdown_text, encoding="utf-8")
    latest_path.write_text(markdown_text, encoding="utf-8")
    return timestamped_path
