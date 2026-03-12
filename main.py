import argparse
import contextlib
import json
from pathlib import Path
import sys
from typing import Any

import canvas
import gemma
import local_sources

TASKED_SUMMARY_FILENAMES = {"tasked_items.json", "custom_items.json"}
SUBMITTED_SUMMARY_FILENAME = "submitted_assignments.json"
DEFAULT_AUTOMATION_QUESTION_COUNT = 20


def prompt_source_mode() -> str:
    while True:
        print("Source mode:")
        print("1. Canvas")
        print("2. Local study files")
        raw_value = input("Pick source mode (1-2): ").strip().lower()

        if raw_value in {"1", "canvas"}:
            return "canvas"
        if raw_value in {"2", "local", "local study files"}:
            return "local"

        print("Invalid choice. Please enter 1 or 2.")


def prompt_study_set_name() -> str:
    while True:
        raw_value = input("\nEnter a short study-set name: ").strip()
        if raw_value:
            return raw_value
        print("Study-set name cannot be empty.")


def prompt_local_source_cache(*, cache_dir: str = "Cache") -> local_sources.LocalSourceCacheResult:
    study_set_name = prompt_study_set_name()
    supported_types = ", ".join(sorted(local_sources.SUPPORTED_LOCAL_SOURCE_EXTENSIONS))

    while True:
        raw_path = input(
            f"Enter a directory containing {supported_types} files: "
        ).strip()
        if not raw_path:
            print("Directory path cannot be empty.")
            continue

        try:
            return local_sources.build_local_source_cache(
                raw_path,
                study_set_name,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Invalid local source directory: {exc}")


def prompt_question_count() -> int:
    while True:
        raw_value = input("\nHow many multiple choice questions do you want (1-30)? ").strip()
        if not raw_value.isdigit():
            print("Invalid number. Please enter a value between 1 and 30.")
            continue

        question_count = int(raw_value)
        if 1 <= question_count <= 30:
            return question_count

        print("Invalid number. Please enter a value between 1 and 30.")


def _normalize_question_count(question_count: int) -> int:
    if 1 <= question_count <= 30:
        return question_count
    raise ValueError("question_count must be between 1 and 30.")


def _question_source_label(question: dict[str, Any]) -> str:
    source_file = str(question.get("source_file") or "Unknown source").strip() or "Unknown source"
    source_page = question.get("source_page")
    source_page_label = source_page if isinstance(source_page, int) and source_page >= 1 else 1
    source_figure_id = str(question.get("source_figure_id") or "").strip()
    return (
        f"[Source: {source_file} p.{source_page_label} {source_figure_id}]"
        if source_figure_id
        else f"[Source: {source_file} p.{source_page_label}]"
    )


def _question_objective_line(question: dict[str, Any]) -> str:
    objective_label = str(question.get("objective_label") or "").strip()
    objective_topic = str(question.get("objective_topic") or "").strip()
    objective_skill = str(question.get("objective_skill") or "").strip()
    objective_difficulty = str(question.get("objective_difficulty") or "").strip()
    blueprint_slot_id = str(question.get("blueprint_slot_id") or "").strip()
    if not any((objective_label, objective_topic, objective_skill, objective_difficulty)):
        return ""

    return (
        f"Objective: {objective_label or 'Unknown objective'}"
        + (f" [{blueprint_slot_id}]" if blueprint_slot_id else "")
        + (f" | Topic: {objective_topic}" if objective_topic else "")
        + (f" | Skill: {objective_skill}" if objective_skill else "")
        + (f" | Difficulty: {objective_difficulty}" if objective_difficulty else "")
    )


def _question_actual_difficulty_line(question: dict[str, Any]) -> str:
    actual_difficulty_label = str(question.get("difficulty_actual_label") or "").strip()
    actual_target_correct_rate = question.get("difficulty_actual_target_correct_rate")
    if not actual_difficulty_label:
        return ""

    actual_line = f"Actual difficulty: {actual_difficulty_label}"
    if isinstance(actual_target_correct_rate, (int, float)) and float(actual_target_correct_rate) > 0:
        actual_line += f" (~{int(round(float(actual_target_correct_rate) * 100))}% correct)"
    return actual_line


def print_question_preview(questions: list[dict], *, max_questions: int = 2) -> None:
    if not questions:
        return

    print(f"\nPreview ({min(len(questions), max_questions)} question(s)):")
    for index, question in enumerate(questions[:max_questions], start=1):
        question_text = str(question.get("question") or "").strip()
        choices = question.get("choices")
        if not question_text or not isinstance(choices, list) or len(choices) < 4:
            continue

        print(f"{index}. {question_text} {_question_source_label(question)}")
        objective_line = _question_objective_line(question)
        if objective_line:
            print(objective_line)
        actual_line = _question_actual_difficulty_line(question)
        if actual_line:
            print(actual_line)
        for label, choice in zip(("A", "B", "C", "D"), choices[:4]):
            print(f"{label}) {str(choice).strip()}")
        print()


def _build_generation_summary(
    report: dict[str, Any],
    *,
    question_count: int,
    source_mode_label: str,
    markdown_path: str,
) -> list[str]:
    # Keep report assembly in one place so generation diagnostics stay readable
    # and new fields can be added without expanding the main control flow.
    lines = [
        f"\nGenerated {question_count} multiple-choice questions.",
        f"Source mode: {source_mode_label}",
    ]

    using_submitted = report.get("using_submitted_sources")
    if isinstance(using_submitted, bool):
        lines.append(f"Included submitted assignment sources: {'yes' if using_submitted else 'no'}")

    figure_candidates_count = report.get("figure_candidates_count")
    figure_selected_count = report.get("figure_selected_count")
    figure_deleted_count = report.get("figure_deleted_count")
    if isinstance(figure_candidates_count, int) and isinstance(figure_selected_count, int):
        summary = f"Figures kept after importance filter: {figure_selected_count} / {figure_candidates_count}"
        if isinstance(figure_deleted_count, int):
            summary += f" (deleted: {figure_deleted_count})"
        lines.append(summary)

    importance_model = str(report.get("importance_model") or "").strip()
    if importance_model:
        lines.append(f"Importance model: {importance_model}")

    generation_model = str(report.get("generation_model") or "").strip()
    if generation_model:
        lines.append(f"Generation model: {generation_model}")

    misconception_selected_count = report.get("misconception_selected_count")
    misconception_cache_hit = report.get("misconception_cache_hit")
    if isinstance(misconception_selected_count, int):
        cache_label = "cache hit" if isinstance(misconception_cache_hit, bool) and misconception_cache_hit else "fresh mine"
        lines.append(f"Misconception bank records used: {misconception_selected_count} ({cache_label})")

    misconception_sources_used = report.get("misconception_sources_used")
    if isinstance(misconception_sources_used, int):
        lines.append(f"Misconception evidence sources used: {misconception_sources_used}")

    diagnostics_passed = report.get("diagnostics_passed")
    if isinstance(diagnostics_passed, bool):
        lines.append(f"Distractor diagnostics valid: {'yes' if diagnostics_passed else 'no'}")

    difficulty_control_passed = report.get("difficulty_control_passed")
    if isinstance(difficulty_control_passed, bool):
        lines.append(f"Difficulty control valid: {'yes' if difficulty_control_passed else 'no'}")

    difficulty_calibration_source = str(report.get("difficulty_calibration_source") or "").strip()
    if difficulty_calibration_source:
        lines.append(f"Difficulty calibration source: {difficulty_calibration_source}")

    difficulty_passed_count = report.get("difficulty_passed_count")
    difficulty_required_pass_count = report.get("difficulty_required_pass_count")
    if (
        isinstance(difficulty_passed_count, int)
        and isinstance(difficulty_required_pass_count, int)
        and difficulty_required_pass_count >= 1
        and difficulty_control_passed is False
    ):
        lines.append(
            "Difficulty control summary: "
            f"{difficulty_passed_count}/{question_count} questions passed; "
            f"needed {difficulty_required_pass_count}"
        )

    difficulty_failed_indices = report.get("difficulty_failed_indices")
    if isinstance(difficulty_failed_indices, list) and difficulty_failed_indices:
        lines.append("Difficulty mismatches: " + ", ".join(str(index) for index in difficulty_failed_indices))

    difficulty_control_feedback = str(report.get("difficulty_control_feedback") or "").strip()
    if difficulty_control_feedback:
        lines.append(f"Difficulty review: {difficulty_control_feedback}")

    blueprint_passed = report.get("blueprint_passed")
    if isinstance(blueprint_passed, bool):
        lines.append(f"Blueprint alignment valid: {'yes' if blueprint_passed else 'no'}")

    blueprint_slots_count = report.get("blueprint_slots_count")
    blueprint_topics_count = report.get("blueprint_topics_count")
    blueprint_skills_count = report.get("blueprint_skills_count")
    blueprint_difficulties_count = report.get("blueprint_difficulties_count")
    if (
        isinstance(blueprint_slots_count, int)
        and isinstance(blueprint_topics_count, int)
        and isinstance(blueprint_skills_count, int)
        and isinstance(blueprint_difficulties_count, int)
        and blueprint_slots_count > 0
    ):
        lines.append(
            "Blueprint coverage: "
            f"{blueprint_slots_count} slots, {blueprint_topics_count} topics, "
            f"{blueprint_skills_count} skills, {blueprint_difficulties_count} difficulty levels"
        )

    attempts_used = report.get("attempts_used")
    if isinstance(attempts_used, int) and attempts_used >= 1:
        lines.append(f"Quality attempts used: {attempts_used}")

    warning = str(report.get("warning") or "").strip()
    if warning:
        lines.append(f"Note: {warning}")

    lines.append(f"Saved markdown file: {markdown_path}")
    return lines


def _load_summary_metadata(summary_path: str | Path) -> tuple[str, str]:
    payload = json.loads(Path(summary_path).expanduser().resolve().read_text(encoding="utf-8"))
    course = payload.get("course") if isinstance(payload, dict) else {}
    module = payload.get("module") if isinstance(payload, dict) else {}
    course_name = (
        str(course.get("name") or "").strip()
        if isinstance(course, dict)
        else ""
    ) or "Unknown course"
    module_name = (
        str(module.get("name") or "").strip()
        if isinstance(module, dict)
        else ""
    ) or "Unknown module"
    return course_name, module_name


def _find_submitted_summary_path(tasked_summary_path: str | Path) -> str | None:
    summary_path = Path(tasked_summary_path).expanduser().resolve()
    module_root = summary_path.parent.parent if summary_path.parent.name == "tasked" else summary_path.parent
    submitted_summary_path = module_root / "submitted" / SUBMITTED_SUMMARY_FILENAME
    if submitted_summary_path.is_file():
        return str(submitted_summary_path)
    return None


def _discover_tasked_summary_paths(*, cache_dir: str | Path = "Cache") -> list[Path]:
    cache_root = Path(cache_dir).expanduser().resolve()
    if not cache_root.exists():
        return []

    paths = [
        path
        for path in cache_root.rglob("*.json")
        if path.is_file() and path.name in TASKED_SUMMARY_FILENAMES
    ]
    paths.sort(key=lambda path: (-path.stat().st_mtime, str(path).lower()))
    return paths


def _relative_cache_path(path: str | Path, *, cache_dir: str | Path = "Cache") -> str | None:
    cache_root = Path(cache_dir).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        return resolved_path.relative_to(cache_root).as_posix()
    except ValueError:
        return None


def generate_quiz_from_summary(
    *,
    tasked_summary_path: str,
    question_count: int,
    submitted_summary_path: str | None = None,
    course_name: str | None = None,
    module_name: str | None = None,
    source_mode_label: str = "Cache",
    verbose: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    resolved_question_count = _normalize_question_count(question_count)
    resolved_tasked_summary_path = str(Path(tasked_summary_path).expanduser().resolve())
    resolved_submitted_summary_path = submitted_summary_path or _find_submitted_summary_path(resolved_tasked_summary_path)
    resolved_course_name, resolved_module_name = _load_summary_metadata(resolved_tasked_summary_path)
    resolved_course_name = str(course_name or resolved_course_name).strip() or resolved_course_name
    resolved_module_name = str(module_name or resolved_module_name).strip() or resolved_module_name

    questions = gemma.generate_mcqs_from_tasked_module(
        resolved_tasked_summary_path,
        question_count=resolved_question_count,
        submitted_items_summary_path=resolved_submitted_summary_path,
        difficulty_profile="exam_mixed",
        verbose=verbose,
    )
    markdown_path = gemma.save_mcqs_markdown(
        questions,
        tasked_items_summary_path=resolved_tasked_summary_path,
        course_name=resolved_course_name,
        module_name=resolved_module_name,
    )
    report = gemma.LAST_GENERATION_REPORT if isinstance(gemma.LAST_GENERATION_REPORT, dict) else {}
    if print_summary:
        for line in _build_generation_summary(
            report,
            question_count=len(questions),
            source_mode_label=source_mode_label,
            markdown_path=markdown_path,
        ):
            print(line)
        print_question_preview(questions, max_questions=2)

    return {
        "success": True,
        "course_name": resolved_course_name,
        "module_name": resolved_module_name,
        "question_count": len(questions),
        "source_mode_label": source_mode_label,
        "tasked_summary_path": resolved_tasked_summary_path,
        "submitted_summary_path": resolved_submitted_summary_path,
        "markdown_path": str(markdown_path),
        "relative_markdown_path": _relative_cache_path(markdown_path),
        "report": report,
    }


def run_generation_flow(
    *,
    tasked_summary_path: str,
    course_name: str,
    module_name: str,
    source_mode_label: str,
    submitted_summary_path: str | None = None,
) -> None:
    question_count = prompt_question_count()
    try:
        generate_quiz_from_summary(
            tasked_summary_path=tasked_summary_path,
            question_count=question_count,
            submitted_summary_path=submitted_summary_path,
            course_name=course_name,
            module_name=module_name,
            source_mode_label=source_mode_label,
            verbose=True,
            print_summary=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nMCQ generation error: {exc}")


def run_latest_cached_generation(
    *,
    question_count: int = DEFAULT_AUTOMATION_QUESTION_COUNT,
    cache_dir: str | Path = "Cache",
    verbose: bool = False,
    print_summary: bool = False,
) -> dict[str, Any]:
    latest_summary_paths = _discover_tasked_summary_paths(cache_dir=cache_dir)
    if not latest_summary_paths:
        raise RuntimeError(f"No tasked module summary was found under {Path(cache_dir).expanduser().resolve()}.")

    latest_summary_path = latest_summary_paths[0]
    return generate_quiz_from_summary(
        tasked_summary_path=str(latest_summary_path),
        question_count=question_count,
        submitted_summary_path=_find_submitted_summary_path(latest_summary_path),
        source_mode_label="Latest cached module",
        verbose=verbose,
        print_summary=print_summary,
    )


def run_canvas_generation(
    *,
    course_selection: int | str,
    module_selection: int | str,
    question_count: int = DEFAULT_AUTOMATION_QUESTION_COUNT,
    cache_dir: str | Path = "Cache",
    verbose: bool = False,
    print_summary: bool = False,
) -> dict[str, Any]:
    result = canvas.run_canvas(
        course_selection=course_selection,
        prompt_for_selection=False,
        module_selection=module_selection,
        prompt_for_module_selection=False,
        save_module_assignments=False,
        save_submitted_assignments=False,
        save_submitted_module_files=True,
        save_tasked_module_files=True,
        cache_dir=cache_dir,
        print_output=print_summary,
    )
    if result is None:
        raise RuntimeError("No Canvas course/module data was returned.")
    if not result.saved_tasked_module_path:
        raise RuntimeError("No tasked module cache was generated for the selected Canvas module.")

    module_name = result.module_name or result.module_label or "Unknown module"
    return generate_quiz_from_summary(
        tasked_summary_path=result.saved_tasked_module_path,
        question_count=question_count,
        submitted_summary_path=result.saved_submitted_module_path,
        course_name=result.course.name,
        module_name=module_name,
        source_mode_label="Canvas",
        verbose=verbose,
        print_summary=print_summary,
    )


def run_local_generation(
    *,
    source_dir: str | Path,
    study_set_name: str,
    question_count: int = DEFAULT_AUTOMATION_QUESTION_COUNT,
    cache_dir: str | Path = "Cache",
    verbose: bool = False,
    print_summary: bool = False,
) -> dict[str, Any]:
    local_result = local_sources.build_local_source_cache(
        source_dir,
        study_set_name,
        cache_dir=cache_dir,
    )
    return generate_quiz_from_summary(
        tasked_summary_path=local_result.summary_path,
        question_count=question_count,
        course_name="Custom Upload",
        module_name=local_result.study_set_name,
        source_mode_label="Local study files",
        verbose=verbose,
        print_summary=print_summary,
    )


def _run_canvas_mode() -> None:
    result = canvas.run_canvas(
        prompt_for_selection=True,
        prompt_for_module_selection=True,
        save_module_assignments=False,
        save_submitted_assignments=False,
        save_submitted_module_files=True,
        save_tasked_module_files=True,
        cache_dir="Cache",
        print_output=True,
    )
    if result is None:
        return

    if not result.saved_tasked_module_path:
        print("\nNo tasked module cache was generated for this module.")
        return

    module_name = result.module_name or result.module_label or "Unknown module"
    run_generation_flow(
        tasked_summary_path=result.saved_tasked_module_path,
        submitted_summary_path=result.saved_submitted_module_path,
        course_name=result.course.name,
        module_name=module_name,
        source_mode_label="Canvas",
    )


def _run_local_mode() -> None:
    local_result = prompt_local_source_cache(cache_dir="Cache")
    print(f"\nSelected source mode: Local study files")
    print(f"Study set: {local_result.study_set_name}")
    print(f"Source directory: {local_result.source_dir}")
    print(f"Staged files: {local_result.staged_file_count}")
    print(f"Custom module cache file: {local_result.summary_path}")
    run_generation_flow(
        tasked_summary_path=local_result.summary_path,
        course_name="Custom Upload",
        module_name=local_result.study_set_name,
        source_mode_label="Local study files",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple-choice quizzes from Canvas or cached study sources.")
    parser.add_argument(
        "--run-latest-cache",
        action="store_true",
        help="Generate a new quiz from the most recently updated staged cache summary.",
    )
    parser.add_argument("--tasked-summary", help="Generate a new quiz from a specific tasked summary JSON file.")
    parser.add_argument(
        "--submitted-summary",
        help="Optional submitted summary JSON file to merge with the tasked summary during generation.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=DEFAULT_AUTOMATION_QUESTION_COUNT,
        help=f"Number of questions to generate for non-interactive runs (default: {DEFAULT_AUTOMATION_QUESTION_COUNT}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object describing the non-interactive generation result.",
    )
    parser.add_argument(
        "--progress-to-stderr",
        action="store_true",
        help="Keep verbose progress logging enabled for JSON mode by redirecting generation logs to stderr.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("canvas", "local"),
        help="Run a non-interactive generation flow for a specific source mode.",
    )
    parser.add_argument("--course-selection", help="Canvas course number, ID, or name for --source-mode canvas.")
    parser.add_argument("--module-selection", help="Canvas module number, ID, or name for --source-mode canvas.")
    parser.add_argument("--source-dir", help="Local source directory for --source-mode local.")
    parser.add_argument("--study-set-name", help="Study-set name for --source-mode local.")
    args = parser.parse_args()

    selected_modes = [bool(args.run_latest_cache), bool(args.tasked_summary), bool(args.source_mode)]
    if sum(selected_modes) > 1:
        parser.error("Use only one non-interactive mode: --run-latest-cache, --tasked-summary, or --source-mode.")

    if args.run_latest_cache or args.tasked_summary or args.source_mode:
        try:
            enable_progress_logging = not args.json or args.progress_to_stderr
            progress_context = (
                contextlib.redirect_stdout(sys.stderr)
                if args.json and args.progress_to_stderr
                else contextlib.nullcontext()
            )

            with progress_context:
                if args.run_latest_cache:
                    result = run_latest_cached_generation(
                        question_count=args.question_count,
                        verbose=enable_progress_logging,
                        print_summary=enable_progress_logging,
                    )
                elif args.source_mode == "canvas":
                    if not args.course_selection or not args.module_selection:
                        parser.error("--course-selection and --module-selection are required for --source-mode canvas.")
                    result = run_canvas_generation(
                        course_selection=args.course_selection,
                        module_selection=args.module_selection,
                        question_count=args.question_count,
                        verbose=enable_progress_logging,
                        print_summary=enable_progress_logging,
                    )
                elif args.source_mode == "local":
                    if not args.source_dir or not args.study_set_name:
                        parser.error("--source-dir and --study-set-name are required for --source-mode local.")
                    result = run_local_generation(
                        source_dir=args.source_dir,
                        study_set_name=args.study_set_name,
                        question_count=args.question_count,
                        verbose=enable_progress_logging,
                        print_summary=enable_progress_logging,
                    )
                else:
                    result = generate_quiz_from_summary(
                        tasked_summary_path=args.tasked_summary,
                        question_count=args.question_count,
                        submitted_summary_path=args.submitted_summary,
                        source_mode_label="Direct summary",
                        verbose=enable_progress_logging,
                        print_summary=enable_progress_logging,
                    )
        except Exception as exc:  # noqa: BLE001
            if args.json:
                print(json.dumps({"success": False, "error": str(exc)}))
                raise SystemExit(1) from exc
            raise

        if args.json:
            print(json.dumps(result))
        return

    if prompt_source_mode() == "canvas":
        _run_canvas_mode()
        return
    _run_local_mode()


if __name__ == "__main__":
    main()
