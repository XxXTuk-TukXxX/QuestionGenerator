import canvas
import gemini
import local_sources


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


def print_question_preview(questions: list[dict], *, max_questions: int = 2) -> None:
    if not questions:
        return

    print(f"\nPreview ({min(len(questions), max_questions)} question(s)):")
    for index, question in enumerate(questions[:max_questions], start=1):
        question_text = str(question.get("question") or "").strip()
        choices = question.get("choices")
        if not question_text or not isinstance(choices, list) or len(choices) < 4:
            continue

        source_file = str(question.get("source_file") or "Unknown source").strip() or "Unknown source"
        source_page = question.get("source_page")
        source_page_label = source_page if isinstance(source_page, int) and source_page >= 1 else 1
        source_figure_id = str(question.get("source_figure_id") or "").strip()
        if source_figure_id:
            source_label = f"[Source: {source_file} p.{source_page_label} {source_figure_id}]"
        else:
            source_label = f"[Source: {source_file} p.{source_page_label}]"

        print(f"{index}. {question_text} {source_label}")
        objective_label = str(question.get("objective_label") or "").strip()
        objective_topic = str(question.get("objective_topic") or "").strip()
        objective_skill = str(question.get("objective_skill") or "").strip()
        objective_difficulty = str(question.get("objective_difficulty") or "").strip()
        blueprint_slot_id = str(question.get("blueprint_slot_id") or "").strip()
        if objective_label or objective_topic or objective_skill or objective_difficulty:
            objective_line = (
                f"Objective: {objective_label or 'Unknown objective'}"
                + (f" [{blueprint_slot_id}]" if blueprint_slot_id else "")
                + (f" | Topic: {objective_topic}" if objective_topic else "")
                + (f" | Skill: {objective_skill}" if objective_skill else "")
                + (f" | Difficulty: {objective_difficulty}" if objective_difficulty else "")
            )
            print(objective_line)
        actual_difficulty_label = str(question.get("difficulty_actual_label") or "").strip()
        actual_target_correct_rate = question.get("difficulty_actual_target_correct_rate")
        if actual_difficulty_label:
            actual_line = f"Actual difficulty: {actual_difficulty_label}"
            if isinstance(actual_target_correct_rate, (int, float)) and float(actual_target_correct_rate) > 0:
                actual_line += f" (~{int(round(float(actual_target_correct_rate) * 100))}% correct)"
            print(actual_line)
        for label, choice in zip(("A", "B", "C", "D"), choices[:4]):
            print(f"{label}) {str(choice).strip()}")
        print()


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
        questions = gemini.generate_mcqs_from_tasked_module(
            tasked_summary_path,
            question_count=question_count,
            submitted_items_summary_path=submitted_summary_path,
            difficulty_profile="exam_mixed",
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nMCQ generation error: {exc}")
        return

    try:
        markdown_path = gemini.save_mcqs_markdown(
            questions,
            tasked_items_summary_path=tasked_summary_path,
            course_name=course_name,
            module_name=module_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nFailed to save markdown output: {exc}")
        return

    report = gemini.LAST_GENERATION_REPORT if isinstance(gemini.LAST_GENERATION_REPORT, dict) else {}
    attempts_used = report.get("attempts_used")
    warning = report.get("warning")
    using_submitted = report.get("using_submitted_sources")
    figure_candidates_count = report.get("figure_candidates_count")
    figure_selected_count = report.get("figure_selected_count")
    figure_deleted_count = report.get("figure_deleted_count")
    importance_model = report.get("importance_model")
    misconception_cache_hit = report.get("misconception_cache_hit")
    misconception_selected_count = report.get("misconception_selected_count")
    misconception_sources_used = report.get("misconception_sources_used")
    diagnostics_passed = report.get("diagnostics_passed")
    blueprint_passed = report.get("blueprint_passed")
    difficulty_control_passed = report.get("difficulty_control_passed")
    difficulty_control_feedback = report.get("difficulty_control_feedback")
    difficulty_failed_indices = report.get("difficulty_failed_indices")
    difficulty_passed_count = report.get("difficulty_passed_count")
    difficulty_required_pass_count = report.get("difficulty_required_pass_count")
    difficulty_calibration_source = report.get("difficulty_calibration_source")
    blueprint_slots_count = report.get("blueprint_slots_count")
    blueprint_topics_count = report.get("blueprint_topics_count")
    blueprint_skills_count = report.get("blueprint_skills_count")
    blueprint_difficulties_count = report.get("blueprint_difficulties_count")

    print(f"\nGenerated {len(questions)} multiple-choice questions.")
    print(f"Source mode: {source_mode_label}")
    if isinstance(using_submitted, bool):
        print(f"Included submitted assignment sources: {'yes' if using_submitted else 'no'}")
    if isinstance(figure_candidates_count, int) and isinstance(figure_selected_count, int):
        summary = f"Figures kept after importance filter: {figure_selected_count} / {figure_candidates_count}"
        if isinstance(figure_deleted_count, int):
            summary += f" (deleted: {figure_deleted_count})"
        print(summary)
    if isinstance(importance_model, str) and importance_model.strip():
        print(f"Importance model: {importance_model}")
    if isinstance(misconception_selected_count, int):
        cache_label = (
            "cache hit"
            if isinstance(misconception_cache_hit, bool) and misconception_cache_hit
            else "fresh mine"
        )
        print(
            "Misconception bank records used: "
            f"{misconception_selected_count} ({cache_label})"
        )
    if isinstance(misconception_sources_used, int):
        print(f"Misconception evidence sources used: {misconception_sources_used}")
    if isinstance(diagnostics_passed, bool):
        print(f"Distractor diagnostics valid: {'yes' if diagnostics_passed else 'no'}")
    if isinstance(difficulty_control_passed, bool):
        print(f"Difficulty control valid: {'yes' if difficulty_control_passed else 'no'}")
    if isinstance(difficulty_calibration_source, str) and difficulty_calibration_source.strip():
        print(f"Difficulty calibration source: {difficulty_calibration_source}")
    if (
        isinstance(difficulty_passed_count, int)
        and isinstance(difficulty_required_pass_count, int)
        and difficulty_required_pass_count >= 1
        and isinstance(difficulty_control_passed, bool)
        and not difficulty_control_passed
    ):
        print(
            "Difficulty control summary: "
            f"{difficulty_passed_count}/{len(questions)} questions passed; "
            f"needed {difficulty_required_pass_count}"
        )
    if isinstance(difficulty_failed_indices, list) and difficulty_failed_indices:
        print(
            "Difficulty mismatches: "
            + ", ".join(str(index) for index in difficulty_failed_indices)
        )
    if isinstance(difficulty_control_feedback, str) and difficulty_control_feedback.strip():
        print(f"Difficulty review: {difficulty_control_feedback}")
    if isinstance(blueprint_passed, bool):
        print(f"Blueprint alignment valid: {'yes' if blueprint_passed else 'no'}")
    if (
        isinstance(blueprint_slots_count, int)
        and isinstance(blueprint_topics_count, int)
        and isinstance(blueprint_skills_count, int)
        and isinstance(blueprint_difficulties_count, int)
        and blueprint_slots_count > 0
    ):
        print(
            "Blueprint coverage: "
            f"{blueprint_slots_count} slots, {blueprint_topics_count} topics, "
            f"{blueprint_skills_count} skills, {blueprint_difficulties_count} difficulty levels"
        )
    if isinstance(attempts_used, int) and attempts_used >= 1:
        print(f"Quality attempts used: {attempts_used}")
    if isinstance(warning, str) and warning.strip():
        print(f"Note: {warning}")
    print(f"Saved markdown file: {markdown_path}")

    print_question_preview(questions, max_questions=2)


def main() -> None:
    source_mode = prompt_source_mode()

    if source_mode == "canvas":
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
        return

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


if __name__ == "__main__":
    main()
