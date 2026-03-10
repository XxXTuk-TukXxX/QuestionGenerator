from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import gemini
from study_engine import (
    StudentProfile,
    build_concise_why,
    build_local_mini_retest,
    build_local_repair_plan,
    concept_mastery,
    extract_source_evidence,
    load_student_profile,
    question_identity,
    question_metadata,
    record_intervention_result,
    resolve_source_file,
    save_student_profile,
    schedule_questions,
    source_reference,
    update_profile_from_attempt,
)

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - Pillow is optional at runtime.
    Image = None
    ImageTk = None


QUESTION_HEADER_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
CHOICE_RE = re.compile(r"^\s*([A-D])\)\s+(.*)$")
ANSWER_KEY_PAIR_RE = re.compile(r"(\d+)\s*:\s*([A-D])")
SOURCE_SUFFIX_RE = re.compile(r"^(.*)\s+\[Source:\s*(.+?)\]\s*$")
FIGURE_LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
DIAGNOSTIC_RE = re.compile(r"^\s*-\s*([A-D]):\s*(.*?)(?:\s+\(([^()]+)\))?\s*$")
SUPPORTED_MD_GLOB = "mcq*.md"
REMEDIATION_TIER_ORDER = {"easier": 0, "practice": 1, "transfer": 2, "retest": 3}
DIFFICULTY_ORDER = {"foundation": 0, "standard": 1, "challenge": 2}


@dataclass
class DistractorDiagnostic:
    option: str
    summary: str = ""
    misconception_id: str = ""
    why_chosen: str = ""
    why_wrong: str = ""


@dataclass
class QuizQuestion:
    number: int
    text: str
    source: str = ""
    source_file: str = ""
    source_page: int = 1
    source_figure_id: str = ""
    question_id: str = ""
    question_kind: str = "main"
    remediation_tier: str = ""
    explanation: str = ""
    generation_origin: str = "markdown"
    objective: str = ""
    objective_label: str = ""
    blueprint_slot_id: str = ""
    topic: str = ""
    skill: str = ""
    difficulty: str = ""
    difficulty_dial: str = ""
    actual_difficulty: str = ""
    markdown_block: str = ""
    figure_path: str = ""
    choices: dict[str, str] = field(default_factory=dict)
    correct_answer: str = ""
    distractor_diagnostics: dict[str, DistractorDiagnostic] = field(default_factory=dict)


@dataclass
class QuizDocument:
    path: Path
    course: str = ""
    module: str = ""
    generated: str = ""
    question_count: int = 0
    questions: list[QuizQuestion] = field(default_factory=list)


@dataclass
class QuestionCardState:
    question: QuizQuestion
    frame: ttk.Frame
    answer_var: tk.StringVar
    choice_buttons: list[ttk.Radiobutton]
    feedback_label: tk.Label
    check_button: ttk.Button | None
    remediation_frame: ttk.Frame
    figure_image: object | None = None
    completed: bool = False
    busy: bool = False
    selected_option: str = ""
    intervention_started_at: str = ""
    repair_questions: list[QuizQuestion] = field(default_factory=list)
    repair_correct_count: int = 0
    repair_index: int = 0
    retest_questions: list[QuizQuestion] = field(default_factory=list)
    retest_correct_count: int = 0
    retest_index: int = 0
    intervention_complete: bool = False


def discover_markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.rglob(SUPPORTED_MD_GLOB) if path.is_file()]
    return sorted(
        files,
        key=lambda path: (
            path.name != "mcq_latest.md",
            -path.stat().st_mtime,
            str(path),
        ),
    )


def _split_source_suffix(line: str) -> tuple[str, str]:
    match = SOURCE_SUFFIX_RE.match(line.strip())
    if not match:
        return line.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def _parse_source_metadata(source: str) -> dict[str, str | int]:
    reference = source_reference(type("SourceRef", (), {"source": source, "source_file": "", "source_page": 0, "source_figure_id": ""})())
    return reference


def _stable_question_id(question_text: str, correct_answer: str, source_file: str, source_page: int) -> str:
    seed = "||".join([
        question_text.strip(),
        correct_answer.strip().upper(),
        source_file.strip(),
        str(max(1, int(source_page or 1))),
    ])
    return f"q_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]}"


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


def _parse_question_block(lines: list[str]) -> QuizQuestion | None:
    if not lines:
        return None

    header_match = QUESTION_HEADER_RE.match(lines[0].rstrip())
    if not header_match:
        return None

    number = int(header_match.group(1))
    question_text, source = _split_source_suffix(header_match.group(2).strip())
    source_meta = _parse_source_metadata(source)
    question = QuizQuestion(
        number=number,
        text=question_text,
        source=source,
        source_file=str(source_meta["source_file"]),
        source_page=int(source_meta["source_page"]),
        source_figure_id=str(source_meta["source_figure_id"]),
        markdown_block="\n".join(line.rstrip() for line in lines).strip(),
    )
    in_diagnostics = False
    current_diagnostic: DistractorDiagnostic | None = None
    current_diagnostic_field = ""

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            current_diagnostic_field = ""
            continue

        if in_diagnostics:
            diagnostic_match = DIAGNOSTIC_RE.match(raw_line)
            if diagnostic_match:
                option = diagnostic_match.group(1)
                current_diagnostic = DistractorDiagnostic(
                    option=option,
                    summary=diagnostic_match.group(2).strip(),
                    misconception_id=(diagnostic_match.group(3) or "").strip(),
                )
                question.distractor_diagnostics[option] = current_diagnostic
                current_diagnostic_field = "summary"
                continue

            if current_diagnostic is None:
                continue

            if stripped.startswith("Why chosen:"):
                current_diagnostic.why_chosen = stripped.removeprefix("Why chosen:").strip()
                current_diagnostic_field = "why_chosen"
                continue
            if stripped.startswith("Why wrong:"):
                current_diagnostic.why_wrong = stripped.removeprefix("Why wrong:").strip()
                current_diagnostic_field = "why_wrong"
                continue

            if current_diagnostic_field == "summary":
                current_diagnostic.summary = f"{current_diagnostic.summary} {stripped}".strip()
            elif current_diagnostic_field == "why_chosen":
                current_diagnostic.why_chosen = f"{current_diagnostic.why_chosen} {stripped}".strip()
            elif current_diagnostic_field == "why_wrong":
                current_diagnostic.why_wrong = f"{current_diagnostic.why_wrong} {stripped}".strip()
            continue

        choice_match = CHOICE_RE.match(raw_line)
        if choice_match:
            question.choices[choice_match.group(1)] = choice_match.group(2).strip()
            continue

        if stripped.startswith("Objective:"):
            objective_metadata = _parse_objective_metadata(stripped.removeprefix("Objective:").strip())
            question.objective = objective_metadata["objective"]
            question.objective_label = objective_metadata["objective_label"]
            question.blueprint_slot_id = objective_metadata["blueprint_slot_id"]
            question.topic = objective_metadata["topic"]
            question.skill = objective_metadata["skill"]
            question.difficulty = objective_metadata["difficulty"]
            continue
        if stripped.startswith("Difficulty Dial:"):
            question.difficulty_dial = stripped.removeprefix("Difficulty Dial:").strip()
            continue
        if stripped.startswith("Actual Difficulty:"):
            question.actual_difficulty = stripped.removeprefix("Actual Difficulty:").strip()
            continue
        if stripped.startswith("Figure:"):
            match = FIGURE_LINK_RE.search(stripped)
            question.figure_path = match.group(2).strip() if match else stripped.removeprefix("Figure:").strip()
            continue
        if stripped == "Distractor Diagnostics:":
            in_diagnostics = True
            current_diagnostic = None
            current_diagnostic_field = ""
            continue

    return question


def load_quiz_document(path: Path) -> QuizDocument:
    resolved = path.expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    document = QuizDocument(path=resolved)

    answer_key: dict[int, str] = {}
    question_start: int | None = None
    answer_key_start: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- Course:"):
            document.course = stripped.removeprefix("- Course:").strip()
        elif stripped.startswith("- Module:"):
            document.module = stripped.removeprefix("- Module:").strip()
        elif stripped.startswith("- Generated:"):
            document.generated = stripped.removeprefix("- Generated:").strip()
        elif stripped.startswith("- Question Count:"):
            raw_count = stripped.removeprefix("- Question Count:").strip()
            if raw_count.isdigit():
                document.question_count = int(raw_count)
        elif stripped == "## Questions":
            question_start = index + 1
        elif stripped == "## Answer Key":
            answer_key_start = index + 1
            break

    if question_start is None:
        raise ValueError(f"Markdown file does not include a '## Questions' section: {resolved}")

    if answer_key_start is not None:
        for line in lines[answer_key_start:]:
            if "Answer Key:" not in line:
                continue
            for question_number, answer in ANSWER_KEY_PAIR_RE.findall(line):
                answer_key[int(question_number)] = answer

    block_lines: list[str] = []
    question_lines = lines[question_start:answer_key_start - 1 if answer_key_start is not None else len(lines)]
    for raw_line in question_lines:
        if QUESTION_HEADER_RE.match(raw_line):
            if block_lines:
                question = _parse_question_block(block_lines)
                if question is not None:
                    question.correct_answer = answer_key.get(question.number, "")
                    question.question_id = _stable_question_id(
                        question.text,
                        question.correct_answer,
                        question.source_file,
                        question.source_page,
                    )
                    document.questions.append(question)
                block_lines = []
        if block_lines or raw_line.strip():
            block_lines.append(raw_line)

    if block_lines:
        question = _parse_question_block(block_lines)
        if question is not None:
            question.correct_answer = answer_key.get(question.number, "")
            question.question_id = _stable_question_id(
                question.text,
                question.correct_answer,
                question.source_file,
                question.source_page,
            )
            document.questions.append(question)

    return document


def _generated_row_to_question(row: dict[str, object], *, number: int, question_kind: str, remediation_tier: str) -> QuizQuestion:
    choices_list = row.get("choices") if isinstance(row.get("choices"), list) else []
    choices = {
        label: str(choices_list[index] or "").strip()
        for index, label in enumerate(("A", "B", "C", "D"))
        if index < len(choices_list)
    }
    source_file = str(row.get("source_file") or "Unknown source").strip() or "Unknown source"
    source_page = int(row.get("source_page") or 1)
    source_figure_id = str(row.get("source_figure_id") or "").strip()
    source = f"{source_file} p.{source_page}"
    if source_figure_id:
        source = f"{source} {source_figure_id}"
    question = QuizQuestion(
        number=number,
        text=str(row.get("question") or "").strip(),
        source=source,
        source_file=source_file,
        source_page=max(1, source_page),
        source_figure_id=source_figure_id,
        question_kind=question_kind,
        remediation_tier=remediation_tier,
        explanation=str(row.get("why") or "").strip(),
        generation_origin="generated",
        objective_label=str(row.get("objective_label") or "").strip(),
        topic=str(row.get("objective_topic") or "").strip(),
        skill=str(row.get("objective_skill") or "").strip(),
        difficulty=str(row.get("objective_difficulty") or "").strip(),
        choices=choices,
        correct_answer=str(row.get("answer") or "").strip().upper(),
    )
    question.question_id = _stable_question_id(question.text, question.correct_answer, question.source_file, question.source_page)
    return question


def _clone_for_remediation(question: QuizQuestion, *, question_kind: str, remediation_tier: str, origin: str) -> QuizQuestion:
    return replace(
        question,
        question_kind=question_kind,
        remediation_tier=remediation_tier,
        generation_origin=origin,
    )


class QuizApp(tk.Tk):
    def __init__(self, cache_root: Path) -> None:
        super().__init__()
        self.title("MCQ Frontend")
        self.geometry("1400x900")
        self.minsize(1100, 760)

        self.cache_root = cache_root
        self.profile_root = Path(__file__).resolve().parent / "User" / "StudyProfiles"
        self.current_document: QuizDocument | None = None
        self.current_module_root: Path | None = None
        self.current_module_question_pool: list[QuizQuestion] = []
        self.current_profile: StudentProfile | None = None
        self.markdown_files: list[Path] = []
        self.current_session_questions: list[QuizQuestion] = []
        self.card_states: list[QuestionCardState] = []
        self.figure_images: list[object] = []
        self.session_used_question_ids: set[str] = set()
        self.session_used_question_texts: set[str] = set()
        self.test_submitted = False
        self.current_stage = "initial"
        self.pending_wrong_states: list[QuestionCardState] = []
        self.pending_repair_contexts: list[dict[str, object]] = []
        self.loading_dialog: tk.Toplevel | None = None
        self.loading_message_var = tk.StringVar(value="")

        self._build_ui()
        self.refresh_file_list()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self, padding=12)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)

        ttk.Label(sidebar, text="Available Tests", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        button_row = ttk.Frame(sidebar)
        button_row.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(button_row, text="Refresh", command=self.refresh_file_list).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_row, text="Open File", command=self.open_file_dialog).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        list_frame = ttk.Frame(sidebar)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(list_frame, exportselection=False)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        self.file_listbox.bind("<<ListboxSelect>>", lambda _event: self.load_selected_file())

        sidebar_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_listbox.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        self.file_listbox.configure(yscrollcommand=sidebar_scroll.set)

        self.file_path_label = tk.Label(
            sidebar,
            text="",
            justify="left",
            anchor="w",
            wraplength=300,
            fg="#555555",
        )
        self.file_path_label.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        session_frame = ttk.LabelFrame(sidebar, text="Study Session", padding=10)
        session_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        session_frame.columnconfigure(1, weight=1)

        ttk.Label(session_frame, text="User").grid(row=0, column=0, sticky="w")
        self.user_var = tk.StringVar(value="default")
        user_entry = ttk.Entry(session_frame, textvariable=self.user_var)
        user_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(session_frame, text="Mode").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.mode_var = tk.StringVar(value="adaptive")
        mode_box = ttk.Combobox(
            session_frame,
            textvariable=self.mode_var,
            state="readonly",
            values=("adaptive", "baseline"),
        )
        mode_box.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Button(session_frame, text="Apply Session", command=self.reload_current_document).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )

        self.profile_label = tk.Label(
            session_frame,
            text="",
            justify="left",
            anchor="w",
            wraplength=280,
            fg="#333333",
        )
        self.profile_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        main_panel = ttk.Frame(self, padding=12)
        main_panel.grid(row=0, column=1, sticky="nsew")
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(2, weight=1)

        self.header_label = ttk.Label(main_panel, text="Select an `.md` test file.", font=("TkDefaultFont", 13, "bold"))
        self.header_label.grid(row=0, column=0, sticky="w")

        self.meta_label = tk.Label(
            main_panel,
            text="",
            justify="left",
            anchor="w",
            wraplength=980,
            fg="#333333",
        )
        self.meta_label.grid(row=1, column=0, sticky="ew", pady=(6, 10))

        control_row = ttk.Frame(main_panel)
        control_row.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        control_row.columnconfigure(2, weight=1)

        ttk.Button(control_row, text="Reload Session", command=self.reload_current_document).grid(row=0, column=0, sticky="w")
        ttk.Button(control_row, text="Reset Answers", command=self.reset_unchecked).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.submit_button = ttk.Button(control_row, text="Submit Test", command=self.submit_initial_test)
        self.submit_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.continue_button = ttk.Button(
            control_row,
            text="Continue with New Test",
            command=self.start_repair_test_build,
            state="disabled",
        )
        self.continue_button.grid(row=0, column=3, sticky="w", padx=(8, 0))

        self.summary_label = tk.Label(
            control_row,
            text="",
            justify="left",
            anchor="w",
            fg="#1f3a5f",
        )
        self.summary_label.grid(row=0, column=4, sticky="e")

        canvas_frame = ttk.Frame(main_panel)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        canvas_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        canvas_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=canvas_scrollbar.set)

        self.question_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.question_container, anchor="nw")

        self.question_container.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._resize_question_container)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _sync_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_question_container(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def refresh_file_list(self) -> None:
        self.markdown_files = discover_markdown_files(self.cache_root)
        self.file_listbox.delete(0, tk.END)
        for path in self.markdown_files:
            label = str(path.relative_to(self.cache_root.parent))
            self.file_listbox.insert(tk.END, label)

        if self.markdown_files:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(0)
            self.file_listbox.activate(0)
            self.load_selected_file()
        else:
            self.file_path_label.config(text=f"No `{SUPPORTED_MD_GLOB}` files found under {self.cache_root}")

    def open_file_dialog(self) -> None:
        initial_dir = self.cache_root if self.cache_root.exists() else Path.cwd()
        selection = filedialog.askopenfilename(
            title="Open Markdown Test",
            initialdir=str(initial_dir),
            filetypes=[("Markdown Tests", "*.md"), ("Markdown", "*.md"), ("All Files", "*.*")],
        )
        if not selection:
            return
        self.load_document(Path(selection))

    def _current_user_id(self) -> str:
        return self.user_var.get().strip() or "default"

    def reload_current_document(self) -> None:
        if self.current_document is None:
            return
        self.load_document(self.current_document.path)

    def load_selected_file(self) -> None:
        selection = self.file_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index < 0 or index >= len(self.markdown_files):
            return
        self.load_document(self.markdown_files[index])

    def _resolve_figure_path(self, raw_path: str) -> Path | None:
        if self.current_document is None:
            return None
        cleaned = raw_path.strip()
        if not cleaned:
            return None
        candidate = Path(cleaned).expanduser()
        resolved = candidate if candidate.is_absolute() else (self.current_document.path.parent / candidate).resolve()
        return resolved if resolved.exists() else None

    def _build_choice_display(self, question: QuizQuestion, option: str) -> str:
        choice_text = question.choices.get(option, "").strip()
        return f"{option}) {choice_text}" if choice_text else option

    def _load_profile(self) -> StudentProfile:
        profile = load_student_profile(self.profile_root, self._current_user_id())
        self.current_profile = profile
        return profile

    def _save_profile(self) -> None:
        if self.current_profile is None:
            return
        save_student_profile(self.profile_root, self.current_profile)

    def _update_profile_label(self, profile: StudentProfile) -> None:
        tracked = len(profile.concepts)
        weakest_lines = sorted(
            profile.concepts.values(),
            key=lambda state: (state.mastery_score, state.last_seen or ""),
        )[:3]
        weak_text = ", ".join(
            f"{state.concept_label} ({int(round(state.mastery_score * 100))}%)"
            for state in weakest_lines
        )
        lines = [
            f"User: {profile.user_id}",
            f"Tracked concepts: {tracked}",
            f"Mode: {self.mode_var.get().strip() or 'adaptive'}",
        ]
        if weak_text:
            lines.append(f"Lowest mastery: {weak_text}")
        self.profile_label.config(text="\n".join(lines))

    def _show_loading_screen(self, message: str) -> None:
        self.loading_message_var.set(message)
        if self.loading_dialog is not None and self.loading_dialog.winfo_exists():
            self.loading_dialog.deiconify()
            self.loading_dialog.lift()
            self.loading_dialog.focus_force()
            self.update_idletasks()
            return

        dialog = tk.Toplevel(self)
        dialog.title("Preparing New Test")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        dialog.geometry("420x150")

        container = ttk.Frame(dialog, padding=18)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="Building a new test from the missed questions",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            textvariable=self.loading_message_var,
            justify="left",
            anchor="w",
            wraplength=360,
        ).pack(anchor="w", pady=(10, 8))
        progress = ttk.Progressbar(container, mode="indeterminate", length=340)
        progress.pack(anchor="w", pady=(6, 0))
        progress.start(10)

        dialog.grab_set()
        dialog.lift()
        dialog.focus_force()
        self.config(cursor="watch")
        self.loading_dialog = dialog
        self.update_idletasks()

    def _hide_loading_screen(self) -> None:
        self.config(cursor="")
        dialog = self.loading_dialog
        self.loading_dialog = None
        if dialog is None:
            return
        try:
            dialog.grab_release()
        except Exception:  # noqa: BLE001
            pass
        if dialog.winfo_exists():
            dialog.destroy()

    def _load_figure_photo(self, path: Path, max_width: int = 860, max_height: int = 440) -> object:
        if Image is not None and ImageTk is not None:
            with Image.open(path) as image:
                prepared = image.copy()
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            prepared.thumbnail((max_width, max_height), resampling)
            return ImageTk.PhotoImage(prepared)

        photo = tk.PhotoImage(file=str(path))
        scale = max(
            1,
            math.ceil(photo.width() / max_width) if photo.width() > max_width else 1,
            math.ceil(photo.height() / max_height) if photo.height() > max_height else 1,
        )
        return photo.subsample(scale, scale) if scale > 1 else photo

    def _clear_frame(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _resolve_source_document(self, question: QuizQuestion) -> Path | None:
        module_root = self.current_module_root or Path.cwd()
        return resolve_source_file(module_root, question.source_file, question.source_page)

    def _open_with_default_app(self, path: Path) -> bool:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=True)
                return True
            return webbrowser.open(path.as_uri(), new=2)
        except Exception:  # noqa: BLE001
            return False

    def _open_pdf_in_preview(self, path: Path, page_number: int) -> bool:
        if sys.platform != "darwin":
            return self._open_with_default_app(path)

        script = """
on run argv
    set targetPath to POSIX file (item 1 of argv)
    set targetPage to item 2 of argv
    tell application "Preview"
        activate
        open targetPath
    end tell
    delay 0.8
    tell application "System Events"
        tell process "Preview"
            keystroke "g" using {option down, command down}
            delay 0.2
            keystroke targetPage
            delay 0.1
            key code 36
        end tell
    end tell
end run
"""
        try:
            subprocess.run(
                ["osascript", "-e", script, "--", str(path), str(max(1, page_number))],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except Exception:  # noqa: BLE001
            return self._open_with_default_app(path)

    def _open_source_document(self, question: QuizQuestion) -> None:
        resolved_path = self._resolve_source_document(question)
        if resolved_path is None:
            messagebox.showinfo(
                "Source Unavailable",
                f"Could not locate the source file for:\n{question.source or question.source_file or 'Unknown source'}",
            )
            return

        if resolved_path.suffix.lower() == ".pdf":
            opened = self._open_pdf_in_preview(resolved_path, max(1, int(question.source_page or 1)))
        else:
            opened = self._open_with_default_app(resolved_path)
        if not opened:
            messagebox.showinfo(
                "Open Source Failed",
                f"Could not open:\n{resolved_path}",
            )

    def _show_source_button(self, state: QuestionCardState) -> None:
        self._clear_frame(state.remediation_frame)
        resolved_path = self._resolve_source_document(state.question)
        if resolved_path is None:
            message = tk.Label(
                state.remediation_frame,
                text=f"Source file unavailable: {state.question.source or state.question.source_file or 'Unknown source'}",
                justify="left",
                anchor="w",
                wraplength=980,
                fg="#8a6d1d",
            )
            message.grid(row=0, column=0, sticky="ew")
            return

        page_number = max(1, int(state.question.source_page or 1))
        source_kind = "PDF" if resolved_path.suffix.lower() == ".pdf" else "Source File"
        button = ttk.Button(
            state.remediation_frame,
            text=f"Open Source {source_kind} (Page {page_number})",
            command=lambda question=state.question: self._open_source_document(question),
        )
        button.grid(row=0, column=0, sticky="w")

        path_label = tk.Label(
            state.remediation_frame,
            text=str(resolved_path),
            justify="left",
            anchor="w",
            wraplength=980,
            fg="#555555",
        )
        path_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _module_question_pool(self, current_path: Path) -> list[QuizQuestion]:
        mcq_dir = current_path.parent
        if not mcq_dir.exists():
            return []
        deduped: dict[str, QuizQuestion] = {}
        for path in sorted(mcq_dir.glob(SUPPORTED_MD_GLOB)):
            try:
                document = load_quiz_document(path)
            except Exception:
                continue
            for question in document.questions:
                key = question.question_id or question_identity(question)
                if key not in deduped:
                    deduped[key] = question
        return list(deduped.values())

    def _update_summary_label(self) -> None:
        total = len(self.card_states)
        completed = sum(1 for state in self.card_states if state.completed)
        if self.test_submitted:
            incorrect = sum(1 for state in self.card_states if state.completed and state.selected_option != state.question.correct_answer)
            label = "Repair Test" if self.current_stage == "repair" else "Graded"
            summary = f"{label} {completed}/{total} | Incorrect: {incorrect}"
        else:
            answered = sum(1 for state in self.card_states if state.answer_var.get().strip())
            label = "Repair Test" if self.current_stage == "repair" else "Answered"
            summary = f"{label} {answered}/{total}"
        self.summary_label.config(text=summary)

    def _render_questions(self, questions: list[QuizQuestion], *, session_label: str = "", stage: str = "initial") -> None:
        self.current_session_questions = list(questions)
        self.figure_images = []
        self.card_states = []
        self.session_used_question_ids = {question.question_id or question_identity(question) for question in questions}
        self.session_used_question_texts = {question.text.strip().lower() for question in questions if question.text.strip()}
        self.test_submitted = False
        self.current_stage = stage
        self.pending_wrong_states = []
        self.pending_repair_contexts = []
        self.submit_button.configure(state="normal", text="Submit Repair Test" if stage == "repair" else "Submit Test")
        self.continue_button.configure(state="disabled")
        self.summary_label.config(text="")

        for child in self.question_container.winfo_children():
            child.destroy()

        profile = self.current_profile
        for row_index, question in enumerate(self.current_session_questions):
            base_row = row_index * 2
            frame = ttk.Frame(self.question_container, padding=(12, 10))
            frame.grid(row=base_row, column=0, sticky="ew", pady=(0, 10))
            frame.columnconfigure(0, weight=1)

            question_label = tk.Label(
                frame,
                text=f"{question.number}. {question.text}",
                justify="left",
                anchor="w",
                wraplength=980,
                font=("TkDefaultFont", 11, "bold"),
            )
            question_label.grid(row=0, column=0, sticky="ew")

            content_row = 1
            meta_lines: list[str] = []
            if question.source:
                meta_lines.append(f"Source: {question.source}")
            if question.objective:
                meta_lines.append(f"Objective: {question.objective}")
            if question.difficulty_dial:
                meta_lines.append(f"Target Difficulty: {question.difficulty_dial}")
            if question.actual_difficulty:
                meta_lines.append(f"Actual Difficulty: {question.actual_difficulty}")
            if profile is not None:
                mastery = concept_mastery(profile, question)
                concept_label = question_metadata(question)["concept_label"]
                meta_lines.append(f"Concept: {concept_label} | Mastery: {int(round(mastery * 100))}%")

            if meta_lines:
                meta_label = tk.Label(
                    frame,
                    text="\n".join(meta_lines),
                    justify="left",
                    anchor="w",
                    wraplength=980,
                    fg="#555555",
                )
                meta_label.grid(row=content_row, column=0, sticky="ew", pady=(4, 8))
                content_row += 1

            figure_image: object | None = None
            if question.figure_path:
                resolved_figure_path = self._resolve_figure_path(question.figure_path)
                if resolved_figure_path is not None:
                    try:
                        photo = self._load_figure_photo(resolved_figure_path)
                    except Exception as exc:  # noqa: BLE001
                        figure_label = tk.Label(
                            frame,
                            text=f"Figure could not be loaded: {resolved_figure_path}\n{exc}",
                            justify="left",
                            anchor="w",
                            wraplength=980,
                            fg="#8a6d1d",
                        )
                        figure_label.grid(row=content_row, column=0, sticky="ew", pady=(0, 8))
                    else:
                        figure_image = photo
                        self.figure_images.append(photo)
                        figure_label = tk.Label(
                            frame,
                            image=photo,
                            justify="left",
                            anchor="w",
                            bd=1,
                            relief="solid",
                            background="#ffffff",
                        )
                        figure_label.grid(row=content_row, column=0, sticky="w", pady=(0, 4))
                        content_row += 1
                        figure_caption = tk.Label(
                            frame,
                            text=f"Figure: {resolved_figure_path}",
                            justify="left",
                            anchor="w",
                            wraplength=980,
                            fg="#555555",
                        )
                        figure_caption.grid(row=content_row, column=0, sticky="ew", pady=(0, 8))
                else:
                    figure_label = tk.Label(
                        frame,
                        text=f"Figure file not found: {question.figure_path}",
                        justify="left",
                        anchor="w",
                        wraplength=980,
                        fg="#8a6d1d",
                    )
                    figure_label.grid(row=content_row, column=0, sticky="ew", pady=(0, 8))
                content_row += 1

            answer_var = tk.StringVar(value="")
            answer_var.trace_add("write", lambda *_args: self._update_summary_label())
            choice_buttons: list[ttk.Radiobutton] = []
            choice_row = content_row
            for label in ("A", "B", "C", "D"):
                choice_text = question.choices.get(label, "")
                if not choice_text:
                    continue
                radio = ttk.Radiobutton(
                    frame,
                    text=f"{label}) {choice_text}",
                    variable=answer_var,
                    value=label,
                )
                radio.grid(row=choice_row, column=0, sticky="w", pady=2)
                choice_buttons.append(radio)
                choice_row += 1

            note_label = tk.Label(
                frame,
                text=(
                    "Grading happens after you submit the repair test."
                    if stage == "repair"
                    else "Grading happens after you submit the full test."
                ),
                justify="left",
                anchor="w",
                wraplength=980,
                fg="#555555",
            )
            note_label.grid(row=choice_row, column=0, sticky="ew", pady=(8, 0))

            feedback_label = tk.Label(
                frame,
                text="",
                justify="left",
                anchor="w",
                wraplength=980,
            )
            feedback_label.grid(row=choice_row + 1, column=0, sticky="ew", pady=(8, 0))

            remediation_frame = ttk.Frame(frame)
            remediation_frame.grid(row=choice_row + 2, column=0, sticky="ew", pady=(8, 0))
            remediation_frame.columnconfigure(0, weight=1)

            state = QuestionCardState(
                question=question,
                frame=frame,
                answer_var=answer_var,
                choice_buttons=choice_buttons,
                feedback_label=feedback_label,
                check_button=None,
                remediation_frame=remediation_frame,
                figure_image=figure_image,
            )
            self.card_states.append(state)

            separator = ttk.Separator(self.question_container, orient="horizontal")
            separator.grid(row=base_row + 1, column=0, sticky="ew", pady=(0, 10))

        if self.current_document is not None:
            header = self.current_document.module or self.current_document.path.name
            if session_label:
                header = f"{header} | {session_label}"
            self.header_label.config(text=header)
        self._update_summary_label()
        self.canvas.yview_moveto(0.0)

    def load_document(self, path: Path) -> None:
        try:
            document = load_quiz_document(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Failed to Load Test", str(exc))
            return

        self.current_document = document
        self.current_module_root = document.path.parent.parent
        self.current_module_question_pool = self._module_question_pool(document.path)
        profile = self._load_profile()
        self._update_profile_label(profile)

        meta_parts = [
            f"Course: {document.course or 'Unknown'}",
            f"Module: {document.module or 'Unknown'}",
            f"Generated: {document.generated or 'Unknown'}",
            f"Questions: {len(document.questions)}",
            f"Pool: {len(self.current_module_question_pool)}",
            f"User: {profile.user_id}",
            f"Mode: {self.mode_var.get().strip() or 'adaptive'}",
        ]
        self.meta_label.config(text=" | ".join(meta_parts))
        self.file_path_label.config(text=str(document.path))

        seed_text = f"{profile.user_id}:{document.path}"
        scheduled_questions = schedule_questions(
            document.questions,
            profile,
            mode=self.mode_var.get().strip().lower(),
            seed_text=seed_text,
        )
        session_label = "Adaptive Session" if self.mode_var.get().strip().lower() == "adaptive" else "Baseline Session"
        self._render_questions(scheduled_questions, session_label=session_label, stage="initial")

    def reset_unchecked(self) -> None:
        if self.test_submitted:
            return
        for state in self.card_states:
            if state.completed:
                continue
            state.answer_var.set("")
            if not state.busy:
                state.feedback_label.config(text="", fg="#000000")
        self._update_summary_label()

    def _lock_state(self, state: QuestionCardState) -> None:
        for radio in state.choice_buttons:
            radio.configure(state="disabled")
        if state.check_button is not None:
            state.check_button.configure(state="disabled")

    def _format_feedback(
        self,
        *,
        status: str,
        correct_display: str,
        selected_display: str = "",
        diagnostic_summary: str = "",
        misconception_id: str = "",
        why_chosen: str = "",
        why_wrong: str = "",
        why_text: str = "",
    ) -> str:
        sections = [status]
        if selected_display:
            sections.append(f"Your answer:\n{selected_display}")
        sections.append(f"Correct answer:\n{correct_display}")
        if diagnostic_summary or misconception_id:
            diagnostic_lines: list[str] = []
            if diagnostic_summary:
                diagnostic_lines.append(f"Summary: {diagnostic_summary}")
            if misconception_id:
                diagnostic_lines.append(f"ID: {misconception_id}")
            sections.append("Distractor Diagnostics:\n" + "\n".join(diagnostic_lines))
        if why_chosen:
            sections.append(f"Why you might pick it:\n{why_chosen}")
        if why_wrong:
            sections.append(f"Why it is wrong:\n{why_wrong}")
        if why_text:
            sections.append(f"Why:\n{why_text}")
        return "\n\n".join(section for section in sections if section.strip())

    def submit_initial_test(self) -> None:
        if self.current_stage == "repair":
            self.submit_repair_test()
            return
        if self.current_document is None:
            messagebox.showinfo("No Test Loaded", "Load a markdown test file first.")
            return
        if self.test_submitted:
            return
        unanswered = [state.question.number for state in self.card_states if not state.answer_var.get().strip()]
        if unanswered:
            messagebox.showinfo(
                "Finish the Test",
                "Answer every question before submitting.\n\nMissing: " + ", ".join(str(number) for number in unanswered),
            )
            return

        profile = self.current_profile or self._load_profile()
        mode = self.mode_var.get().strip().lower()
        wrong_states: list[QuestionCardState] = []
        for state in self.card_states:
            question = state.question
            selected = state.answer_var.get().strip().upper()
            correct_answer = question.correct_answer.strip().upper()
            correct_display = self._build_choice_display(question, correct_answer) if correct_answer else "?"
            correct = bool(correct_answer and selected == correct_answer)
            state.selected_option = selected
            state.completed = True
            self._lock_state(state)
            update_profile_from_attempt(profile, question, correct=correct, selected_option=selected, answered_at=None)
            if correct:
                self._clear_frame(state.remediation_frame)
                state.feedback_label.config(
                    text=self._format_feedback(status="Correct.", correct_display=correct_display),
                    fg="#2e7d32",
                )
                continue

            wrong_states.append(state)
            selected_display = self._build_choice_display(question, selected)
            diagnostic = question.distractor_diagnostics.get(selected)
            if diagnostic is not None:
                state.feedback_label.config(
                    text=self._format_feedback(
                        status="Incorrect.",
                        selected_display=selected_display,
                        correct_display=correct_display,
                        diagnostic_summary=diagnostic.summary,
                        misconception_id=diagnostic.misconception_id,
                        why_chosen=diagnostic.why_chosen,
                        why_wrong=diagnostic.why_wrong,
                    ),
                    fg="#b3261e",
                )
                self._show_source_button(state)
                continue
            if mode != "adaptive":
                why_text = build_concise_why(question, selected)
                state.feedback_label.config(
                    text=self._format_feedback(
                        status="Incorrect.",
                        selected_display=selected_display,
                        correct_display=correct_display,
                        why_text=why_text,
                    ),
                    fg="#b3261e",
                )
                self._show_source_button(state)
                continue

            why_text = build_concise_why(question, selected)
            state.feedback_label.config(
                text=self._format_feedback(
                    status="Incorrect.",
                    selected_display=selected_display,
                    correct_display=correct_display,
                    why_text=why_text,
                ),
                fg="#b3261e",
            )
            self._show_source_button(state)

        self.test_submitted = True
        self.submit_button.configure(state="disabled")
        self.pending_wrong_states = wrong_states
        self._save_profile()
        self._update_profile_label(profile)
        self._update_summary_label()

        if mode == "adaptive" and wrong_states:
            self.continue_button.configure(state="normal")

    def _question_difficulty_for_tier(self, question: QuizQuestion, tier: str) -> str:
        current = question.difficulty.strip().lower()
        if current not in DIFFICULTY_ORDER:
            actual = question.actual_difficulty.strip().lower().split(" ", 1)[0]
            current = actual if actual in DIFFICULTY_ORDER else "standard"
        level = DIFFICULTY_ORDER.get(current, 1)
        if tier == "easier":
            level = max(0, level - 1)
        elif tier == "transfer":
            level = min(2, level + 1)
        reverse = {value: key for key, value in DIFFICULTY_ORDER.items()}
        return reverse.get(level, "standard")

    def start_repair_test_build(self) -> None:
        if self.current_stage != "initial" or not self.test_submitted:
            return
        if not self.pending_wrong_states:
            return

        self.continue_button.configure(state="disabled", text="Preparing New Test...")
        self.summary_label.config(text="Preparing new test from missed questions...")
        self._show_loading_screen("Gemini is generating a new repair test. The app is locked until it finishes.")
        pool_snapshot = list(self.current_module_question_pool)
        wrong_specs = [
            {
                "question": state.question,
                "selected_option": state.selected_option,
                "started_at": state.intervention_started_at or datetime.now(timezone.utc).isoformat(),
            }
            for state in self.pending_wrong_states
        ]
        exclude_ids = set(self.session_used_question_ids)
        exclude_texts = set(self.session_used_question_texts)
        module_root = self.current_module_root or Path.cwd()

        def worker() -> None:
            try:
                payload = self._build_repair_test_payload(
                    wrong_specs=wrong_specs,
                    module_root=module_root,
                    question_pool=pool_snapshot,
                    exclude_ids=exclude_ids,
                    exclude_texts=exclude_texts,
                )
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "questions": [],
                    "contexts": [],
                    "warning": "",
                    "error": str(exc),
                }
            self.after(0, lambda: self._finish_repair_test_build(payload))

        threading.Thread(target=worker, daemon=True).start()

    def _build_repair_test_payload(
        self,
        *,
        wrong_specs: list[dict[str, object]],
        module_root: Path,
        question_pool: list[QuizQuestion],
        exclude_ids: set[str],
        exclude_texts: set[str],
    ) -> dict[str, object]:
        selected_questions: list[QuizQuestion] = []
        contexts: list[dict[str, object]] = []
        warnings: list[str] = []
        used_ids = set(exclude_ids)
        used_texts = set(exclude_texts)
        generation_specs: list[dict[str, object]] = []

        for spec_index, spec in enumerate(wrong_specs, start=1):
            question = spec["question"]
            if not isinstance(question, QuizQuestion):
                continue
            selected_option = str(spec.get("selected_option") or "").strip().upper()
            evidence = extract_source_evidence(module_root, question)
            local_plan = build_local_repair_plan(
                question_pool,
                question,
                selected_option=selected_option,
                exclude_question_ids=used_ids,
            )
            chosen_local: QuizQuestion | None = None
            for tier_name in ("practice", "easier", "transfer"):
                for item in local_plan:
                    candidate = item.get("question")
                    if not isinstance(item, dict) or not isinstance(candidate, QuizQuestion):
                        continue
                    if str(item.get("tier") or "") != tier_name:
                        continue
                    candidate_id = candidate.question_id or question_identity(candidate)
                    candidate_text = candidate.text.strip().lower()
                    if candidate_id in used_ids or candidate_text in used_texts:
                        continue
                    chosen_local = _clone_for_remediation(
                        candidate,
                        question_kind="repair_test",
                        remediation_tier=str(item.get("tier") or ""),
                        origin="local_pool",
                    )
                    break
                if chosen_local is not None:
                    break

            if chosen_local is not None:
                selected_questions.append(chosen_local)
                used_ids.add(chosen_local.question_id or question_identity(chosen_local))
                if chosen_local.text.strip():
                    used_texts.add(chosen_local.text.strip().lower())
                contexts.append(
                    {
                        "trigger_question": question,
                        "trigger_selected_option": selected_option,
                        "repair_questions": [chosen_local],
                        "started_at": str(spec.get("started_at") or ""),
                    }
                )
                continue

            generation_specs.append(
                {
                    "spec_index": spec_index,
                    "trigger_question": question,
                    "trigger_selected_option": selected_option,
                    "started_at": str(spec.get("started_at") or ""),
                    "concept_label": question_metadata(question)["concept_label"],
                    "difficulty_target": self._question_difficulty_for_tier(question, "practice"),
                    "source_file": question.source_file or str(evidence.get("source_file") or "Unknown source"),
                    "source_page": int(evidence.get("source_page") or question.source_page or 1),
                    "source_text": str(evidence.get("page_text") or ""),
                    "evidence_lines": list(evidence.get("evidence_lines") or []),
                    "misconception_id": (
                        question.distractor_diagnostics.get(selected_option).misconception_id
                        if selected_option in question.distractor_diagnostics
                        else ""
                    ),
                    "misconception_summary": (
                        question.distractor_diagnostics.get(selected_option).summary
                        if selected_option in question.distractor_diagnostics
                        else ""
                    ),
                    "excluded_question_texts": list(used_texts),
                }
            )

        if generation_specs:
            try:
                generated_rows = gemini.generate_batch_remediation_questions(
                    course_name=self.current_document.course if self.current_document is not None else "Unknown course",
                    module_name=self.current_document.module if self.current_document is not None else "Unknown module",
                    remediation_specs=generation_specs,
                    question_kind="repair_test",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"AI remediation generation fallback used: {exc}")
            else:
                for raw_spec, row in zip(generation_specs, generated_rows):
                    repair_question = _generated_row_to_question(
                        row,
                        number=900 + len(selected_questions),
                        question_kind="repair_test",
                        remediation_tier="practice",
                    )
                    selected_questions.append(repair_question)
                    used_ids.add(repair_question.question_id or question_identity(repair_question))
                    if repair_question.text.strip():
                        used_texts.add(repair_question.text.strip().lower())
                    contexts.append(
                        {
                            "trigger_question": raw_spec["trigger_question"],
                            "trigger_selected_option": raw_spec["trigger_selected_option"],
                            "repair_questions": [repair_question],
                            "started_at": raw_spec["started_at"],
                        }
                    )

        return {
            "questions": selected_questions,
            "contexts": contexts,
            "warning": " ".join(part for part in warnings if part).strip(),
        }

    def _finish_repair_test_build(self, payload: dict[str, object]) -> None:
        self._hide_loading_screen()
        questions = [question for question in (payload.get("questions") or []) if isinstance(question, QuizQuestion)]
        warning = str(payload.get("warning") or "").strip()
        error_message = str(payload.get("error") or "").strip()
        contexts = [context for context in (payload.get("contexts") or []) if isinstance(context, dict)]
        self.continue_button.configure(text="Continue with New Test")

        if not questions:
            message = "No follow-up questions could be prepared."
            if error_message:
                message = f"{message}\n\n{error_message}"
            if warning:
                message = f"{message}\n\n{warning}"
            messagebox.showinfo("New Test Unavailable", message)
            self.continue_button.configure(state="disabled")
            self._update_summary_label()
            return

        self.pending_repair_contexts = contexts
        if warning:
            self.summary_label.config(text=f"New test ready. Note: {warning}")
        self._render_questions(questions, session_label="Repair Test", stage="repair")
        self.pending_repair_contexts = contexts

    def submit_repair_test(self) -> None:
        if self.current_stage != "repair" or self.test_submitted:
            return
        unanswered = [state.question.number for state in self.card_states if not state.answer_var.get().strip()]
        if unanswered:
            messagebox.showinfo(
                "Finish the Repair Test",
                "Answer every question before submitting.\n\nMissing: " + ", ".join(str(number) for number in unanswered),
            )
            return

        profile = self.current_profile or self._load_profile()
        for state in self.card_states:
            question = state.question
            selected = state.answer_var.get().strip().upper()
            correct_answer = question.correct_answer.strip().upper()
            correct_display = self._build_choice_display(question, correct_answer) if correct_answer else "?"
            correct = bool(correct_answer and selected == correct_answer)
            state.selected_option = selected
            state.completed = True
            self._lock_state(state)
            update_profile_from_attempt(profile, question, correct=correct, selected_option=selected, answered_at=None)
            if correct:
                self._clear_frame(state.remediation_frame)
                state.feedback_label.config(
                    text=self._format_feedback(status="Correct.", correct_display=correct_display),
                    fg="#2e7d32",
                )
            else:
                diagnostic = question.distractor_diagnostics.get(selected)
                if diagnostic is not None:
                    state.feedback_label.config(
                        text=self._format_feedback(
                            status="Incorrect.",
                            selected_display=self._build_choice_display(question, selected),
                            correct_display=correct_display,
                            diagnostic_summary=diagnostic.summary,
                            misconception_id=diagnostic.misconception_id,
                            why_chosen=diagnostic.why_chosen,
                            why_wrong=diagnostic.why_wrong,
                        ),
                        fg="#b3261e",
                    )
                    self._show_source_button(state)
                else:
                    state.feedback_label.config(
                        text=self._format_feedback(
                            status="Incorrect.",
                            selected_display=self._build_choice_display(question, selected),
                            correct_display=correct_display,
                            why_text=build_concise_why(question, selected),
                        ),
                        fg="#b3261e",
                    )
                    self._show_source_button(state)

        for context in self.pending_repair_contexts:
            trigger_question = context.get("trigger_question")
            repair_questions = context.get("repair_questions")
            if not isinstance(trigger_question, QuizQuestion) or not isinstance(repair_questions, list):
                continue
            repair_correct_count = 0
            matched_questions = [question for question in repair_questions if isinstance(question, QuizQuestion)]
            for repair_question in matched_questions:
                matching_state = next(
                    (state for state in self.card_states if (state.question.question_id or question_identity(state.question)) == (repair_question.question_id or question_identity(repair_question))),
                    None,
                )
                if matching_state is not None and matching_state.selected_option == matching_state.question.correct_answer:
                    repair_correct_count += 1
            record_intervention_result(
                profile,
                trigger_question,
                trigger_selected_option=str(context.get("trigger_selected_option") or ""),
                repair_questions=matched_questions,
                repair_correct_count=repair_correct_count,
                retest_questions=[],
                retest_correct_count=0,
                started_at=str(context.get("started_at") or ""),
            )

        self.test_submitted = True
        self.submit_button.configure(state="disabled")
        self.continue_button.configure(state="disabled")
        self._save_profile()
        self._update_profile_label(profile)
        self._update_summary_label()


def main() -> None:
    cache_root = Path(__file__).resolve().parent / "Cache"
    app = QuizApp(cache_root=cache_root)
    app.mainloop()


if __name__ == "__main__":
    main()
