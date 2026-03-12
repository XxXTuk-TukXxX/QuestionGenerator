from __future__ import annotations

import io
import json
import random
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None

try:
    from docx import Document
except Exception:  # noqa: BLE001
    Document = None

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None


OBJECTIVE_TOPIC_RE = re.compile(r"\bTopic:\s*(.+?)(?:\s*\||$)", flags=re.IGNORECASE)
OBJECTIVE_SKILL_RE = re.compile(r"\bSkill:\s*(.+?)(?:\s*\||$)", flags=re.IGNORECASE)
OBJECTIVE_DIFFICULTY_RE = re.compile(r"\bDifficulty:\s*(.+?)(?:\s*\||$)", flags=re.IGNORECASE)
SOURCE_PAGE_RE = re.compile(r"\bp\.(\d+)\b", flags=re.IGNORECASE)
SOURCE_FIG_RE = re.compile(r"\b(fig#\d+)\b", flags=re.IGNORECASE)
SOURCE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "have", "will",
    "what", "which", "when", "where", "they", "their", "about", "because", "under",
    "factor", "markets", "market", "question", "correct", "answer", "would", "most",
}
SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx"}
DIFFICULTY_ORDER = {"foundation": 0, "standard": 1, "challenge": 2}
TIER_LABELS = ("easier", "practice", "transfer")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class StudentConceptState:
    concept_key: str
    concept_label: str
    mastery_score: float = 0.5
    last_seen: str = ""
    correct_count: int = 0
    incorrect_count: int = 0
    error_type_counts: dict[str, int] = field(default_factory=dict)
    misconception_counts: dict[str, int] = field(default_factory=dict)
    intervention_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StudentProfile:
    user_id: str
    updated_at: str = ""
    concepts: dict[str, StudentConceptState] = field(default_factory=dict)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "default_user"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _canonical_source_name(filename: str) -> str:
    path = Path(str(filename or "").strip())
    stem = re.sub(r"(?:_\d+)+$", "", path.stem.lower())
    suffix = path.suffix.lower()
    return f"{stem}{suffix}" if suffix else stem


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _question_attr(question: Any, name: str) -> Any:
    return getattr(question, name, "")


def source_reference(question: Any) -> dict[str, Any]:
    source = str(_question_attr(question, "source") or "").strip()
    source_file = str(_question_attr(question, "source_file") or "").strip()
    source_page_value = _question_attr(question, "source_page")
    source_figure_id = str(_question_attr(question, "source_figure_id") or "").strip()

    if not source_file and source:
        source_file = re.sub(r"\s+p\.\d+.*$", "", source, flags=re.IGNORECASE).strip()
    if not source_file:
        source_file = "Unknown source"

    source_page = 1
    if isinstance(source_page_value, int) and source_page_value >= 1:
        source_page = int(source_page_value)
    elif source:
        match = SOURCE_PAGE_RE.search(source)
        if match:
            source_page = max(1, int(match.group(1)))

    if not source_figure_id and source:
        fig_match = SOURCE_FIG_RE.search(source)
        if fig_match:
            source_figure_id = fig_match.group(1).strip()

    return {
        "source_file": source_file,
        "source_page": source_page,
        "source_figure_id": source_figure_id,
    }


def _parse_objective_fallback(question: Any) -> tuple[str, str, str]:
    objective = str(_question_attr(question, "objective") or "").strip()
    topic_match = OBJECTIVE_TOPIC_RE.search(objective)
    skill_match = OBJECTIVE_SKILL_RE.search(objective)
    difficulty_match = OBJECTIVE_DIFFICULTY_RE.search(objective)
    topic = topic_match.group(1).strip() if topic_match else ""
    skill = skill_match.group(1).strip() if skill_match else ""
    difficulty = difficulty_match.group(1).strip() if difficulty_match else ""
    return topic, skill, difficulty


def question_metadata(question: Any) -> dict[str, str]:
    topic = str(_question_attr(question, "topic") or "").strip()
    skill = str(_question_attr(question, "skill") or "").strip()
    difficulty = str(_question_attr(question, "difficulty") or "").strip()
    if not (topic and skill and difficulty):
        fallback_topic, fallback_skill, fallback_difficulty = _parse_objective_fallback(question)
        topic = topic or fallback_topic
        skill = skill or fallback_skill
        difficulty = difficulty or fallback_difficulty

    objective_label = str(_question_attr(question, "objective_label") or "").strip()
    objective = str(_question_attr(question, "objective") or "").strip()
    blueprint_slot_id = str(_question_attr(question, "blueprint_slot_id") or "").strip()
    concept_label = topic or objective_label or objective or f"Question {_question_attr(question, 'number')}"
    concept_key = _slugify(concept_label)
    return {
        "concept_key": concept_key,
        "concept_label": concept_label,
        "topic": topic,
        "skill": skill,
        "difficulty": difficulty,
        "objective_label": objective_label,
        "blueprint_slot_id": blueprint_slot_id,
    }


def profile_path(profile_root: Path, user_id: str) -> Path:
    return profile_root.expanduser().resolve() / f"{_slugify(user_id)}.json"


def load_student_profile(profile_root: Path, user_id: str) -> StudentProfile:
    path = profile_path(profile_root, user_id)
    if not path.exists():
        return StudentProfile(user_id=user_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts_payload = payload.get("concepts") if isinstance(payload, dict) else {}
    concepts: dict[str, StudentConceptState] = {}
    if isinstance(concepts_payload, dict):
        for concept_key, row in concepts_payload.items():
            if not isinstance(row, dict):
                continue
            concepts[str(concept_key)] = StudentConceptState(
                concept_key=str(concept_key),
                concept_label=str(row.get("concept_label") or concept_key),
                mastery_score=float(row.get("mastery_score") or 0.5),
                last_seen=str(row.get("last_seen") or ""),
                correct_count=int(row.get("correct_count") or 0),
                incorrect_count=int(row.get("incorrect_count") or 0),
                error_type_counts=dict(row.get("error_type_counts") or {}),
                misconception_counts=dict(row.get("misconception_counts") or {}),
                intervention_history=list(row.get("intervention_history") or []),
            )
    return StudentProfile(
        user_id=str(payload.get("user_id") or user_id) if isinstance(payload, dict) else user_id,
        updated_at=str(payload.get("updated_at") or "") if isinstance(payload, dict) else "",
        concepts=concepts,
    )


def save_student_profile(profile_root: Path, profile: StudentProfile) -> Path:
    path = profile_path(profile_root, profile.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": profile.user_id,
        "updated_at": profile.updated_at or _now_iso(),
        "concepts": {
            concept_key: asdict(state)
            for concept_key, state in sorted(profile.concepts.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def _classify_error_type(question: Any, selected_option: str) -> tuple[str, str]:
    diagnostics = _question_attr(question, "distractor_diagnostics")
    diagnostic = diagnostics.get(selected_option) if isinstance(diagnostics, dict) else None
    summary = str(getattr(diagnostic, "summary", "") or "").lower()
    why_wrong = str(getattr(diagnostic, "why_wrong", "") or "").lower()
    blob = f"{summary} {why_wrong}".strip()

    misconception_id = str(getattr(diagnostic, "misconception_id", "") or "").strip()
    if re.search(r"\bdefine|definition|classif|identify|categor", blob):
        return "definition_confusion", misconception_id
    if re.search(r"\bcause|effect|lead|result|impact|shift", blob):
        return "cause_effect_confusion", misconception_id
    if re.search(r"\bcalculate|formula|multiply|divide|revenue|ratio|marginal|compute", blob):
        return "formula_or_calculation_error", misconception_id
    if re.search(r"\bgraph|curve|figure|slope|axis", blob):
        return "graph_interpretation_error", misconception_id
    if re.search(r"\bmonopoly|monopsony|competitive|market\b", blob):
        return "market_structure_confusion", misconception_id
    if re.search(r"\bcompare|difference|versus|instead\b", blob):
        return "comparison_confusion", misconception_id
    return "general_misconception", misconception_id


def selected_misconception_id(question: Any, selected_option: str) -> str:
    diagnostics = _question_attr(question, "distractor_diagnostics")
    diagnostic = diagnostics.get(selected_option) if isinstance(diagnostics, dict) else None
    return str(getattr(diagnostic, "misconception_id", "") or "").strip()


def update_profile_from_attempt(
    profile: StudentProfile,
    question: Any,
    *,
    correct: bool,
    selected_option: str,
    answered_at: str | None = None,
) -> StudentConceptState:
    metadata = question_metadata(question)
    concept_key = metadata["concept_key"]
    state = profile.concepts.get(concept_key)
    if state is None:
        state = StudentConceptState(
            concept_key=concept_key,
            concept_label=metadata["concept_label"],
        )
        profile.concepts[concept_key] = state

    if correct:
        state.correct_count += 1
    else:
        state.incorrect_count += 1
        error_type, misconception_id = _classify_error_type(question, selected_option)
        state.error_type_counts[error_type] = int(state.error_type_counts.get(error_type) or 0) + 1
        if misconception_id:
            state.misconception_counts[misconception_id] = int(state.misconception_counts.get(misconception_id) or 0) + 1

    total_attempts = state.correct_count + state.incorrect_count
    state.mastery_score = round((state.correct_count + 1.0) / (total_attempts + 2.0), 4)
    state.last_seen = answered_at or _now_iso()
    state.concept_label = metadata["concept_label"]
    profile.updated_at = state.last_seen
    return state


def concept_mastery(profile: StudentProfile, question: Any) -> float:
    metadata = question_metadata(question)
    state = profile.concepts.get(metadata["concept_key"])
    return float(state.mastery_score) if state is not None else 0.5


def question_identity(question: Any) -> str:
    question_id = str(_question_attr(question, "question_id") or "").strip()
    if question_id:
        return question_id
    source = source_reference(question)
    seed = "||".join(
        [
            str(_question_attr(question, "text") or "").strip(),
            str(_question_attr(question, "correct_answer") or "").strip(),
            source["source_file"],
            str(source["source_page"]),
        ]
    )
    return f"q_{sha1(seed.encode('utf-8')).hexdigest()[:16]}"


def question_text_signature(question: Any) -> str:
    return _normalize_text(str(_question_attr(question, "text") or "").strip())


def _hours_since(timestamp: str, *, now: datetime) -> float:
    if not timestamp:
        return 9999.0
    try:
        seen = datetime.fromisoformat(timestamp)
    except ValueError:
        return 9999.0
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    return max(0.0, (now - seen.astimezone(timezone.utc)).total_seconds() / 3600.0)


def _difficulty_label(question: Any) -> str:
    difficulty = str(_question_attr(question, "difficulty") or "").strip().lower()
    if difficulty in DIFFICULTY_ORDER:
        return difficulty
    actual_difficulty = str(_question_attr(question, "actual_difficulty") or "").strip().lower()
    if actual_difficulty:
        actual_label = actual_difficulty.split(" ", 1)[0]
        if actual_label in DIFFICULTY_ORDER:
            return actual_label
    return "standard"


def _source_candidate_files(module_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for folder_name in ("tasked", "submitted"):
        root = module_dir / folder_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES:
                candidates.append(path)
    return sorted(set(candidates))


def _extract_pdf_page_text(path: Path, page_number: int) -> str:
    if PdfReader is None or page_number < 1:
        return ""
    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            reader = PdfReader(str(path))
            index = page_number - 1
            if index < 0 or index >= len(reader.pages):
                return ""
            return str(reader.pages[index].extract_text() or "")
    except Exception:  # noqa: BLE001
        return ""


def _extract_docx_text(path: Path) -> str:
    if Document is None:
        return ""
    try:
        document = Document(str(path))
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def _extract_image_text(path: Path) -> str:
    if cv2 is None or np is None or pytesseract is None:
        return ""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        encoded = None

    image = None
    if encoded is not None and getattr(encoded, "size", 0) > 0:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return ""

    try:
        return str(pytesseract.image_to_string(image) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _extract_source_text(path: Path, page_number: int) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_page_text(path, page_number)
    if suffix in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _extract_image_text(path)
    return ""


def resolve_source_file(module_dir: Path, source_file: str, source_page: int = 1) -> Path | None:
    target_name = _canonical_source_name(source_file)
    if not target_name:
        return None
    candidates = [
        path
        for path in _source_candidate_files(module_dir)
        if _canonical_source_name(path.name) == target_name
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranked: list[tuple[int, int, str, Path]] = []
    for path in candidates:
        page_text = _extract_source_text(path, source_page)
        ranked.append((len(page_text.strip()), len(path.name), str(path), path))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    return ranked[0][3]


def _line_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{3,}", _normalize_text(text))
        if token not in SOURCE_STOPWORDS
    }


def _split_source_lines(text: str) -> list[str]:
    if not text:
        return []
    raw_lines = [segment.strip() for segment in text.splitlines()]
    lines = [line for line in raw_lines if line]
    if len(lines) >= 3:
        return lines
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if sentences:
        return sentences
    return lines


def _score_evidence_line(line: str, tokens: set[str], answer_text: str, concept_bits: list[str]) -> float:
    normalized_line = _normalize_text(line)
    line_tokens = _line_tokens(line)
    if not normalized_line:
        return 0.0
    overlap = len(tokens & line_tokens)
    answer_hits = 1 if answer_text and _normalize_text(answer_text) in normalized_line else 0
    concept_hits = sum(1 for bit in concept_bits if bit and bit in normalized_line)
    exact_hits = sum(1 for token in tokens if token and token in normalized_line)
    if overlap <= 0 and answer_hits <= 0 and exact_hits < 2:
        return 0.0
    return (overlap * 2.5) + (answer_hits * 3.5) + (concept_hits * 1.5) + (exact_hits * 0.3)


def extract_source_evidence(module_dir: Path, question: Any, *, limit: int = 3) -> dict[str, Any]:
    source = source_reference(question)
    resolved_path = resolve_source_file(module_dir, source["source_file"], source["source_page"])
    if resolved_path is None:
        return {
            "available": False,
            "resolved_source_path": None,
            "source_file": source["source_file"],
            "source_page": source["source_page"],
            "page_text": "",
            "evidence_lines": [],
            "message": "Evidence unavailable on cited page.",
        }

    page_text = _extract_source_text(resolved_path, source["source_page"])
    if not page_text.strip():
        return {
            "available": False,
            "resolved_source_path": str(resolved_path),
            "source_file": source["source_file"],
            "source_page": source["source_page"],
            "page_text": "",
            "evidence_lines": [],
            "message": "Evidence unavailable on cited page.",
        }

    lines = _split_source_lines(page_text)
    correct_answer = str(_question_attr(question, "correct_answer") or "").strip().upper()
    answer_text = str((_question_attr(question, "choices") or {}).get(correct_answer, "") or "").strip()
    metadata = question_metadata(question)
    concept_bits = [
        _normalize_text(metadata.get("concept_label") or ""),
        _normalize_text(metadata.get("topic") or ""),
        _normalize_text(metadata.get("objective_label") or ""),
    ]
    tokens = _line_tokens(str(_question_attr(question, "text") or ""))
    tokens |= _line_tokens(answer_text)
    tokens |= _line_tokens(metadata.get("concept_label") or "")

    scored_lines: list[tuple[float, int, str]] = []
    for index, line in enumerate(lines):
        score = _score_evidence_line(line, tokens, answer_text, concept_bits)
        if score > 0:
            scored_lines.append((score, index, line))

    if not scored_lines:
        return {
            "available": False,
            "resolved_source_path": str(resolved_path),
            "source_file": source["source_file"],
            "source_page": source["source_page"],
            "page_text": page_text,
            "evidence_lines": [],
            "message": "Evidence unavailable on cited page.",
        }

    chosen = sorted(scored_lines, key=lambda row: (-row[0], row[1]))[: max(1, limit)]
    evidence_lines = [line for _, _, line in sorted(chosen, key=lambda row: row[1])]
    return {
        "available": True,
        "resolved_source_path": str(resolved_path),
        "source_file": source["source_file"],
        "source_page": source["source_page"],
        "page_text": page_text,
        "evidence_lines": evidence_lines,
        "message": "",
    }


def _first_sentences(text: str, *, max_sentences: int = 2) -> str:
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(str(text or "").strip()) if segment.strip()]
    if not parts:
        return ""
    return " ".join(parts[:max_sentences]).strip()


def build_concise_why(question: Any, selected_option: str) -> str:
    diagnostics = _question_attr(question, "distractor_diagnostics")
    diagnostic = diagnostics.get(selected_option) if isinstance(diagnostics, dict) else None
    why_wrong = str(getattr(diagnostic, "why_wrong", "") or "").strip()
    if why_wrong:
        concise = _first_sentences(why_wrong, max_sentences=2)
        if concise:
            return concise
    explanation = str(_question_attr(question, "explanation") or _question_attr(question, "why") or "").strip()
    if explanation:
        concise = _first_sentences(explanation, max_sentences=2)
        if concise:
            return concise
    correct_answer = str(_question_attr(question, "correct_answer") or "").strip().upper()
    choices = _question_attr(question, "choices")
    correct_choice = ""
    if isinstance(choices, dict):
        correct_choice = str(choices.get(correct_answer, "") or "").strip()
    metadata = question_metadata(question)
    concept_label = metadata.get("concept_label") or "the concept"
    if correct_choice:
        return (
            f"The correct answer is {correct_answer} because it matches {concept_label.lower()} and the cited evidence. "
            f"{correct_choice}"
        ).strip()
    return (
        f"The correct answer matches {concept_label.lower()} and the cited evidence, while the selected option does not."
    )


def _preferred_misconception_match(question: Any, misconception_id: str) -> bool:
    diagnostics = _question_attr(question, "distractor_diagnostics")
    if not isinstance(diagnostics, dict) or not misconception_id:
        return False
    return any(
        str(getattr(diagnostic, "misconception_id", "") or "").strip() == misconception_id
        for diagnostic in diagnostics.values()
    )


def build_local_repair_plan(
    questions: list[Any],
    trigger_question: Any,
    *,
    selected_option: str,
    exclude_question_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not questions:
        return []
    exclude_ids = set(exclude_question_ids or set())
    trigger_meta = question_metadata(trigger_question)
    trigger_concept = trigger_meta["concept_key"]
    trigger_difficulty = _difficulty_label(trigger_question)
    trigger_level = DIFFICULTY_ORDER.get(trigger_difficulty, 1)
    target_misconception = selected_misconception_id(trigger_question, selected_option)
    target_tiers = {
        "easier": max(0, trigger_level - 1),
        "practice": trigger_level,
        "transfer": min(max(DIFFICULTY_ORDER.values()), trigger_level + 1),
    }

    candidates: list[Any] = []
    for question in questions:
        question_id = question_identity(question)
        if question_id in exclude_ids:
            continue
        if question_metadata(question)["concept_key"] != trigger_concept:
            continue
        if question_text_signature(question) == question_text_signature(trigger_question):
            continue
        candidates.append(question)

    planned: list[dict[str, Any]] = []
    used_ids = set(exclude_ids)
    for tier_name in TIER_LABELS:
        desired_level = target_tiers[tier_name]
        ranked: list[tuple[float, Any]] = []
        for candidate in candidates:
            candidate_id = question_identity(candidate)
            if candidate_id in used_ids:
                continue
            candidate_level = DIFFICULTY_ORDER.get(_difficulty_label(candidate), 1)
            distance_penalty = abs(candidate_level - desired_level)
            misconception_bonus = 3.0 if _preferred_misconception_match(candidate, target_misconception) else 0.0
            skill_bonus = 1.0 if question_metadata(candidate).get("skill") == trigger_meta.get("skill") else 0.0
            ranked.append((misconception_bonus + skill_bonus - (distance_penalty * 2.0), candidate))
        ranked.sort(
            key=lambda row: (
                -row[0],
                abs(DIFFICULTY_ORDER.get(_difficulty_label(row[1]), 1) - desired_level),
                int(_question_attr(row[1], "number") or 0),
            )
        )
        if not ranked:
            continue
        selected = ranked[0][1]
        used_ids.add(question_identity(selected))
        planned.append({"tier": tier_name, "question": selected, "source": "local"})
    return planned


def record_intervention_result(
    profile: StudentProfile,
    trigger_question: Any,
    *,
    trigger_selected_option: str,
    repair_questions: list[Any],
    repair_correct_count: int,
    retest_questions: list[Any],
    retest_correct_count: int,
    started_at: str | None = None,
) -> dict[str, Any]:
    metadata = question_metadata(trigger_question)
    concept_key = metadata["concept_key"]
    state = profile.concepts.get(concept_key)
    if state is None:
        state = StudentConceptState(
            concept_key=concept_key,
            concept_label=metadata["concept_label"],
        )
        profile.concepts[concept_key] = state

    record = {
        "started_at": started_at or _now_iso(),
        "trigger_question_id": question_identity(trigger_question),
        "trigger_selected_option": str(trigger_selected_option or "").strip().upper(),
        "trigger_misconception_id": selected_misconception_id(trigger_question, trigger_selected_option),
        "repair_question_ids": [question_identity(question) for question in repair_questions],
        "repair_correct_count": int(repair_correct_count),
        "repair_total": len(repair_questions),
        "retest_question_ids": [question_identity(question) for question in retest_questions],
        "retest_correct_count": int(retest_correct_count),
        "retest_total": len(retest_questions),
        "intervention_success": bool(retest_questions) and retest_correct_count == len(retest_questions),
    }
    state.intervention_history.append(record)
    profile.updated_at = _now_iso()
    return record


def schedule_questions(
    questions: list[Any],
    profile: StudentProfile,
    *,
    mode: str,
    seed_text: str = "",
) -> list[Any]:
    pool = list(questions)
    if mode != "adaptive":
        rng = random.Random(seed_text or profile.user_id)
        randomized = list(pool)
        rng.shuffle(randomized)
        return randomized

    remaining = list(pool)
    ordered: list[Any] = []
    recent_concepts: list[str] = []
    now = datetime.now(timezone.utc)

    while remaining:
        best_question: Any | None = None
        best_score = float("-inf")
        # Adaptive ordering prioritizes weak concepts while spacing repeats out
        # enough to avoid immediate re-asking of the same idea.
        for question in remaining:
            metadata = question_metadata(question)
            state = profile.concepts.get(metadata["concept_key"])
            mastery = float(state.mastery_score) if state is not None else 0.5
            low_mastery_score = (1.0 - mastery) * 3.0
            last_seen = str(state.last_seen) if state is not None else ""
            spacing_bonus = min(_hours_since(last_seen, now=now) / 24.0, 3.0)
            recent_penalty = 1.5 if metadata["concept_key"] in recent_concepts[-2:] else 0.0
            score = low_mastery_score + spacing_bonus - recent_penalty
            if best_question is None or score > best_score:
                best_question = question
                best_score = score

        if best_question is None:
            break
        ordered.append(best_question)
        recent_concepts.append(question_metadata(best_question)["concept_key"])
        remaining.remove(best_question)

    return ordered
