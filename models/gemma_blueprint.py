from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .gemma_common import (
    BLUEPRINT_ALIGNMENT_MIN_PHRASE_HITS,
    BLUEPRINT_ALIGNMENT_MIN_TOKEN_HITS,
    BLUEPRINT_DIFFICULTIES,
    BLUEPRINT_SKILLS,
    DEFAULT_DIFFICULTY_DIALS,
    DEPTH_MIN_PASS_RATIO,
    DEPTH_PASS_SCORE,
    DIFFICULTY_CALIBRATION_MARGIN,
    DIFFICULTY_CALIBRATION_PATH,
    MIN_DIFFICULTY_PILOT_SAMPLES,
    RECALL_MAX_RATIO,
    _BLUEPRINT_STOPWORDS,
    extract_json_object,
    normalize_text,
)


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
    cleaned = normalize_text(str(value or "")).lower()
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
        term = normalize_text(raw_term).strip(" -")
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
    topic = normalize_text(module_name)
    topic = re.sub(r"^(unit|chapter)\s*\d+\s*[:\-]\s*", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"^(unit|chapter)\s*\d+\s*", "", topic, flags=re.IGNORECASE)
    return topic.strip(" -:") or normalize_text(module_name)


def _extract_blueprint_topic_candidates(module_name: str, study_text: str, limit: int) -> list[str]:
    cleaned_text = re.sub(r"\[(?:SRC|FIG_SRC|FIG_IMAGE)[^\]]*\]", " ", study_text)
    cleaned_text = normalize_text(cleaned_text)
    base_topic = _clean_module_blueprint_topic(module_name)

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        candidate = normalize_text(value).strip(" -:;,")
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
    # This fallback keeps quiz generation alive even when the model cannot
    # produce a usable blueprint; coverage is approximated from local evidence.
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

        slot_id = normalize_text(str(row.get("slot_id") or "")).lower() or f"bp{index:02d}"
        objective_label = normalize_text(str(row.get("objective_label") or ""))
        topic = normalize_text(str(row.get("topic") or ""))
        objective_id = normalize_text(str(row.get("objective_id") or ""))
        if not objective_id and objective_label and topic:
            objective_id = f"obj_{abs(hash((objective_label.lower(), topic.lower()))) % 10**8:08d}"

        skill = _normalize_blueprint_skill(row.get("skill"))
        difficulty = _normalize_blueprint_difficulty(row.get("difficulty"))

        key_terms_raw = row.get("key_terms")
        key_terms: list[str] = []
        seen_terms: set[str] = set()
        if isinstance(key_terms_raw, list):
            for term in key_terms_raw:
                cleaned_term = normalize_text(str(term or "")).strip(" -")
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
    unique_cells = len(
        {
            (
                str(slot.get("topic") or "").strip().lower(),
                str(slot.get("skill") or "").strip().lower(),
                str(slot.get("difficulty") or "").strip().lower(),
            )
            for slot in slots
        }
    )

    min_topics = 1 if total == 1 else min(total, max(2, int(math.ceil(total * 0.4))))
    min_skills = 1 if total == 1 else min(total, max(2, int(math.ceil(total * 0.35))))
    min_difficulties = 1 if total <= 3 else 2
    passed = (
        len(unique_topics) >= min_topics
        and len(unique_skills) >= min_skills
        and len(unique_difficulties) >= min_difficulties
    )

    quality_score = (
        (len(unique_topics) * 3.0)
        + (len(unique_skills) * 2.5)
        + (len(unique_difficulties) * 2.0)
        + (unique_cells * 0.75)
    )

    feedback_parts: list[str] = []
    if len(unique_topics) < min_topics:
        feedback_parts.append(
            f"Increase topic coverage: only {len(unique_topics)} unique topic(s), need at least {min_topics}."
        )
    if len(unique_skills) < min_skills:
        feedback_parts.append(
            f"Increase skill coverage: only {len(unique_skills)} unique skill(s), need at least {min_skills}."
        )
    if len(unique_difficulties) < min_difficulties:
        feedback_parts.append(
            f"Increase difficulty spread: only {len(unique_difficulties)} level(s), need at least {min_difficulties}."
        )
    feedback = " ".join(feedback_parts).strip() or "Blueprint coverage passed."

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
                parsed_payload = extract_json_object(response_text)
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
    slot_id = normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
    slot = blueprint_lookup.get(slot_id)
    if slot is None:
        return {"passed": False, "score": -5.0, "feedback": "Missing or unknown blueprint slot."}

    if normalize_text(str(question.get("objective_id") or "")) != str(slot.get("objective_id") or ""):
        return {"passed": False, "score": -4.0, "feedback": "Question objective_id did not match its blueprint slot."}
    if normalize_text(str(question.get("objective_label") or "")).lower() != str(slot.get("objective_label") or "").lower():
        return {"passed": False, "score": -4.0, "feedback": "Question objective_label did not match its blueprint slot."}
    if normalize_text(str(question.get("objective_topic") or "")).lower() != str(slot.get("topic") or "").lower():
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
        slot_id = normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
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
    text_blob = normalize_text(f"{stem} {choice_text}")
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
    slot_id = normalize_text(str(question.get("blueprint_slot_id") or "")).lower()
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
        message = normalize_text(str(raw_message or ""))
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
