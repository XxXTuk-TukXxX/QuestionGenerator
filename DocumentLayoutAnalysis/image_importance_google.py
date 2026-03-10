from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DEFAULT_GOOGLE_MODEL = "models/gemma-3-27b-it"
PROMPT_VERSION = "v1"


@dataclass
class ImageImportanceDecision:
    important: bool
    confidence: float
    category: str
    reason: str
    cache_key: str
    source_file: str
    source_page: int
    source_figure_id: str
    error: str | None = None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "important", "keep", "1"}:
            return True
        if lowered in {"false", "no", "not important", "discard", "delete", "0"}:
            return False
    return None


def _normalize_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    if isinstance(value, str):
        try:
            return _clamp(float(value.strip()))
        except ValueError:
            return None
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
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

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None

    candidate = match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _save_cache(cache_path: Path | None, payload: dict[str, dict[str, Any]]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_cache_key(
    *,
    model: str,
    image_bytes: bytes,
    associated_text: str,
) -> str:
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:24]
    text_hash = hashlib.sha256(associated_text.strip().lower().encode("utf-8", "ignore")).hexdigest()[:16]
    model_hash = hashlib.sha256(model.encode("utf-8", "ignore")).hexdigest()[:12]
    prompt_hash = hashlib.sha256(PROMPT_VERSION.encode("utf-8", "ignore")).hexdigest()[:8]
    return f"{model_hash}:{prompt_hash}:{image_hash}:{text_hash}"


def _build_prompt(*, source_file: str, source_page: int, source_figure_id: str, associated_text: str) -> str:
    text_context = associated_text.strip()
    if len(text_context) > 1400:
        text_context = text_context[:1400].rstrip() + "..."

    return (
        "Classify whether this educational figure should be kept for generating rigorous study MCQs.\n"
        "KEEP important figures like: graphs, charts, tables, equations, concept diagrams, or visuals tied to analysis/data.\n"
        "DELETE unimportant figures like: decorative photos, icons, logos, unrelated pictures, aesthetic backgrounds.\n"
        "Use both the image and nearby text context.\n"
        "Return strict JSON only with this schema:\n"
        '{"important": true|false, "confidence": 0.0-1.0, "category": "graph|table|diagram|equation|photo|other", "reason": "..."}\n'
        f"Source: {source_file} page {source_page} {source_figure_id}\n"
        "Nearby text context:\n"
        f"{text_context or '[none]'}"
    )


def classify_figures_with_google(
    figures: list[dict[str, Any]],
    *,
    api_key: str,
    model: str = DEFAULT_GOOGLE_MODEL,
    cache_path: str | Path | None = None,
) -> list[ImageImportanceDecision]:
    if not api_key.strip():
        raise RuntimeError("Missing GOOGLE_API_KEY for Google image importance classification.")

    resolved_model = model.strip() or DEFAULT_GOOGLE_MODEL
    cache_file = Path(cache_path).expanduser().resolve() if cache_path is not None else None
    cache_payload = _load_cache(cache_file)
    cache_changed = False

    client = genai.Client(api_key=api_key)
    decisions: list[ImageImportanceDecision] = []

    for figure in figures:
        source_file = str(figure.get("source_file") or "").strip()
        source_page_raw = figure.get("source_page")
        source_page = int(source_page_raw) if isinstance(source_page_raw, (int, float, str)) else 1
        source_figure_id = str(figure.get("source_figure_id") or "").strip()
        associated_text = str(figure.get("associated_text") or "").strip()
        image_bytes = figure.get("image_bytes")

        if not isinstance(image_bytes, (bytes, bytearray)) or not source_file or not source_figure_id:
            raise RuntimeError(
                "Google importance classifier received an invalid figure payload "
                f"(source={source_file or 'unknown'} {source_figure_id or 'unknown'})."
            )

        cache_key = _build_cache_key(
            model=resolved_model,
            image_bytes=bytes(image_bytes),
            associated_text=associated_text,
        )

        cached = cache_payload.get(cache_key)
        if isinstance(cached, dict):
            important = _normalize_bool(cached.get("important"))
            confidence = _normalize_float(cached.get("confidence"))
            category = str(cached.get("category") or "other").strip() or "other"
            reason = str(cached.get("reason") or "cached decision").strip() or "cached decision"
            if important is not None and confidence is not None:
                decisions.append(
                    ImageImportanceDecision(
                        important=important,
                        confidence=confidence,
                        category=category,
                        reason=reason,
                        cache_key=cache_key,
                        source_file=source_file,
                        source_page=source_page,
                        source_figure_id=source_figure_id,
                        error=None,
                    )
                )
                continue

        prompt = _build_prompt(
            source_file=source_file,
            source_page=source_page,
            source_figure_id=source_figure_id,
            associated_text=associated_text,
        )

        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=[prompt, types.Part.from_bytes(data=bytes(image_bytes), mime_type="image/png")],
                config=types.GenerateContentConfig(temperature=0),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Google importance classification request failed "
                f"for {source_file} p.{source_page} {source_figure_id}: {exc}"
            ) from exc

        response_text = response.text if isinstance(response.text, str) else ""
        parsed = _extract_json_object(response_text)
        if parsed is None:
            raise RuntimeError(
                "Google importance classifier returned unparsable JSON "
                f"for {source_file} p.{source_page} {source_figure_id}."
            )

        important = _normalize_bool(parsed.get("important"))
        confidence = _normalize_float(parsed.get("confidence"))
        category = str(parsed.get("category") or "other").strip().lower() or "other"
        reason = str(parsed.get("reason") or "").strip() or "no reason provided"

        if important is None or confidence is None:
            raise RuntimeError(
                "Google importance classifier returned invalid decision fields "
                f"for {source_file} p.{source_page} {source_figure_id}."
            )

        decision = ImageImportanceDecision(
            important=important,
            confidence=confidence,
            category=category,
            reason=reason,
            cache_key=cache_key,
            source_file=source_file,
            source_page=source_page,
            source_figure_id=source_figure_id,
            error=None,
        )
        decisions.append(decision)

        cache_payload[cache_key] = {
            "important": decision.important,
            "confidence": decision.confidence,
            "category": decision.category,
            "reason": decision.reason,
        }
        cache_changed = True

    if cache_changed:
        _save_cache(cache_file, cache_payload)

    return decisions


def decisions_as_rows(decisions: list[ImageImportanceDecision]) -> list[dict[str, Any]]:
    return [asdict(decision) for decision in decisions]

