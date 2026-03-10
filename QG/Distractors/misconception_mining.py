from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pypdf import PdfReader

MISCONCEPTION_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "User" / "Misconceptions"
GLOBAL_BANK_FILENAME = "global_bank_v1.json"
MODULE_INDEX_FILENAME = "module_index.json"
MINING_DEBUG_FILENAME = "mining_debug.json"
MINING_PROMPT_VERSION = "v7"

MIN_RAW_RECORDS = 6
MIN_SELECTED_RECORDS = 3
MAX_RAW_RECORDS = 48
MAX_SELECTED_RECORDS = 30
CLUSTER_SIMILARITY_THRESHOLD = 0.86
MAX_CITATION_QUOTE_CHARS = 280
BROAD_TOPIC_TOKENS = {
    "factor",
    "factors",
    "market",
    "markets",
    "economics",
    "microeconomics",
    "module",
    "unit",
    "chapter",
    "concept",
    "concepts",
}

_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "an",
    "and",
    "another",
    "because",
    "before",
    "between",
    "being",
    "below",
    "both",
    "could",
    "course",
    "class",
    "details",
    "during",
    "each",
    "for",
    "from",
    "further",
    "have",
    "having",
    "in",
    "into",
    "module",
    "more",
    "most",
    "of",
    "on",
    "other",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "these",
    "those",
    "through",
    "to",
    "the",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "this",
    "your",
    "students",
    "student",
}


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned[:120] if cleaned else fallback


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_source_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n+ *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_verification_text(value: str) -> str:
    normalized = _normalize_source_text(value)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize_verification_text(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", _normalize_verification_text(value).lower())


def _split_source_excerpt_units(source_text: str) -> list[str]:
    normalized_source = _normalize_source_text(source_text)
    if not normalized_source:
        return []

    blocks = re.split(r"\n{2,}", normalized_source)
    units: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        line_candidates = [line.strip() for line in block.splitlines() if line.strip()]
        if not line_candidates:
            line_candidates = [block]

        for line in line_candidates:
            sentence_candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", line)
            for candidate in sentence_candidates:
                candidate = _normalize_verification_text(candidate)
                if len(candidate) < 35 or len(candidate) > MAX_CITATION_QUOTE_CHARS:
                    continue
                if not re.search(r"[.!?]$", candidate):
                    continue
                lowered = candidate.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                units.append(candidate)

    return units


def _status(status_callback: Callable[[str], None] | None, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def _derive_misconception_label(misconception: str, topic: str) -> str:
    text = _normalize_text(misconception)
    if not text:
        return ""

    first_clause = re.split(r"[.;:]", text, maxsplit=1)[0].strip()
    if first_clause.lower().startswith("students"):
        first_clause = re.sub(
            r"^students\s+(often|may|might|frequently|commonly)?\s*",
            "",
            first_clause,
            flags=re.IGNORECASE,
        ).strip()
    label = first_clause or text
    label = re.sub(r"\s+", " ", label).strip(" -")

    if len(label) > 120:
        label = label[:117].rstrip() + "..."

    if label.lower() == _normalize_text(topic).lower():
        return ""
    return label


def _is_specific_misconception(
    *,
    topic: str,
    misconception: str,
    correct_idea: str,
    tags: list[str],
) -> bool:
    topic_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", topic)
        if token.lower() not in BROAD_TOPIC_TOKENS
    }
    misconception_text = misconception.lower()
    correct_text = correct_idea.lower()
    tags_set = {str(tag).strip().lower() for tag in tags if isinstance(tag, str) and str(tag).strip()}

    if len(misconception) < 28 or len(correct_idea) < 28:
        return False

    explicit_error_language = (
        "students" in misconception_text
        or "confus" in misconception_text
        or any(
            marker in misconception_text
            for marker in (
                "assume",
                "believe",
                "think",
                "treat",
                "use",
                "include",
                "ignore",
                "set",
                "predict",
                "equate",
                "apply",
                "cancel",
                "divide",
                "multiply",
                "substitute",
            )
        )
    )
    if not explicit_error_language:
        return False

    if misconception_text == _normalize_text(topic).lower():
        return False

    if len(topic_tokens) > 0:
        hits = sum(1 for token in topic_tokens if token in misconception_text or token in correct_text)
        if hits < 1 and not topic_tokens.intersection(tags_set):
            return False

    if len(tags_set) < 1:
        return False

    return True


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_timestamp(timestamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp)
    except Exception:  # noqa: BLE001
        return None


def _is_cache_fresh(generated_at: str, ttl_hours: int) -> bool:
    generated_dt = _parse_timestamp(generated_at)
    if generated_dt is None:
        return False
    if generated_dt.tzinfo is None:
        generated_dt = generated_dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    now_dt = datetime.now().astimezone()
    return generated_dt + timedelta(hours=max(1, ttl_hours)) >= now_dt


def _domain_from_course(course_name: str) -> str:
    lowered = course_name.lower()
    if any(term in lowered for term in ("chem", "biology", "physics", "science")):
        return "science"
    if any(term in lowered for term in ("econ", "economics", "market", "finance")):
        return "economics"
    if any(term in lowered for term in ("history", "war", "government", "politics")):
        return "history"
    if any(term in lowered for term in ("math", "calculus", "trigonometry", "algebra", "geometry")):
        return "mathematics"
    return "general_academics"


def _topic_terms(module_name: str, source_chunks: list[dict[str, Any]]) -> list[str]:
    counter: Counter[str] = Counter()
    seed = f"{module_name} "
    text_blob = seed + " ".join(str(chunk.get("text") or "")[:900] for chunk in source_chunks[:12])
    for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text_blob.lower()):
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        counter[token] += 1

    return [token for token, _ in counter.most_common(14)]


def _build_module_fingerprint(
    *,
    course_id: int | None,
    module_id: int | None,
    course_name: str,
    module_name: str,
    source_files: list[str],
    source_chunks: list[dict[str, Any]],
) -> str:
    basis = {
        "course_id": course_id,
        "module_id": module_id,
        "course_name": course_name,
        "module_name": module_name,
        "source_files": sorted(source_files),
        "text_hash": hashlib.sha256(
            "\n".join(_normalize_text(str(chunk.get("text") or ""))[:500] for chunk in source_chunks[:30]).encode(
                "utf-8", "ignore"
            )
        ).hexdigest()[:24],
        "prompt_version": MINING_PROMPT_VERSION,
    }
    encoded = json.dumps(basis, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8", "ignore")).hexdigest()


def _source_quality_score(url: str, title: str) -> float:
    score = 0.0
    lowered_url = url.lower()
    lowered_title = title.lower()

    if lowered_url.startswith("local://"):
        score += 2.0

    host = urlparse(url).netloc.lower()
    if host.endswith(".edu"):
        score += 4.0

    if any(term in host for term in ("springer", "sciencedirect", "wiley", "jstor", "sagepub", "nature")):
        score += 3.0

    if any(term in host for term in ("collegeboard", "ibo.org", "cambridge", "pearson")):
        score += 3.0

    if any(term in host for term in ("nsta.org", "nctm.org", "aps.org", "aft.org", "nea.org")):
        score += 2.0

    if any(term in lowered_title for term in ("concept inventory", "examiner report", "research", "misconception")):
        score += 1.5

    if lowered_url.startswith("https://"):
        score += 0.4

    return round(score, 3)


def _extract_html_text(html: str) -> tuple[str, str]:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )
    extracted_text = _normalize_verification_text(extracted or "")

    soup = BeautifulSoup(html, "html.parser")
    title = _normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    if not extracted_text:
        extracted_text = _normalize_verification_text(soup.get_text(" ", strip=True))

    return extracted_text, title


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:  # noqa: BLE001
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            pages.append(text)
    return _normalize_verification_text("\n".join(pages))


def _fetch_source_snapshot(
    source_url: str,
    *,
    source_cache: dict[str, dict[str, str] | None],
) -> dict[str, str] | None:
    cached = source_cache.get(source_url)
    if cached is not None or source_url in source_cache:
        return cached

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(source_url, headers=headers, timeout=20, allow_redirects=True)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        source_cache[source_url] = None
        return None

    final_url = _normalize_text(response.url or source_url)
    content_type = (response.headers.get("content-type") or "").lower()
    title = ""
    text = ""

    if "pdf" in content_type or final_url.lower().endswith(".pdf"):
        text = _extract_pdf_text(response.content)
    else:
        response.encoding = response.encoding or response.apparent_encoding or "utf-8"
        html = response.text
        text, title = _extract_html_text(html)

    if not text:
        source_cache[source_url] = None
        return None

    snapshot = {
        "final_url": final_url,
        "title": title or (urlparse(final_url).netloc or "Source"),
        "text": text,
    }
    source_cache[source_url] = snapshot
    return snapshot


def _best_matching_source_excerpt(
    candidate_quote: str,
    *,
    source_text: str,
) -> str | None:
    normalized_quote = _normalize_verification_text(candidate_quote)
    units = _split_source_excerpt_units(source_text)
    if not normalized_quote or not units:
        return None

    exact_matches = [unit for unit in units if normalized_quote in unit]
    if exact_matches:
        exact_matches.sort(key=len)
        return exact_matches[0][:MAX_CITATION_QUOTE_CHARS].strip()

    quote_tokens = [token for token in _tokenize_verification_text(normalized_quote) if len(token) >= 3]
    if len(quote_tokens) < 4:
        return None

    best_sentence = ""
    best_score = 0.0
    quote_token_set = set(quote_tokens)
    for sentence in units:
        sentence_tokens = {token for token in _tokenize_verification_text(sentence) if len(token) >= 3}
        if len(sentence_tokens) < 4:
            continue
        overlap = quote_token_set.intersection(sentence_tokens)
        if len(overlap) < 4:
            continue
        precision = len(overlap) / max(1, len(quote_token_set))
        recall = len(overlap) / max(1, len(sentence_tokens))
        score = (0.75 * precision) + (0.25 * recall)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score < 0.62 or not best_sentence:
        return None
    return best_sentence[:MAX_CITATION_QUOTE_CHARS].strip()


def _misconception_id_for_record(record: dict[str, Any]) -> str:
    id_basis = "||".join(
        [
            _normalize_text(str(record.get("domain") or "")).lower(),
            _normalize_text(str(record.get("topic") or "")).lower(),
            _normalize_text(str(record.get("misconception") or "")).lower(),
            _normalize_text(str(record.get("correct_idea") or "")).lower(),
        ]
    )
    return f"m_{hashlib.sha256(id_basis.encode('utf-8', 'ignore')).hexdigest()[:16]}"


def _normalize_grounding_indices(raw_indices: Any) -> list[int]:
    if not isinstance(raw_indices, list):
        return []
    normalized: list[int] = []
    for value in raw_indices:
        try:
            index = int(value)
        except Exception:  # noqa: BLE001
            continue
        if index < 0 or index in normalized:
            continue
        normalized.append(index)
    return normalized


def _normalize_citation(raw_citation: dict[str, Any]) -> dict[str, Any] | None:
    quote_text = _normalize_text(str(raw_citation.get("quote_text") or ""))
    source_title = _normalize_text(str(raw_citation.get("source_title") or ""))
    source_url = _normalize_text(str(raw_citation.get("source_url") or ""))
    search_query = _normalize_text(str(raw_citation.get("search_query") or ""))
    grounding_chunk_indices = _normalize_grounding_indices(raw_citation.get("grounding_chunk_indices"))

    if not quote_text or len(quote_text) < 20:
        return None
    if any(token in quote_text.lower() for token in ('"misconception"', '"instructional_usefulness"', '"source_citations"')):
        return None
    if not source_url.startswith(("http://", "https://", "local://")):
        return None
    if not source_title:
        source_title = urlparse(source_url).netloc or "Source"

    return {
        "quote_text": quote_text,
        "source_title": source_title,
        "source_url": source_url,
        "search_query": search_query,
        "grounding_chunk_indices": grounding_chunk_indices,
    }


def _record_has_grounded_citations(record: dict[str, Any]) -> bool:
    raw_citations = record.get("evidence_citations")
    if not isinstance(raw_citations, list) or not raw_citations:
        return False
    return any(isinstance(citation, dict) and _normalize_citation(citation) is not None for citation in raw_citations)


def _verify_and_enrich_citation(
    raw_citation: dict[str, Any],
    *,
    source_cache: dict[str, dict[str, str] | None],
) -> dict[str, Any] | None:
    citation = _normalize_citation(raw_citation)
    if citation is None:
        return None

    if citation["source_url"].startswith("local://"):
        local_source_text = _normalize_verification_text(str(raw_citation.get("source_text") or ""))
        if not local_source_text:
            return None
        verified_quote = _best_matching_source_excerpt(
            citation["quote_text"],
            source_text=local_source_text,
        )
        if verified_quote is None:
            return None
        citation["quote_text"] = verified_quote
        return citation

    snapshot = _fetch_source_snapshot(citation["source_url"], source_cache=source_cache)
    if snapshot is None:
        return None

    verified_quote = _best_matching_source_excerpt(
        citation["quote_text"],
        source_text=snapshot["text"],
    )
    if verified_quote is None:
        return None

    citation["quote_text"] = verified_quote
    citation["source_url"] = snapshot["final_url"]
    citation["source_title"] = snapshot["title"] or citation["source_title"]
    return citation


def _extract_grounded_evidence_pool(
    search_response: Any,
    *,
    fallback_search_query: str,
) -> dict[str, Any]:
    candidates = getattr(search_response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return {"search_queries": [], "evidence_pool": []}

    grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
    if grounding_metadata is None:
        return {"search_queries": [], "evidence_pool": []}

    raw_web_queries = getattr(grounding_metadata, "web_search_queries", None)
    raw_retrieval_queries = getattr(grounding_metadata, "retrieval_queries", None)
    search_queries: list[str] = []
    for query_group in (raw_web_queries, raw_retrieval_queries):
        if not isinstance(query_group, list):
            continue
        for value in query_group:
            query = _normalize_text(str(value or ""))
            if not query or query in search_queries:
                continue
            search_queries.append(query)

    query_label = " | ".join(search_queries[:3]) if search_queries else _normalize_text(fallback_search_query)

    grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None)
    chunks = grounding_chunks if isinstance(grounding_chunks, list) else []
    grounding_supports = getattr(grounding_metadata, "grounding_supports", None)
    supports = grounding_supports if isinstance(grounding_supports, list) else []

    evidence_pool: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, tuple[int, ...]]] = set()

    for support in supports:
        segment = getattr(support, "segment", None)
        quote_text = _normalize_text(str(getattr(segment, "text", "") or ""))
        if not quote_text:
            continue

        grounding_chunk_indices = _normalize_grounding_indices(getattr(support, "grounding_chunk_indices", None))
        valid_indices = [index for index in grounding_chunk_indices if index < len(chunks)]
        if not valid_indices:
            continue

        for chunk_index in valid_indices:
            chunk = chunks[chunk_index]
            web = getattr(chunk, "web", None)
            source_title = _normalize_text(str(getattr(web, "title", "") or ""))
            source_url = _normalize_text(str(getattr(web, "uri", "") or ""))
            if not source_url.startswith(("http://", "https://")):
                continue
            if not source_title:
                source_title = urlparse(source_url).netloc or "Source"

            key = (quote_text.lower(), source_url.lower(), tuple(valid_indices))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            evidence_pool.append(
                {
                    "evidence_id": len(evidence_pool) + 1,
                    "quote_text": quote_text,
                    "source_title": source_title,
                    "source_url": source_url,
                    "search_query": query_label,
                    "grounding_chunk_indices": valid_indices,
                }
            )

    return {
        "search_queries": search_queries,
        "evidence_pool": evidence_pool,
        "citation_source_mode": "grounding_metadata",
    }


def _request_quote_backed_evidence_pool(
    *,
    client: genai.Client,
    model: str,
    course_name: str,
    module_name: str,
    domain_guess: str,
    exam_profile: str,
    topic_terms: list[str],
    search_queries_hint: list[str] | None = None,
    source_cache: dict[str, dict[str, str] | None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    topic_hint = ", ".join(topic_terms[:10])
    hinted_queries = [
        _normalize_text(str(query))
        for query in (search_queries_hint or [])
        if isinstance(query, str) and _normalize_text(str(query))
    ]
    prompt = (
        "Use Google Search to build a grounded evidence pool of short text excerpts about student misconceptions.\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "search_queries": ["..."],\n'
        '  "evidence_pool": [\n'
        "    {\n"
        '      "quote_text": "...",\n'
        '      "source_title": "...",\n'
        '      "source_url": "https://...",\n'
        '      "search_query": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Each quote_text must be a short direct excerpt or very close quote from the source text.\n"
        "- Use fully-qualified http/https source URLs.\n"
        "- Prefer .edu sites, journals, examiner reports, and teacher associations.\n"
        f"- Return between {max(12, MIN_RAW_RECORDS * 2)} and {max(24, MAX_RAW_RECORDS)} evidence rows.\n\n"
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Domain hint: {domain_guess}\n"
        f"Exam profile: {exam_profile}\n"
        f"Topic hints: {topic_hint}\n"
        + (
            "Suggested search queries from the prior search pass:\n"
            + "\n".join(f"- {query}" for query in hinted_queries[:8])
            + "\n"
            if hinted_queries
            else ""
        )
    )

    try:
        _status(status_callback, "Requesting quote-backed evidence snippets for misconception citations...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Misconception evidence-pool request failed: {exc}") from exc

    response_text = response.text if isinstance(response.text, str) else ""
    payload = _extract_json_object(response_text)
    if not isinstance(payload, dict):
        grounded_payload = _extract_grounded_evidence_pool(
            response,
            fallback_search_query=hinted_queries[0] if hinted_queries else f"{module_name} misconception",
        )
        grounded_pool = grounded_payload.get("evidence_pool") if isinstance(grounded_payload, dict) else None
        if isinstance(grounded_pool, list) and grounded_pool:
            _status(
                status_callback,
                "Quote-backed evidence request was not strict JSON; using grounded search metadata instead.",
            )
            payload = {
                "search_queries": grounded_payload.get("search_queries") if isinstance(grounded_payload, dict) else [],
                "evidence_pool": grounded_pool,
            }
        else:
            raise RuntimeError("Misconception evidence-pool request did not return parseable JSON.")

    raw_queries = payload.get("search_queries")
    search_queries = [
        _normalize_text(str(query))
        for query in raw_queries
        if isinstance(query, str) and _normalize_text(query)
    ] if isinstance(raw_queries, list) else []
    if not search_queries:
        search_queries = [f"{module_name} misconception"]

    raw_pool = payload.get("evidence_pool")
    if not isinstance(raw_pool, list):
        raise RuntimeError("Misconception evidence-pool response did not include a valid evidence_pool list.")

    active_source_cache = source_cache if source_cache is not None else {}
    normalized_pool: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in raw_pool:
        if not isinstance(row, dict):
            continue
        grounding_indices = _normalize_grounding_indices(row.get("grounding_chunk_indices"))
        citation = _verify_and_enrich_citation(
            {
                "quote_text": row.get("quote_text"),
                "source_title": row.get("source_title"),
                "source_url": row.get("source_url"),
                "search_query": row.get("search_query") or search_queries[0],
                "grounding_chunk_indices": grounding_indices,
            },
            source_cache=active_source_cache,
        )
        if citation is None:
            continue
        citation_key = (
            citation["quote_text"].lower(),
            citation["source_url"].lower(),
        )
        if citation_key in seen_keys:
            continue
        seen_keys.add(citation_key)
        normalized_pool.append(
            {
                "evidence_id": len(normalized_pool) + 1,
                **citation,
            }
        )

    if not normalized_pool:
        grounded_payload = _extract_grounded_evidence_pool(
            response,
            fallback_search_query=hinted_queries[0] if hinted_queries else f"{module_name} misconception",
        )
        grounded_pool = grounded_payload.get("evidence_pool") if isinstance(grounded_payload, dict) else None
        if isinstance(grounded_pool, list) and grounded_pool:
            _status(
                status_callback,
                "Verified quote-backed evidence was empty; falling back to grounded search metadata snippets.",
            )
            seen_keys = set()
            for row in grounded_pool:
                if not isinstance(row, dict):
                    continue
                citation = _verify_and_enrich_citation(row, source_cache=active_source_cache)
                if citation is None:
                    continue
                citation_key = (
                    citation["quote_text"].lower(),
                    citation["source_url"].lower(),
                )
                if citation_key in seen_keys:
                    continue
                seen_keys.add(citation_key)
                normalized_pool.append(
                    {
                        "evidence_id": len(normalized_pool) + 1,
                        **citation,
                    }
                )

    if not normalized_pool:
        raise RuntimeError("Misconception evidence-pool request returned no valid quote-backed evidence snippets.")

    return {
        "search_queries": search_queries,
        "evidence_pool": normalized_pool,
        "citation_source_mode": "search_quotes",
    }


def _build_mining_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "search_queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
            },
            "misconceptions": {
                "type": "array",
                "minItems": MIN_RAW_RECORDS,
                "maxItems": MAX_RAW_RECORDS,
                "items": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "topic": {"type": "string"},
                        "misconception": {"type": "string"},
                        "correct_idea": {"type": "string"},
                        "evidence_source": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["title", "url"],
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": [
                        "domain",
                        "topic",
                        "misconception",
                        "correct_idea",
                        "evidence_source",
                        "tags",
                    ],
                },
            },
        },
        "required": ["search_queries", "misconceptions"],
    }


def _build_mining_prompt(
    *,
    course_name: str,
    module_name: str,
    domain_guess: str,
    exam_profile: str,
    topic_terms: list[str],
) -> str:
    topic_hint = ", ".join(topic_terms[:10])
    return (
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Domain guess: {domain_guess}\n"
        f"Exam profile: {exam_profile}\n"
        f"Module topic hints: {topic_hint}\n\n"
        "Use Google Search to mine student misconceptions for this module.\n"
        "Generate targeted search queries such as:\n"
        '"<topic> misconception", "<topic> common student errors", "<topic> concept inventory", '
        '"<exam> common mistakes", "<topic> distractor misconceptions".\n\n'
        "Prefer high-quality sources: .edu sites, journals, teacher associations, official examiner reports.\n"
        "Return only JSON. Each misconception must be concise, specific, and instructionally useful.\n"
        "Do not fabricate URLs.\n"
    )


def _build_local_evidence_pool(
    *,
    course_name: str,
    module_name: str,
    source_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    course_slug = _slugify(course_name, "course")
    module_slug = _slugify(module_name, "module")
    evidence_pool: list[dict[str, Any]] = []

    for chunk in source_chunks:
        if not isinstance(chunk, dict):
            continue
        source_file = _normalize_text(str(chunk.get("source_file") or ""))
        source_page_raw = chunk.get("source_page")
        try:
            source_page = int(source_page_raw)
        except Exception:  # noqa: BLE001
            source_page = 1
        source_page = max(1, source_page)
        source_text = _normalize_source_text(str(chunk.get("text") or ""))
        if not source_file or len(source_text) < 40:
            continue

        evidence_pool.append(
            {
                "evidence_id": len(evidence_pool) + 1,
                "quote_text": source_text[:MAX_CITATION_QUOTE_CHARS],
                "source_title": f"{source_file} p.{source_page}",
                "source_url": (
                    f"local://{course_slug}/{module_slug}/{_slugify(source_file, 'source')}"
                    f"?page={source_page}"
                ),
                "search_query": f"local module source: {module_name}",
                "grounding_chunk_indices": [len(evidence_pool)],
                "source_text": source_text,
            }
        )

    return evidence_pool


def _merge_misconception_records(
    primary_records: list[dict[str, Any]],
    supplemental_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in [*primary_records, *supplemental_records]:
        if not isinstance(record, dict):
            continue
        key_basis = "||".join(
            [
                _normalize_text(str(record.get("topic") or "")).lower(),
                _normalize_text(str(record.get("misconception") or "")).lower(),
                _normalize_text(str(record.get("correct_idea") or "")).lower(),
            ]
        )
        key = hashlib.sha256(key_basis.encode("utf-8", "ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)

    return merged


def mine_misconceptions_from_local_sources(
    *,
    api_key: str,
    course_name: str,
    module_name: str,
    domain_guess: str,
    exam_profile: str,
    topic_terms: list[str],
    source_chunks: list[dict[str, Any]],
    model: str = "gemini-2.5-flash",
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not api_key.strip():
        raise RuntimeError("Missing GOOGLE_API_KEY for local misconception fallback.")

    evidence_pool = _build_local_evidence_pool(
        course_name=course_name,
        module_name=module_name,
        source_chunks=source_chunks,
    )
    if not evidence_pool:
        raise RuntimeError("No local source chunks were available for misconception fallback.")

    client = genai.Client(api_key=api_key)
    topic_hint = ", ".join(topic_terms[:10])
    extraction_prompt = (
        "Infer likely student misconceptions directly from the cited local module excerpts.\n"
        "Return valid JSON only with this shape:\n"
        "{\n"
        '  "misconceptions": [\n'
        "    {\n"
        '      "domain": "...",\n'
        '      "topic": "...",\n'
        '      "misconception": "...",\n'
        '      "correct_idea": "...",\n'
        '      "evidence_refs": [1],\n'
        '      "tags": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Every misconception must cite one or more evidence_refs from the provided local source pool.\n"
        "- Focus on concrete student confusions or wrong procedures, not broad topic labels.\n"
        "- Make the misconception and correct_idea specific enough to support diagnostic distractors.\n"
        f"- Return between {MIN_SELECTED_RECORDS} and 12 misconceptions.\n\n"
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Domain hint: {domain_guess}\n"
        f"Exam profile: {exam_profile}\n"
        f"Topic hints: {topic_hint}\n\n"
        "Local evidence pool (JSON):\n"
        f"{json.dumps(evidence_pool[:48], indent=2)}"
    )

    try:
        _status(status_callback, "Supplementing misconception bank from local module sources...")
        response = client.models.generate_content(
            model=model,
            contents=extraction_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Local misconception fallback request failed: {exc}") from exc

    response_text = response.text if isinstance(response.text, str) else ""
    payload = _extract_json_object(response_text)
    if not isinstance(payload, dict):
        raise RuntimeError("Local misconception fallback did not return parseable JSON.")

    raw_rows = payload.get("misconceptions")
    if not isinstance(raw_rows, list):
        raise RuntimeError("Local misconception fallback did not include a valid misconceptions list.")

    evidence_by_id = {
        int(row["evidence_id"]): row
        for row in evidence_pool
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), int)
    }
    source_cache: dict[str, dict[str, str] | None] = {}
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        record = _normalize_mined_record(
            row,
            evidence_by_id=evidence_by_id,
            source_cache=source_cache,
        )
        if record is not None:
            normalized.append(record)

    if len(normalized) < MIN_SELECTED_RECORDS:
        raise RuntimeError(
            "Local misconception fallback produced too few usable records "
            f"({len(normalized)}). Need at least {MIN_SELECTED_RECORDS}."
        )

    _status(status_callback, f"Collected {len(normalized)} local misconception records.")
    return {
        "raw_records": normalized[:12],
        "search_queries": [f"local module source: {module_name}"],
        "evidence_pool": evidence_pool,
        "citation_source_mode": "local_source_chunks",
    }


def _normalize_mined_record(
    raw_record: dict[str, Any],
    *,
    evidence_by_id: dict[int, dict[str, Any]],
    source_cache: dict[str, dict[str, str] | None],
) -> dict[str, Any] | None:
    domain = _normalize_text(str(raw_record.get("domain") or ""))
    topic = _normalize_text(str(raw_record.get("topic") or ""))
    misconception = _normalize_text(str(raw_record.get("misconception") or ""))
    correct_idea = _normalize_text(str(raw_record.get("correct_idea") or ""))

    tags_raw = raw_record.get("tags")
    tags_list = [
        _normalize_text(str(tag)).lower()
        for tag in tags_raw
        if isinstance(tag, str) and _normalize_text(tag)
    ] if isinstance(tags_raw, list) else []
    if not tags_list and topic:
        topic_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z\\-]{2,}", topic)
        ]
        tags_list = topic_tokens[:3]

    if not domain or not topic or not misconception or not correct_idea:
        return None
    if len(tags_list) < 1:
        return None

    evidence_refs_raw = raw_record.get("evidence_refs")
    evidence_refs = _normalize_grounding_indices(evidence_refs_raw)
    evidence_citations: list[dict[str, Any]] = []
    evidence_sources: list[dict[str, str]] = []
    source_seen: set[tuple[str, str]] = set()
    citation_seen: set[tuple[str, str, tuple[int, ...]]] = set()

    for evidence_ref in evidence_refs:
        evidence_row = evidence_by_id.get(evidence_ref)
        if not isinstance(evidence_row, dict):
            continue
        citation = _verify_and_enrich_citation(evidence_row, source_cache=source_cache)
        if citation is None:
            continue
        citation_key = (
            citation["quote_text"].lower(),
            citation["source_url"].lower(),
            tuple(citation["grounding_chunk_indices"]),
        )
        if citation_key in citation_seen:
            continue
        citation_seen.add(citation_key)
        evidence_citations.append(citation)

        source_pair = (citation["source_title"], citation["source_url"])
        if source_pair not in source_seen:
            source_seen.add(source_pair)
            evidence_sources.append({"title": citation["source_title"], "url": citation["source_url"]})

    if not evidence_citations:
        return None

    misconception_label = _derive_misconception_label(misconception, topic)
    if not misconception_label:
        return None
    if not _is_specific_misconception(
        topic=topic,
        misconception=misconception,
        correct_idea=correct_idea,
        tags=tags_list,
    ):
        return None

    quality_score = max(
        _source_quality_score(source["url"], source["title"])
        for source in evidence_sources
    )
    return {
        "domain": domain,
        "topic": topic,
        "misconception_label": misconception_label,
        "misconception": misconception,
        "correct_idea": correct_idea,
        "tags": sorted(set(tags_list)),
        "evidence_sources": evidence_sources,
        "evidence_citations": evidence_citations,
        "quality_score": quality_score,
        "created_at": datetime.now().astimezone().isoformat(),
    }


def _embed_texts(
    texts: list[str],
    *,
    api_key: str,
    model: str,
) -> list[list[float]]:
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.embed_content(model=model, contents=texts)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

    embeddings = response.embeddings if hasattr(response, "embeddings") else None
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Gemini embedding response did not return the expected embedding count.")

    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if not isinstance(values, list) or not values:
            raise RuntimeError("Gemini embedding response included an empty vector.")
        vectors.append([float(value) for value in values])

    return vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def mine_misconceptions_with_google_search(
    *,
    api_key: str,
    course_name: str,
    module_name: str,
    domain_guess: str,
    exam_profile: str,
    topic_terms: list[str],
    model: str = "gemini-2.5-flash",
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not api_key.strip():
        raise RuntimeError("Missing GOOGLE_API_KEY for misconception mining.")

    client = genai.Client(api_key=api_key)
    source_cache: dict[str, dict[str, str] | None] = {}
    search_prompt = _build_mining_prompt(
        course_name=course_name,
        module_name=module_name,
        domain_guess=domain_guess,
        exam_profile=exam_profile,
        topic_terms=topic_terms,
    )

    try:
        _status(status_callback, "Searching the internet for misconception evidence...")
        search_response = client.models.generate_content(
            model=model,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Misconception mining search request failed: {exc}") from exc

    search_text = search_response.text if isinstance(search_response.text, str) else ""
    grounded_search = _extract_grounded_evidence_pool(
        search_response,
        fallback_search_query=f"{module_name} misconception",
    )
    evidence_payload = _request_quote_backed_evidence_pool(
        client=client,
        model=model,
        course_name=course_name,
        module_name=module_name,
        domain_guess=domain_guess,
        exam_profile=exam_profile,
        topic_terms=topic_terms,
        search_queries_hint=grounded_search.get("search_queries") if isinstance(grounded_search, dict) else None,
        source_cache=source_cache,
        status_callback=status_callback,
    )
    evidence_pool = evidence_payload["evidence_pool"] if isinstance(evidence_payload, dict) else []
    if not isinstance(evidence_pool, list) or not evidence_pool:
        raise RuntimeError("Misconception mining search did not return usable grounded evidence.")
    evidence_by_id = {
        int(row["evidence_id"]): row
        for row in evidence_pool
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), int)
    }
    if not evidence_by_id:
        raise RuntimeError("Misconception mining search did not return usable grounded evidence snippets.")

    extraction_prompt = (
        "Convert the following web-mined notes into strict JSON.\n"
        "Required JSON shape:\n"
        "{\n"
        '  "search_queries": ["..."],\n'
        '  "misconceptions": [\n'
        "    {\n"
        '      "domain": "...",\n'
        '      "topic": "...",\n'
        '      "misconception": "...",\n'
        '      "correct_idea": "...",\n'
        '      "evidence_refs": [1],\n'
        '      "tags": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Output valid JSON only.\n"
        "- Every misconception must cite one or more evidence_refs from the grounded evidence pool below.\n"
        "- Do not invent evidence_refs or sources.\n"
        "- Only keep misconceptions that are directly supported by the quoted evidence.\n"
        f"- Return between {MIN_RAW_RECORDS} and {MAX_RAW_RECORDS} misconceptions.\n\n"
        f"Course: {course_name}\n"
        f"Module: {module_name}\n"
        f"Domain hint: {domain_guess}\n"
        f"Exam profile: {exam_profile}\n\n"
        "Grounded evidence pool (JSON):\n"
        f"{json.dumps(evidence_pool[:96], indent=2)}\n\n"
        "Web-mined notes:\n"
        f"{search_text[:90_000]}"
    )

    try:
        _status(status_callback, "Extracting structured misconceptions from search notes...")
        extraction_response = client.models.generate_content(
            model=model,
            contents=extraction_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Misconception JSON extraction request failed: {exc}") from exc

    extraction_text = extraction_response.text if isinstance(extraction_response.text, str) else ""
    payload = _extract_json_object(extraction_text)
    if not isinstance(payload, dict):
        raise RuntimeError("Misconception mining did not return parseable JSON.")

    raw_rows = payload.get("misconceptions")
    if not isinstance(raw_rows, list):
        raise RuntimeError("Misconception mining response did not include a valid misconceptions list.")

    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        record = _normalize_mined_record(
            row,
            evidence_by_id=evidence_by_id,
            source_cache=source_cache,
        )
        if record is not None:
            normalized.append(record)

    if len(normalized) < MIN_RAW_RECORDS:
        _status(
            status_callback,
            f"Collected only {len(normalized)} valid web misconception records; supplementing may be required.",
        )
    else:
        _status(status_callback, f"Collected {len(normalized)} valid misconception records.")

    return {
        "raw_records": normalized[:MAX_RAW_RECORDS],
        "search_queries": evidence_payload.get("search_queries")
        if isinstance(evidence_payload, dict)
        else grounded_search.get("search_queries") if isinstance(grounded_search, dict) else [],
        "evidence_pool": evidence_pool,
        "citation_source_mode": evidence_payload.get("citation_source_mode", "unknown")
        if isinstance(evidence_payload, dict)
        else "unknown",
        "warning": (
            f"Web misconception mining produced only {len(normalized)} valid records."
            if len(normalized) < MIN_RAW_RECORDS
            else None
        ),
    }


def cluster_misconceptions_with_embeddings(
    records: list[dict[str, Any]],
    *,
    api_key: str,
    model: str = "gemini-embedding-001",
    similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    status_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        raise RuntimeError("No misconception records provided for clustering.")

    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for record in records:
        key_basis = "||".join(
            [
                _normalize_text(str(record.get("topic") or "")).lower(),
                _normalize_text(str(record.get("misconception") or "")).lower(),
                _normalize_text(str(record.get("correct_idea") or "")).lower(),
            ]
        )
        key = hashlib.sha256(key_basis.encode("utf-8", "ignore")).hexdigest()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(record)

    if not deduped:
        raise RuntimeError("All mined misconception records were duplicates.")

    embedding_texts = [
        (
            f"Domain: {record.get('domain')}\n"
            f"Topic: {record.get('topic')}\n"
            f"Misconception: {record.get('misconception')}\n"
            f"Correct idea: {record.get('correct_idea')}\n"
            f"Tags: {', '.join(record.get('tags') or [])}"
        )
        for record in deduped
    ]

    _status(status_callback, f"Embedding {len(embedding_texts)} misconception records for clustering...")
    vectors = _embed_texts(embedding_texts, api_key=api_key, model=model)

    clusters: list[list[int]] = []
    for index, vector in enumerate(vectors):
        assigned = False
        for cluster in clusters:
            representative = vectors[cluster[0]]
            similarity = _cosine_similarity(vector, representative)
            if similarity >= similarity_threshold:
                cluster.append(index)
                assigned = True
                break
        if not assigned:
            clusters.append([index])

    canonical_records: list[dict[str, Any]] = []
    for cluster in clusters:
        ranked = sorted(
            cluster,
            key=lambda idx: (
                float(deduped[idx].get("quality_score") or 0.0),
                len(str(deduped[idx].get("misconception") or "")),
            ),
            reverse=True,
        )
        canonical_index = ranked[0]
        canonical = dict(deduped[canonical_index])

        merged_tags: set[str] = set()
        merged_sources: list[dict[str, str]] = []
        merged_citations: list[dict[str, Any]] = []
        source_seen: set[tuple[str, str]] = set()
        citation_seen: set[tuple[str, str, tuple[int, ...]]] = set()
        for idx in cluster:
            current = deduped[idx]
            for tag in current.get("tags") or []:
                if isinstance(tag, str) and tag.strip():
                    merged_tags.add(tag.strip().lower())
            for source in current.get("evidence_sources") or []:
                if not isinstance(source, dict):
                    continue
                title = _normalize_text(str(source.get("title") or ""))
                url = _normalize_text(str(source.get("url") or ""))
                if not title or not url:
                    continue
                pair = (title, url)
                if pair in source_seen:
                    continue
                source_seen.add(pair)
                merged_sources.append({"title": title, "url": url})
            for citation in current.get("evidence_citations") or []:
                if not isinstance(citation, dict):
                    continue
                normalized_citation = _normalize_citation(citation)
                if normalized_citation is None:
                    continue
                citation_key = (
                    normalized_citation["quote_text"].lower(),
                    normalized_citation["source_url"].lower(),
                    tuple(normalized_citation["grounding_chunk_indices"]),
                )
                if citation_key in citation_seen:
                    continue
                citation_seen.add(citation_key)
                merged_citations.append(normalized_citation)

        misconception_id = _misconception_id_for_record(canonical)

        canonical["misconception_id"] = misconception_id
        canonical["tags"] = sorted(merged_tags)
        canonical["evidence_sources"] = merged_sources[:8]
        canonical["evidence_citations"] = merged_citations[:12]
        canonical["cluster_size"] = len(cluster)
        canonical["embedding_model"] = model
        canonical_records.append(canonical)

    if len(canonical_records) < min(MIN_RAW_RECORDS, len(deduped)):
        _status(
            status_callback,
            f"Canonical misconception count ({len(canonical_records)}) is low; supplementing from raw records.",
        )
        existing_ids = {
            str(record.get("misconception_id") or "").strip()
            for record in canonical_records
            if str(record.get("misconception_id") or "").strip()
        }
        ranked_deduped = sorted(
            deduped,
            key=lambda record: (
                float(record.get("quality_score") or 0.0),
                len(str(record.get("misconception") or "")),
            ),
            reverse=True,
        )
        for record in ranked_deduped:
            misconception_id = _misconception_id_for_record(record)
            if not misconception_id or misconception_id in existing_ids:
                continue
            supplemental = dict(record)
            supplemental["misconception_id"] = misconception_id
            supplemental["cluster_size"] = int(record.get("cluster_size") or 1)
            supplemental["embedding_model"] = model
            canonical_records.append(supplemental)
            existing_ids.add(misconception_id)
            if len(canonical_records) >= min(MIN_RAW_RECORDS, len(deduped)):
                break

    canonical_records.sort(
        key=lambda record: (
            float(record.get("quality_score") or 0.0),
            int(record.get("cluster_size") or 1),
            len(str(record.get("misconception") or "")),
        ),
        reverse=True,
    )

    _status(status_callback, f"Clustered misconceptions down to {len(canonical_records)} canonical records.")

    return canonical_records


def select_module_misconceptions(
    canonical_records: list[dict[str, Any]],
    *,
    module_name: str,
    topic_terms: list[str],
    max_records: int = MAX_SELECTED_RECORDS,
) -> list[dict[str, Any]]:
    if not canonical_records:
        return []

    topic_term_set = {term.lower() for term in topic_terms if isinstance(term, str) and term.strip()}
    module_tokens = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", module_name.lower())}

    scored_rows: list[tuple[float, dict[str, Any]]] = []
    for record in canonical_records:
        tags = {str(tag).lower() for tag in (record.get("tags") or []) if isinstance(tag, str)}
        topic_text = str(record.get("topic") or "").lower()
        misconception_text = str(record.get("misconception") or "").lower()

        score = float(record.get("quality_score") or 0.0)
        score += 0.8 * len(tags.intersection(topic_term_set))
        score += 0.5 * len(module_tokens.intersection(tags))
        if any(term in topic_text for term in topic_term_set):
            score += 1.0
        if any(term in misconception_text for term in topic_term_set):
            score += 0.8

        scored_rows.append((score, record))

    scored_rows.sort(key=lambda row: row[0], reverse=True)
    selected = [row[1] for row in scored_rows[:max(1, max_records)]]
    return selected


def build_or_load_module_misconceptions(
    *,
    api_key: str,
    course_name: str,
    module_name: str,
    course_id: int | None,
    module_id: int | None,
    source_files: list[str],
    source_chunks: list[dict[str, Any]],
    cache_ttl_hours: int = 168,
    search_model: str = "gemini-2.5-flash",
    embedding_model: str = "gemini-embedding-001",
    storage_root: str | Path = MISCONCEPTION_STORAGE_ROOT,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not api_key.strip():
        raise RuntimeError("Missing GOOGLE_API_KEY for misconception pipeline.")

    resolved_root = Path(storage_root).expanduser().resolve()
    global_bank_path = resolved_root / GLOBAL_BANK_FILENAME

    course_slug = _slugify(course_name, "course")
    module_slug = _slugify(module_name, "module")
    module_dir = resolved_root / "by_module" / course_slug / module_slug
    module_index_path = module_dir / MODULE_INDEX_FILENAME
    mining_debug_path = module_dir / MINING_DEBUG_FILENAME

    domain_guess = _domain_from_course(course_name)
    exam_profile = "AP" if course_name.strip().upper().startswith("AP ") else "standard"
    terms = _topic_terms(module_name, source_chunks)

    fingerprint = _build_module_fingerprint(
        course_id=course_id,
        module_id=module_id,
        course_name=course_name,
        module_name=module_name,
        source_files=source_files,
        source_chunks=source_chunks,
    )

    global_payload = _read_json_file(global_bank_path) or {}
    global_records_raw = global_payload.get("records")
    global_records = global_records_raw if isinstance(global_records_raw, list) else []
    global_by_id: dict[str, dict[str, Any]] = {}
    for row in global_records:
        if isinstance(row, dict):
            row_id = str(row.get("misconception_id") or "").strip()
            if row_id and _record_has_grounded_citations(row):
                global_by_id[row_id] = row

    module_index = _read_json_file(module_index_path) or {}
    cached_ids_raw = module_index.get("selected_misconception_ids")
    cached_ids = [
        str(row).strip()
        for row in cached_ids_raw
        if isinstance(row, str) and str(row).strip()
    ] if isinstance(cached_ids_raw, list) else []

    cached_selected_records = [global_by_id[misconception_id] for misconception_id in cached_ids if misconception_id in global_by_id]
    fresh_cache = _is_cache_fresh(str(module_index.get("generated_at") or ""), cache_ttl_hours)
    prompt_version_matches = str(module_index.get("prompt_version") or "") == MINING_PROMPT_VERSION
    module_info = module_index.get("module") if isinstance(module_index.get("module"), dict) else {}
    course_info = module_index.get("course") if isinstance(module_index.get("course"), dict) else {}
    module_matches = (
        (module_id is not None and int(module_info.get("id") or 0) == module_id)
        or _normalize_text(str(module_info.get("name") or "")).lower() == _normalize_text(module_name).lower()
    )
    course_matches = (
        (course_id is not None and int(course_info.get("id") or 0) == course_id)
        or _normalize_text(str(course_info.get("name") or "")).lower() == _normalize_text(course_name).lower()
    )

    cache_hit = (
        bool(module_index)
        and str(module_index.get("fingerprint") or "") == fingerprint
        and fresh_cache
        and prompt_version_matches
        and str(module_index.get("search_model") or "") == search_model
        and str(module_index.get("embedding_model") or "") == embedding_model
        and all(misconception_id in global_by_id for misconception_id in cached_ids)
    )

    if cache_hit:
        selected_cached = cached_selected_records
        _status(status_callback, f"Using cached misconception bank ({len(selected_cached)} records).")
        return {
            "cache_hit": True,
            "cache_mode": "exact",
            "storage_root": str(resolved_root),
            "global_bank_path": str(global_bank_path),
            "module_index_path": str(module_index_path),
            "mining_debug_path": str(mining_debug_path),
            "raw_count": int(module_index.get("raw_count") or len(selected_cached)),
            "canonical_count": int(module_index.get("canonical_count") or len(selected_cached)),
            "selected_count": len(selected_cached),
            "sources_used": module_index.get("sources_used") if isinstance(module_index.get("sources_used"), list) else [],
            "selected_misconceptions": selected_cached,
            "topic_terms": terms,
            "domain_guess": domain_guess,
        }

    compatible_cache_hit = (
        bool(module_index)
        and fresh_cache
        and prompt_version_matches
        and course_matches
        and module_matches
        and bool(cached_selected_records)
    )
    if compatible_cache_hit:
        _status(
            status_callback,
            "Using compatible misconception cache despite module fingerprint drift "
            f"({len(cached_selected_records)} records).",
        )
        return {
            "cache_hit": True,
            "cache_mode": "compatible",
            "storage_root": str(resolved_root),
            "global_bank_path": str(global_bank_path),
            "module_index_path": str(module_index_path),
            "mining_debug_path": str(mining_debug_path),
            "raw_count": int(module_index.get("raw_count") or len(cached_selected_records)),
            "canonical_count": int(module_index.get("canonical_count") or len(cached_selected_records)),
            "selected_count": len(cached_selected_records),
            "sources_used": module_index.get("sources_used") if isinstance(module_index.get("sources_used"), list) else [],
            "selected_misconceptions": cached_selected_records,
            "topic_terms": terms,
            "domain_guess": domain_guess,
            "warning": "Used compatible misconception cache because the module fingerprint changed.",
        }
    warnings: list[str] = []
    mining_result: dict[str, Any] | None = None
    mining_exception: Exception | None = None
    try:
        mining_result = mine_misconceptions_with_google_search(
            api_key=api_key,
            course_name=course_name,
            module_name=module_name,
            domain_guess=domain_guess,
            exam_profile=exam_profile,
            topic_terms=terms,
            model=search_model,
            status_callback=status_callback,
        )
    except Exception as exc:
        mining_exception = exc

    raw_records_raw = mining_result.get("raw_records") if isinstance(mining_result, dict) else None
    raw_records = raw_records_raw if isinstance(raw_records_raw, list) else []
    search_queries = mining_result.get("search_queries") if isinstance(mining_result, dict) else []
    evidence_pool = mining_result.get("evidence_pool") if isinstance(mining_result, dict) else []
    citation_source_mode = str(mining_result.get("citation_source_mode") or "unknown") if isinstance(mining_result, dict) else "unknown"
    mining_warning = str(mining_result.get("warning") or "").strip() if isinstance(mining_result, dict) else ""
    if mining_warning:
        warnings.append(mining_warning)

    if len(raw_records) < MIN_RAW_RECORDS:
        try:
            local_fallback = mine_misconceptions_from_local_sources(
                api_key=api_key,
                course_name=course_name,
                module_name=module_name,
                domain_guess=domain_guess,
                exam_profile=exam_profile,
                topic_terms=terms,
                source_chunks=source_chunks,
                model=search_model,
                status_callback=status_callback,
            )
            local_raw_records = local_fallback.get("raw_records") if isinstance(local_fallback, dict) else None
            raw_records = _merge_misconception_records(
                raw_records,
                local_raw_records if isinstance(local_raw_records, list) else [],
            )
            local_queries = local_fallback.get("search_queries") if isinstance(local_fallback, dict) else None
            if isinstance(local_queries, list):
                search_queries = list(search_queries) + [
                    query
                    for query in local_queries
                    if isinstance(query, str) and query not in search_queries
                ]
            local_pool = local_fallback.get("evidence_pool") if isinstance(local_fallback, dict) else None
            if isinstance(local_pool, list):
                evidence_pool = list(evidence_pool) + local_pool
            local_mode = str(local_fallback.get("citation_source_mode") or "").strip() if isinstance(local_fallback, dict) else ""
            if local_mode and local_mode not in citation_source_mode:
                citation_source_mode = ",".join(part for part in [citation_source_mode, local_mode] if part)
            warnings.append("Misconception bank was supplemented from local module sources.")
        except Exception as local_exc:
            if mining_exception is not None:
                warnings.append(f"Local misconception fallback also failed: {local_exc}")

    if not raw_records:
        if cached_selected_records:
            fallback_mode = "stale_fallback" if not fresh_cache else "fallback"
            warning_parts = [f"Fresh misconception mining failed and cache was reused: {mining_exception or 'no usable records'}"]
            warning_parts.extend(warnings)
            _status(
                status_callback,
                "Fresh misconception mining failed; using saved misconception bank instead "
                f"({len(cached_selected_records)} records).",
            )
            return {
                "cache_hit": True,
                "cache_mode": fallback_mode,
                "storage_root": str(resolved_root),
                "global_bank_path": str(global_bank_path),
                "module_index_path": str(module_index_path),
                "mining_debug_path": str(mining_debug_path),
                "raw_count": int(module_index.get("raw_count") or len(cached_selected_records)),
                "canonical_count": int(module_index.get("canonical_count") or len(cached_selected_records)),
                "selected_count": len(cached_selected_records),
                "sources_used": module_index.get("sources_used") if isinstance(module_index.get("sources_used"), list) else [],
                "selected_misconceptions": cached_selected_records,
                "topic_terms": terms,
                "domain_guess": domain_guess,
                "warning": " ".join(part for part in warning_parts if part).strip(),
            }
        if mining_exception is not None:
            raise mining_exception
        raise RuntimeError("Misconception mining did not return any raw records.")

    canonical_records = cluster_misconceptions_with_embeddings(
        raw_records,
        api_key=api_key,
        model=embedding_model,
        similarity_threshold=CLUSTER_SIMILARITY_THRESHOLD,
        status_callback=status_callback,
    )

    selected_records = select_module_misconceptions(
        canonical_records,
        module_name=module_name,
        topic_terms=terms,
        max_records=MAX_SELECTED_RECORDS,
    )

    if not selected_records:
        raise RuntimeError("Misconception pipeline found no canonical records for this module.")
    if len(selected_records) < MIN_SELECTED_RECORDS:
        raise RuntimeError(
            f"Misconception pipeline found only {len(selected_records)} usable module misconceptions; "
            f"need at least {MIN_SELECTED_RECORDS} for diagnostic distractors."
        )

    _status(status_callback, f"Selected {len(selected_records)} module-specific misconceptions.")

    for record in canonical_records:
        record_id = str(record.get("misconception_id") or "").strip()
        if record_id:
            global_by_id[record_id] = record

    updated_global_payload = {
        "version": 2,
        "updated_at": datetime.now().astimezone().isoformat(),
        "records": sorted(
            list(global_by_id.values()),
            key=lambda row: (
                float(row.get("quality_score") or 0.0),
                int(row.get("cluster_size") or 1),
                str(row.get("misconception_id") or ""),
            ),
            reverse=True,
        ),
    }
    _write_json_file(global_bank_path, updated_global_payload)

    sources_used: list[str] = []
    seen_urls: set[str] = set()
    for record in selected_records:
        for source in record.get("evidence_sources") or []:
            if not isinstance(source, dict):
                continue
            url = _normalize_text(str(source.get("url") or ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources_used.append(url)

    selected_ids = [
        str(record.get("misconception_id") or "").strip()
        for record in selected_records
        if str(record.get("misconception_id") or "").strip()
    ]

    module_index_payload = {
        "version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "fingerprint": fingerprint,
        "prompt_version": MINING_PROMPT_VERSION,
        "course": {"id": course_id, "name": course_name},
        "module": {"id": module_id, "name": module_name},
        "domain_guess": domain_guess,
        "topic_terms": terms,
        "search_model": search_model,
        "embedding_model": embedding_model,
        "cache_ttl_hours": cache_ttl_hours,
        "raw_count": len(raw_records),
        "canonical_count": len(canonical_records),
        "selected_count": len(selected_ids),
        "selected_misconception_ids": selected_ids,
        "sources_used": sources_used,
        "search_queries": search_queries if isinstance(search_queries, list) else [],
        "citation_source_mode": citation_source_mode,
        "warning": " ".join(part for part in warnings if part).strip(),
    }
    _write_json_file(module_index_path, module_index_payload)

    mining_debug_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "prompt_version": MINING_PROMPT_VERSION,
        "search_queries": search_queries if isinstance(search_queries, list) else [],
        "citation_source_mode": citation_source_mode,
        "warning": " ".join(part for part in warnings if part).strip(),
        "grounded_evidence_pool": evidence_pool if isinstance(evidence_pool, list) else [],
        "raw_records": raw_records,
        "canonical_records": canonical_records,
        "selected_misconceptions": selected_records,
    }
    _write_json_file(mining_debug_path, mining_debug_payload)

    return {
        "cache_hit": False,
        "storage_root": str(resolved_root),
        "global_bank_path": str(global_bank_path),
        "module_index_path": str(module_index_path),
        "mining_debug_path": str(mining_debug_path),
        "raw_count": len(raw_records),
        "canonical_count": len(canonical_records),
        "selected_count": len(selected_records),
        "sources_used": sources_used,
        "selected_misconceptions": selected_records,
        "topic_terms": terms,
        "domain_guess": domain_guess,
        "warning": " ".join(part for part in warnings if part).strip(),
    }
