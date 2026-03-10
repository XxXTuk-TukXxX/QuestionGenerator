#!/usr/bin/env python3
"""Canvas helpers for reading a selected class and exporting assignment data.

Environment variables accepted from .env:
- LCDS_USERNAME (preferred) or CANVAS_USERNAME or USERNAME
- LCDS_PASSWORD (preferred) or CANVAS_PASSWORD or PASSWORD
- HEADLESS=true|false (optional, default: true)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://lcds.instructure.com"
LOGIN_URL = f"{BASE_URL}/login/ldap"
COURSES_API_URL = f"{BASE_URL}/api/v1/courses"
SIMILARITY_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "your",
    "you",
    "this",
    "that",
    "into",
    "onto",
    "over",
    "under",
    "through",
    "work",
    "class",
    "homework",
    "hw",
    "in",
    "on",
    "of",
    "to",
    "a",
    "an",
    "pdf",
}


@dataclass
class CanvasCourse:
    course_id: int
    name: str


@dataclass
class CourseReadResult:
    course: CanvasCourse
    assignment_name: str | None
    assignment_url: str | None
    created_at: str | None
    due_at: str | None
    module_label: str | None
    module_name: str | None = None
    module_id: int | None = None
    module_assignment_count: int | None = None
    saved_assignments_path: str | None = None
    submitted_assignment_count: int | None = None
    saved_submitted_assignments_path: str | None = None
    submitted_module_assignment_count: int | None = None
    submitted_module_files_count: int | None = None
    saved_submitted_module_path: str | None = None
    tasked_module_item_count: int | None = None
    tasked_module_files_count: int | None = None
    saved_tasked_module_path: str | None = None


def pick_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def parse_canvas_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    return parsed.astimezone()


def format_dt(value: str | None) -> str:
    parsed = parse_canvas_datetime(value)
    if parsed is None:
        return "N/A"
    return parsed.strftime("%Y-%m-%d %H:%M %Z").strip()


def absolutize_url(url: str | None) -> str:
    if not url:
        return "N/A"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{BASE_URL}{url}"
    return url


def fetch_paginated_json_list(context: Any, url: str, *, per_page: int = 100) -> list[dict[str, Any]]:
    page = 1
    all_rows: list[dict[str, Any]] = []

    while True:
        separator = "&" if "?" in url else "?"
        page_url = f"{url}{separator}per_page={per_page}&page={page}"
        response = context.request.get(page_url, timeout=30_000)
        if not response.ok:
            raise RuntimeError(f"Failed to fetch '{page_url}' ({response.status} {response.status_text}).")

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected API response shape for '{page_url}'.")

        rows = [row for row in payload if isinstance(row, dict)]
        all_rows.extend(rows)

        if len(payload) < per_page:
            break
        page += 1

    return all_rows


def login_to_canvas(page: Any, username: str, password: str) -> None:
    try:
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)
        page.fill('input[name="pseudonym_session[unique_id]"]', username)
        page.fill('input[name="pseudonym_session[password]"]', password)
        page.click('button[type="submit"], input[type="submit"]')
        page.wait_for_url("**/dashboard", timeout=30_000)
    except PlaywrightTimeoutError:
        # Some Canvas tenants redirect to root after login.
        page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=30_000)

    if "/login" in page.url:
        raise RuntimeError("Canvas login did not complete. Check credentials and any required MFA steps.")


def fetch_active_courses(context: Any) -> list[CanvasCourse]:
    courses_url = f"{COURSES_API_URL}?enrollment_state=active&state[]=available"
    course_rows = fetch_paginated_json_list(context, courses_url)

    courses: list[CanvasCourse] = []
    for row in course_rows:
        raw_id = row.get("id")
        if not isinstance(raw_id, int):
            continue

        name = str(row.get("name") or row.get("course_code") or f"Course {raw_id}").strip()
        if not name:
            name = f"Course {raw_id}"
        courses.append(CanvasCourse(course_id=raw_id, name=name))

    return sorted(courses, key=lambda course: course.name.lower())


def assignment_numeric_id(assignment: dict[str, Any]) -> int:
    assignment_id = assignment.get("id")
    return assignment_id if isinstance(assignment_id, int) else 0


def assignment_updated_ts(assignment: dict[str, Any]) -> float:
    updated_at = parse_canvas_datetime(
        assignment.get("updated_at") if isinstance(assignment.get("updated_at"), str) else None
    )
    return updated_at.timestamp() if updated_at else float("-inf")


def assignment_due_ts(assignment: dict[str, Any]) -> float:
    due_at = parse_canvas_datetime(
        assignment.get("due_at") if isinstance(assignment.get("due_at"), str) else None
    )
    return due_at.timestamp() if due_at else float("-inf")


def select_most_recent_assignment(
    assignments: list[dict[str, Any]],
    now_dt: datetime | None = None,
) -> dict[str, Any] | None:
    current_dt = now_dt or datetime.now().astimezone()
    candidates = [assignment for assignment in assignments if assignment.get("published") is not False]
    if not candidates:
        return None

    upcoming_due_candidates: list[tuple[dict[str, Any], float]] = []
    for assignment in candidates:
        due_at = parse_canvas_datetime(
            assignment.get("due_at") if isinstance(assignment.get("due_at"), str) else None
        )
        if due_at is None or due_at < current_dt:
            continue
        upcoming_due_candidates.append((assignment, due_at.timestamp()))

    if upcoming_due_candidates:
        # Primary: earliest upcoming due_at; ties: latest updated_at, highest id.
        winner, _ = min(
            upcoming_due_candidates,
            key=lambda current: (
                current[1],
                -assignment_updated_ts(current[0]),
                -assignment_numeric_id(current[0]),
            ),
        )
        return winner

    # Fallback: latest updated_at; ties: latest due_at, highest id.
    return max(
        candidates,
        key=lambda assignment: (
            assignment_updated_ts(assignment),
            assignment_due_ts(assignment),
            assignment_numeric_id(assignment),
        ),
    )


def fetch_most_recent_assignment(context: Any, course_id: int) -> dict[str, Any] | None:
    assignments_url = f"{BASE_URL}/api/v1/courses/{course_id}/assignments"
    assignments = fetch_paginated_json_list(context, assignments_url)
    if not assignments:
        return None

    return select_most_recent_assignment(assignments)


def fetch_course_modules_with_items(context: Any, course_id: int) -> list[dict[str, Any]]:
    modules_url = f"{BASE_URL}/api/v1/courses/{course_id}/modules?include[]=items"
    return fetch_paginated_json_list(context, modules_url)


def fetch_module_items(context: Any, course_id: int, module_id: int) -> list[dict[str, Any]]:
    items_url = f"{BASE_URL}/api/v1/courses/{course_id}/modules/{module_id}/items"
    return fetch_paginated_json_list(context, items_url)


def fetch_course_assignments_with_submission(context: Any, course_id: int) -> list[dict[str, Any]]:
    assignments_url = f"{BASE_URL}/api/v1/courses/{course_id}/assignments?include[]=submission"
    return fetch_paginated_json_list(context, assignments_url)


def fetch_assignment_details(context: Any, course_id: int, assignment_id: int) -> dict[str, Any] | None:
    details_url = f"{BASE_URL}/api/v1/courses/{course_id}/assignments/{assignment_id}"
    response = context.request.get(details_url, timeout=30_000)
    if not response.ok:
        return None
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    return payload


def fetch_submission_for_assignment(
    context: Any,
    course_id: int,
    assignment_id: int,
) -> dict[str, Any] | None:
    submission_url = (
        f"{BASE_URL}/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/self"
        "?include[]=submission_history&include[]=submission_comments"
    )
    response = context.request.get(submission_url, timeout=30_000)
    if not response.ok:
        return None
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    return payload


def fetch_course_file_details(
    context: Any,
    course_id: int,
    file_id: int,
    *,
    file_api_url: str | None = None,
) -> dict[str, Any] | None:
    endpoint = file_api_url if isinstance(file_api_url, str) and "/api/v1/" in file_api_url else None
    if not endpoint:
        endpoint = f"{BASE_URL}/api/v1/courses/{course_id}/files/{file_id}"

    response = context.request.get(endpoint, timeout=30_000)
    if not response.ok:
        return None
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    return payload


def assignment_display_name(assignment: dict[str, Any]) -> str:
    return str(
        assignment.get("name")
        or assignment.get("title")
        or f"Assignment {assignment.get('id', 'N/A')}"
    ).strip()


def normalize_similarity_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize_for_similarity(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", value.lower()):
        if raw in SIMILARITY_STOP_WORDS or raw.isdigit():
            continue
        normalized = normalize_similarity_token(raw)
        if len(normalized) <= 2 or normalized in SIMILARITY_STOP_WORDS:
            continue
        tokens.add(normalized)
    return tokens


def score_similarity(assignment_tokens: set[str], candidate_text: str) -> tuple[int, float]:
    candidate_tokens = tokenize_for_similarity(candidate_text)
    if not assignment_tokens or not candidate_tokens:
        return (0, 0.0)

    overlap = assignment_tokens.intersection(candidate_tokens)
    overlap_count = len(overlap)
    if overlap_count == 0:
        return (0, 0.0)

    # Jaccard similarity gives a stable tie-break after overlap count.
    ratio = overlap_count / len(assignment_tokens.union(candidate_tokens))
    return (overlap_count, ratio)


def find_modules_for_assignment(
    modules: list[dict[str, Any]],
    assignment: dict[str, Any],
) -> list[str]:
    assignment_id = assignment.get("id")
    assignment_name = assignment_display_name(assignment).lower()
    direct_matches: list[str] = []

    for module in modules:
        module_id = module.get("id")
        module_name = str(module.get("name") or module.get("title") or "").strip()
        if not module_name:
            module_name = f"Module {module_id}" if isinstance(module_id, int) else "Unnamed module"

        items = module.get("items")
        if not isinstance(items, list):
            continue

        found_in_module = False
        for item in items:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "").strip().lower()
            item_content_id = item.get("content_id")
            item_title = str(item.get("title") or "").strip().lower()

            id_match = (
                isinstance(assignment_id, int)
                and item_type == "assignment"
                and isinstance(item_content_id, int)
                and item_content_id == assignment_id
            )
            title_match = bool(assignment_name and item_title and assignment_name == item_title)

            if id_match or title_match:
                found_in_module = True
                break

        if found_in_module and module_name not in direct_matches:
            direct_matches.append(module_name)

    if direct_matches:
        return direct_matches

    assignment_tokens = tokenize_for_similarity(assignment_name)
    if not assignment_tokens:
        return []

    fuzzy_candidates: list[tuple[str, int, float, int]] = []
    for module in modules:
        module_id = module.get("id")
        module_name = str(module.get("name") or module.get("title") or "").strip()
        if not module_name:
            module_name = f"Module {module_id}" if isinstance(module_id, int) else "Unnamed module"

        module_position = module.get("position")
        normalized_position = module_position if isinstance(module_position, int) else 10_000
        best_overlap = 0
        best_ratio = 0.0

        module_overlap, module_ratio = score_similarity(assignment_tokens, module_name)
        if module_overlap > best_overlap or (
            module_overlap == best_overlap and module_ratio > best_ratio
        ):
            best_overlap = module_overlap
            best_ratio = module_ratio

        items = module.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_title = str(item.get("title") or "").strip()
                if not item_title:
                    continue
                overlap, ratio = score_similarity(assignment_tokens, item_title)
                if overlap > best_overlap or (overlap == best_overlap and ratio > best_ratio):
                    best_overlap = overlap
                    best_ratio = ratio

        if best_overlap > 0:
            fuzzy_candidates.append((module_name, best_overlap, best_ratio, normalized_position))

    if not fuzzy_candidates:
        return []

    fuzzy_candidates.sort(key=lambda current: (-current[1], -current[2], current[3], current[0].lower()))
    best_module, best_overlap, _, _ = fuzzy_candidates[0]
    if best_overlap < 2:
        return []
    return [best_module]


def sanitize_path_part(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return sanitized[:80] if sanitized else fallback


def module_name_from_payload(module: dict[str, Any]) -> str:
    module_id = module.get("id")
    module_name = str(module.get("name") or module.get("title") or "").strip()
    if module_name:
        return module_name
    if isinstance(module_id, int):
        return f"Module {module_id}"
    return "Unnamed module"


def resolve_module_id_by_name(modules: list[dict[str, Any]], module_name: str) -> int | None:
    for module in modules:
        current_name = module_name_from_payload(module)
        if current_name != module_name:
            continue
        module_id = module.get("id")
        if isinstance(module_id, int):
            return module_id
    return None


def collect_module_assignments(
    context: Any,
    course_id: int,
    module_id: int,
) -> list[dict[str, Any]]:
    module_items = fetch_module_items(context, course_id, module_id)

    ordered_assignment_ids: list[int] = []
    assignment_item_meta: dict[int, dict[str, Any]] = {}
    non_assignment_rows: list[dict[str, Any]] = []
    for item in module_items:
        item_type = str(item.get("type") or "").strip().lower()
        content_id = item.get("content_id")
        if item_type != "assignment" or not isinstance(content_id, int):
            non_assignment_rows.append(
                {
                    "id": content_id if isinstance(content_id, int) else None,
                    "title": str(item.get("title") or "Untitled").strip(),
                    "item_type": item.get("type"),
                    "due_at": None,
                    "due_display": "N/A",
                    "created_at": None,
                    "updated_at": None,
                    "url": absolutize_url(
                        item.get("html_url")
                        if isinstance(item.get("html_url"), str)
                        else None
                    ),
                    "published": item.get("published"),
                    "points_possible": None,
                    "module_item_position": item.get("position"),
                    "module_item_title": item.get("title"),
                }
            )
            continue
        if content_id in assignment_item_meta:
            continue
        ordered_assignment_ids.append(content_id)
        assignment_item_meta[content_id] = item

    collected: list[dict[str, Any]] = []
    for assignment_id in ordered_assignment_ids:
        details_url = f"{BASE_URL}/api/v1/courses/{course_id}/assignments/{assignment_id}"
        response = context.request.get(details_url, timeout=30_000)
        item_meta = assignment_item_meta[assignment_id]

        details: dict[str, Any] | None = None
        if response.ok:
            payload = response.json()
            if isinstance(payload, dict):
                details = payload

        if details is not None:
            due_at_raw = details.get("due_at") if isinstance(details.get("due_at"), str) else None
            created_raw = (
                details.get("created_at") if isinstance(details.get("created_at"), str) else None
            )
            updated_raw = (
                details.get("updated_at") if isinstance(details.get("updated_at"), str) else None
            )
            collected.append(
                {
                    "id": assignment_id,
                    "title": assignment_display_name(details),
                    "item_type": "Assignment",
                    "due_at": due_at_raw,
                    "due_display": format_dt(due_at_raw),
                    "created_at": created_raw,
                    "updated_at": updated_raw,
                    "url": absolutize_url(
                        details.get("html_url")
                        if isinstance(details.get("html_url"), str)
                        else None
                    ),
                    "published": details.get("published"),
                    "points_possible": details.get("points_possible"),
                    "module_item_position": item_meta.get("position"),
                    "module_item_title": item_meta.get("title"),
                }
            )
            continue

        fallback_title = str(item_meta.get("title") or f"Assignment {assignment_id}").strip()
        collected.append(
            {
                "id": assignment_id,
                "title": fallback_title,
                "item_type": "Assignment",
                "due_at": None,
                "due_display": "N/A",
                "created_at": None,
                "updated_at": None,
                "url": absolutize_url(
                    item_meta.get("html_url")
                    if isinstance(item_meta.get("html_url"), str)
                    else None
                ),
                "published": None,
                "points_possible": None,
                "module_item_position": item_meta.get("position"),
                "module_item_title": item_meta.get("title"),
            }
        )

    all_rows = collected + non_assignment_rows
    all_rows.sort(
        key=lambda row: (
            row.get("module_item_position")
            if isinstance(row.get("module_item_position"), int)
            else 10_000,
            str(row.get("title") or "").lower(),
        )
    )
    return all_rows


def save_module_assignments_to_cache(
    context: Any,
    result: CourseReadResult,
    *,
    cache_dir: str | Path = "Cache",
) -> CourseReadResult:
    if result.module_id is None or not result.module_name:
        result.module_assignment_count = 0
        result.saved_assignments_path = None
        return result

    module_items = collect_module_assignments(context, result.course.course_id, result.module_id)
    cache_root = Path(cache_dir).resolve()
    course_dir = sanitize_path_part(result.course.name, "course")
    module_dir = sanitize_path_part(result.module_name, "module")
    output_dir = cache_root / course_dir / module_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "assignments.json"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": {
            "id": result.course.course_id,
            "name": result.course.name,
        },
        "module": {
            "id": result.module_id,
            "name": result.module_name,
        },
        "item_count": len(module_items),
        "assignment_count": sum(
            1 for item in module_items if str(item.get("item_type") or "").lower() == "assignment"
        ),
        "items": module_items,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = output_dir / "assignments.txt"
    lines = [f"Course: {result.course.name}", f"Module: {result.module_name}", ""]
    for index, item in enumerate(module_items, start=1):
        lines.append(f"{index}. {item.get('title', 'Untitled')}")
        lines.append(f"   Type: {item.get('item_type', 'Unknown')}")
        lines.append(f"   Due: {item.get('due_display', 'N/A')}")
        lines.append(f"   URL: {item.get('url', 'N/A')}")
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    result.module_assignment_count = len(module_items)
    result.saved_assignments_path = str(json_path)
    return result


def is_assignment_submitted(assignment: dict[str, Any]) -> bool:
    submission = assignment.get("submission")
    if not isinstance(submission, dict):
        return False

    workflow_state = str(submission.get("workflow_state") or "").strip().lower()
    submitted_at = submission.get("submitted_at")
    graded_at = submission.get("graded_at")
    attempt = submission.get("attempt")
    score = submission.get("score")
    grade = submission.get("grade")

    if isinstance(submitted_at, str) and submitted_at:
        return True
    if isinstance(graded_at, str) and graded_at:
        return True
    if attempt not in (None, 0):
        return True
    if score is not None:
        return True
    if grade not in (None, ""):
        return True
    if workflow_state in {"submitted", "graded", "pending_review", "complete"}:
        return True
    return False


def normalize_attachment_filename(value: str | None, fallback: str) -> str:
    decoded = unquote(value or "").strip()
    if not decoded:
        return fallback
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", decoded).strip("._")
    return name or fallback


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def collect_submission_artifacts(submission: dict[str, Any]) -> dict[str, Any]:
    attachments: list[dict[str, Any]] = []
    seen_attachment_keys: set[tuple[str, str, str]] = set()

    def append_attachments(values: list[Any]) -> None:
        for value in values:
            if not isinstance(value, dict):
                continue
            file_id = str(value.get("id") or "")
            file_url = str(value.get("url") or "")
            file_name = str(value.get("filename") or value.get("display_name") or "")
            key = (file_id, file_url, file_name)
            if key in seen_attachment_keys:
                continue
            seen_attachment_keys.add(key)
            attachments.append(value)

    root_attachments = submission.get("attachments")
    if isinstance(root_attachments, list):
        append_attachments(root_attachments)

    submission_history = submission.get("submission_history")
    if isinstance(submission_history, list):
        for history_item in submission_history:
            if not isinstance(history_item, dict):
                continue
            history_attachments = history_item.get("attachments")
            if isinstance(history_attachments, list):
                append_attachments(history_attachments)

    urls: list[str] = []
    for possible_url in [submission.get("url")]:
        if isinstance(possible_url, str) and possible_url.strip():
            urls.append(possible_url.strip())

    if isinstance(submission_history, list):
        for history_item in submission_history:
            if not isinstance(history_item, dict):
                continue
            history_url = history_item.get("url")
            if isinstance(history_url, str) and history_url.strip():
                urls.append(history_url.strip())

    deduped_urls: list[str] = []
    seen_urls: set[str] = set()
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_urls.append(url)

    body = submission.get("body") if isinstance(submission.get("body"), str) else None
    if (not body) and isinstance(submission_history, list):
        for history_item in reversed(submission_history):
            if not isinstance(history_item, dict):
                continue
            history_body = history_item.get("body")
            if isinstance(history_body, str) and history_body.strip():
                body = history_body
                break

    return {
        "attachments": attachments,
        "urls": deduped_urls,
        "body": body,
    }


def download_submission_attachment(
    context: Any,
    url: str,
    target_path: Path,
) -> tuple[bool, str | None]:
    response = context.request.get(url, timeout=30_000)
    if not response.ok:
        return False, f"{response.status} {response.status_text}"

    try:
        target_path.write_bytes(response.body())
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def build_submitted_assignment_row(assignment: dict[str, Any]) -> dict[str, Any]:
    submission = assignment.get("submission") if isinstance(assignment.get("submission"), dict) else {}
    due_at_raw = assignment.get("due_at") if isinstance(assignment.get("due_at"), str) else None
    created_raw = assignment.get("created_at") if isinstance(assignment.get("created_at"), str) else None
    updated_raw = assignment.get("updated_at") if isinstance(assignment.get("updated_at"), str) else None
    submitted_at_raw = (
        submission.get("submitted_at") if isinstance(submission.get("submitted_at"), str) else None
    )
    graded_at_raw = submission.get("graded_at") if isinstance(submission.get("graded_at"), str) else None

    return {
        "id": assignment.get("id"),
        "title": assignment_display_name(assignment),
        "due_at": due_at_raw,
        "due_display": format_dt(due_at_raw),
        "created_at": created_raw,
        "updated_at": updated_raw,
        "url": absolutize_url(
            assignment.get("html_url") if isinstance(assignment.get("html_url"), str) else None
        ),
        "submission": {
            "workflow_state": submission.get("workflow_state"),
            "submitted_at": submitted_at_raw,
            "submitted_display": format_dt(submitted_at_raw),
            "graded_at": graded_at_raw,
            "graded_display": format_dt(graded_at_raw),
            "attempt": submission.get("attempt"),
            "score": submission.get("score"),
            "grade": submission.get("grade"),
            "late": submission.get("late"),
            "missing": submission.get("missing"),
            "excused": submission.get("excused"),
        },
    }


def fetch_submitted_assignments(context: Any, course_id: int) -> list[dict[str, Any]]:
    assignments = fetch_course_assignments_with_submission(context, course_id)
    submitted = [assignment for assignment in assignments if is_assignment_submitted(assignment)]
    submitted_rows = [build_submitted_assignment_row(assignment) for assignment in submitted]
    fallback_dt = datetime.fromtimestamp(0).astimezone()
    submitted_rows.sort(
        key=lambda row: (
            parse_canvas_datetime(
                row.get("submission", {}).get("submitted_at")
                if isinstance(row.get("submission"), dict)
                else None
            )
            or parse_canvas_datetime(row.get("due_at") if isinstance(row.get("due_at"), str) else None)
            or parse_canvas_datetime(row.get("updated_at") if isinstance(row.get("updated_at"), str) else None)
            or fallback_dt,
            str(row.get("title") or "").lower(),
        ),
        reverse=True,
    )
    return submitted_rows


def save_submitted_assignments_to_cache(
    context: Any,
    result: CourseReadResult,
    *,
    cache_dir: str | Path = "Cache",
) -> CourseReadResult:
    submitted_rows = fetch_submitted_assignments(context, result.course.course_id)

    cache_root = Path(cache_dir).resolve()
    course_dir = sanitize_path_part(result.course.name, "course")
    output_dir = cache_root / course_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "submitted_assignments.json"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": {
            "id": result.course.course_id,
            "name": result.course.name,
        },
        "submitted_assignment_count": len(submitted_rows),
        "submitted_assignments": submitted_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = output_dir / "submitted_assignments.txt"
    lines = [f"Course: {result.course.name}", ""]
    if not submitted_rows:
        lines.append("No submitted assignments found.")
    else:
        for index, item in enumerate(submitted_rows, start=1):
            submission = item.get("submission") if isinstance(item.get("submission"), dict) else {}
            lines.append(f"{index}. {item.get('title', 'Untitled')}")
            lines.append(f"   Submitted: {submission.get('submitted_display', 'N/A')}")
            lines.append(f"   Due: {item.get('due_display', 'N/A')}")
            lines.append(f"   Grade: {submission.get('grade', 'N/A')}")
            lines.append(f"   Score: {submission.get('score', 'N/A')}")
            lines.append(f"   URL: {item.get('url', 'N/A')}")
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    result.submitted_assignment_count = len(submitted_rows)
    result.saved_submitted_assignments_path = str(json_path)
    return result


def save_submitted_module_files_to_cache(
    context: Any,
    result: CourseReadResult,
    *,
    cache_dir: str | Path = "Cache",
) -> CourseReadResult:
    if result.module_id is None or not result.module_name:
        result.submitted_module_assignment_count = 0
        result.submitted_module_files_count = 0
        result.saved_submitted_module_path = None
        return result

    module_items = fetch_module_items(context, result.course.course_id, result.module_id)
    assignment_rows: list[tuple[int, int, str]] = []
    seen_assignment_ids: set[int] = set()
    for item in module_items:
        item_type = str(item.get("type") or "").strip().lower()
        content_id = item.get("content_id")
        if item_type != "assignment" or not isinstance(content_id, int):
            continue
        if content_id in seen_assignment_ids:
            continue
        seen_assignment_ids.add(content_id)
        position = item.get("position") if isinstance(item.get("position"), int) else 10_000
        title = str(item.get("title") or f"Assignment {content_id}").strip()
        assignment_rows.append((position, content_id, title))

    assignment_rows.sort(key=lambda current: (current[0], current[2].lower()))

    cache_root = Path(cache_dir).resolve()
    course_dir = sanitize_path_part(result.course.name, "course")
    module_dir = sanitize_path_part(result.module_name, "module")
    output_dir = cache_root / course_dir / module_dir / "submitted"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, Any]] = []
    downloaded_files_count = 0

    for _, assignment_id, module_item_title in assignment_rows:
        submission = fetch_submission_for_assignment(context, result.course.course_id, assignment_id)
        if not isinstance(submission, dict):
            continue
        if not is_assignment_submitted({"submission": submission}):
            continue

        assignment_details = fetch_assignment_details(context, result.course.course_id, assignment_id)
        assignment_title = (
            assignment_display_name(assignment_details)
            if isinstance(assignment_details, dict)
            else module_item_title
        )
        assignment_slug = sanitize_path_part(
            f"{len(exported_rows) + 1:02d}_{assignment_title}_{assignment_id}",
            f"assignment_{assignment_id}",
        )
        assignment_dir = output_dir / assignment_slug
        assignment_dir.mkdir(parents=True, exist_ok=True)

        artifacts = collect_submission_artifacts(submission)
        attachment_records: list[dict[str, Any]] = []
        for index, attachment in enumerate(artifacts.get("attachments", []), start=1):
            if not isinstance(attachment, dict):
                continue
            attachment_url = attachment.get("url")
            if not isinstance(attachment_url, str) or not attachment_url.strip():
                continue

            raw_name = (
                attachment.get("filename")
                if isinstance(attachment.get("filename"), str)
                else attachment.get("display_name")
            )
            filename = normalize_attachment_filename(raw_name, f"attachment_{index}")
            target_path = ensure_unique_path(assignment_dir / filename)
            ok, error_message = download_submission_attachment(context, attachment_url, target_path)

            attachment_record = {
                "filename": filename,
                "source_url": attachment_url,
                "saved_path": str(target_path) if ok else None,
                "saved": ok,
                "error": error_message,
                "content_type": attachment.get("content-type"),
                "size": attachment.get("size"),
            }
            attachment_records.append(attachment_record)
            if ok:
                downloaded_files_count += 1

        body_text = artifacts.get("body")
        if isinstance(body_text, str) and body_text.strip():
            body_path = assignment_dir / "submission_body.txt"
            body_path.write_text(body_text.strip() + "\n", encoding="utf-8")

        urls = artifacts.get("urls") if isinstance(artifacts.get("urls"), list) else []
        normalized_urls = [url for url in urls if isinstance(url, str) and url.strip()]
        if normalized_urls:
            urls_path = assignment_dir / "submission_urls.txt"
            urls_path.write_text("\n".join(normalized_urls).strip() + "\n", encoding="utf-8")

        submission_payload = {
            "course_id": result.course.course_id,
            "assignment_id": assignment_id,
            "assignment_title": assignment_title,
            "assignment_url": absolutize_url(
                assignment_details.get("html_url")
                if isinstance(assignment_details, dict) and isinstance(assignment_details.get("html_url"), str)
                else None
            ),
            "module_name": result.module_name,
            "submission": {
                "workflow_state": submission.get("workflow_state"),
                "submission_type": submission.get("submission_type"),
                "submitted_at": submission.get("submitted_at"),
                "submitted_display": format_dt(
                    submission.get("submitted_at")
                    if isinstance(submission.get("submitted_at"), str)
                    else None
                ),
                "graded_at": submission.get("graded_at"),
                "grade": submission.get("grade"),
                "score": submission.get("score"),
                "attempt": submission.get("attempt"),
                "late": submission.get("late"),
                "missing": submission.get("missing"),
            },
            "attachment_count": len(attachment_records),
            "attachments": attachment_records,
            "url_count": len(normalized_urls),
            "urls": normalized_urls,
            "has_body_text": isinstance(body_text, str) and bool(body_text.strip()),
        }
        submission_json_path = assignment_dir / "submission.json"
        submission_json_path.write_text(json.dumps(submission_payload, indent=2), encoding="utf-8")

        exported_rows.append(
            {
                "assignment_id": assignment_id,
                "assignment_title": assignment_title,
                "assignment_dir": str(assignment_dir),
                "submission_json": str(submission_json_path),
                "attachment_count": len(attachment_records),
                "downloaded_attachment_count": sum(
                    1 for attachment in attachment_records if attachment.get("saved") is True
                ),
                "url_count": len(normalized_urls),
                "has_body_text": isinstance(body_text, str) and bool(body_text.strip()),
            }
        )

    summary_json_path = output_dir / "submitted_assignments.json"
    summary_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": {
            "id": result.course.course_id,
            "name": result.course.name,
        },
        "module": {
            "id": result.module_id,
            "name": result.module_name,
        },
        "submitted_assignment_count": len(exported_rows),
        "downloaded_file_count": downloaded_files_count,
        "submitted_assignments": exported_rows,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_txt_path = output_dir / "submitted_assignments.txt"
    lines = [f"Course: {result.course.name}", f"Module: {result.module_name}", ""]
    if not exported_rows:
        lines.append("No submitted assignments found in this module.")
    else:
        for index, row in enumerate(exported_rows, start=1):
            lines.append(f"{index}. {row.get('assignment_title', 'Untitled')}")
            lines.append(f"   Assignment folder: {row.get('assignment_dir', 'N/A')}")
            lines.append(
                "   Downloaded files: "
                f"{row.get('downloaded_attachment_count', 0)} / {row.get('attachment_count', 0)}"
            )
            lines.append(
                f"   URLs captured: {row.get('url_count', 0)} | Has body text: {row.get('has_body_text', False)}"
            )
    summary_txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    result.submitted_module_assignment_count = len(exported_rows)
    result.submitted_module_files_count = downloaded_files_count
    result.saved_submitted_module_path = str(summary_json_path)
    return result


def save_tasked_module_files_to_cache(
    context: Any,
    result: CourseReadResult,
    *,
    cache_dir: str | Path = "Cache",
) -> CourseReadResult:
    if result.module_id is None or not result.module_name:
        result.tasked_module_item_count = 0
        result.tasked_module_files_count = 0
        result.saved_tasked_module_path = None
        return result

    module_items = fetch_module_items(context, result.course.course_id, result.module_id)
    sorted_items = sorted(
        module_items,
        key=lambda item: (
            item.get("position") if isinstance(item.get("position"), int) else 10_000,
            str(item.get("title") or "").lower(),
        ),
    )

    cache_root = Path(cache_dir).resolve()
    course_dir = sanitize_path_part(result.course.name, "course")
    module_dir = sanitize_path_part(result.module_name, "module")
    output_dir = cache_root / course_dir / module_dir / "tasked"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_items: list[dict[str, Any]] = []
    downloaded_files_count = 0

    for index, item in enumerate(sorted_items, start=1):
        item_type = str(item.get("type") or "Unknown").strip()
        title = str(item.get("title") or f"Item {index}").strip()
        item_slug = sanitize_path_part(f"{index:02d}_{title}", f"item_{index}")
        item_dir = output_dir / item_slug
        item_dir.mkdir(parents=True, exist_ok=True)

        item_record: dict[str, Any] = {
            "position": item.get("position"),
            "module_item_id": item.get("id"),
            "type": item_type,
            "title": title,
            "content_id": item.get("content_id"),
            "html_url": item.get("html_url"),
            "url": item.get("url"),
            "external_url": item.get("external_url"),
            "downloaded_files": [],
            "assignment_info": None,
        }

        item_type_lower = item_type.lower()
        content_id = item.get("content_id")
        if item_type_lower == "file" and isinstance(content_id, int):
            file_details = fetch_course_file_details(
                context,
                result.course.course_id,
                content_id,
                file_api_url=item.get("url") if isinstance(item.get("url"), str) else None,
            )
            if file_details is not None:
                download_url = file_details.get("url") if isinstance(file_details.get("url"), str) else None
                filename = normalize_attachment_filename(
                    file_details.get("display_name")
                    if isinstance(file_details.get("display_name"), str)
                    else file_details.get("filename")
                    if isinstance(file_details.get("filename"), str)
                    else title,
                    f"file_{content_id}",
                )
                if download_url:
                    target_path = ensure_unique_path(item_dir / filename)
                    ok, error_message = download_submission_attachment(context, download_url, target_path)
                    file_record = {
                        "filename": filename,
                        "saved_path": str(target_path) if ok else None,
                        "saved": ok,
                        "error": error_message,
                        "download_url": download_url,
                        "size": file_details.get("size"),
                        "content_type": file_details.get("content-type"),
                    }
                    item_record["downloaded_files"].append(file_record)
                    if ok:
                        downloaded_files_count += 1
                item_record["file_details"] = {
                    "id": file_details.get("id"),
                    "display_name": file_details.get("display_name"),
                    "filename": file_details.get("filename"),
                    "size": file_details.get("size"),
                    "mime_class": file_details.get("mime_class"),
                    "updated_at": file_details.get("updated_at"),
                }
            else:
                item_record["file_details"] = None

        if item_type_lower == "assignment" and isinstance(content_id, int):
            assignment_details = fetch_assignment_details(context, result.course.course_id, content_id)
            if assignment_details is not None:
                item_record["assignment_info"] = {
                    "id": assignment_details.get("id"),
                    "title": assignment_display_name(assignment_details),
                    "due_at": assignment_details.get("due_at"),
                    "due_display": format_dt(
                        assignment_details.get("due_at")
                        if isinstance(assignment_details.get("due_at"), str)
                        else None
                    ),
                    "url": absolutize_url(
                        assignment_details.get("html_url")
                        if isinstance(assignment_details.get("html_url"), str)
                        else None
                    ),
                }

        links: list[str] = []
        for key in ("html_url", "url", "external_url"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                links.append(value.strip())
        deduped_links: list[str] = []
        seen_links: set[str] = set()
        for link in links:
            if link in seen_links:
                continue
            seen_links.add(link)
            deduped_links.append(link)
        if deduped_links:
            links_path = item_dir / "item_links.txt"
            links_path.write_text("\n".join(deduped_links).strip() + "\n", encoding="utf-8")

        item_json_path = item_dir / "item.json"
        item_json_path.write_text(json.dumps(item_record, indent=2), encoding="utf-8")

        exported_items.append(
            {
                "position": item_record.get("position"),
                "type": item_type,
                "title": title,
                "item_dir": str(item_dir),
                "downloaded_file_count": len(
                    [file for file in item_record.get("downloaded_files", []) if file.get("saved") is True]
                ),
            }
        )

    summary_json_path = output_dir / "tasked_items.json"
    summary_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": {
            "id": result.course.course_id,
            "name": result.course.name,
        },
        "module": {
            "id": result.module_id,
            "name": result.module_name,
        },
        "tasked_item_count": len(exported_items),
        "downloaded_file_count": downloaded_files_count,
        "items": exported_items,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_txt_path = output_dir / "tasked_items.txt"
    lines = [f"Course: {result.course.name}", f"Module: {result.module_name}", ""]
    if not exported_items:
        lines.append("No tasked module items found.")
    else:
        for index, row in enumerate(exported_items, start=1):
            lines.append(f"{index}. {row.get('title', 'Untitled')} ({row.get('type', 'Unknown')})")
            lines.append(f"   Item folder: {row.get('item_dir', 'N/A')}")
            lines.append(f"   Downloaded files: {row.get('downloaded_file_count', 0)}")
    summary_txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    result.tasked_module_item_count = len(exported_items)
    result.tasked_module_files_count = downloaded_files_count
    result.saved_tasked_module_path = str(summary_json_path)
    return result


def resolve_course_selection(
    courses: list[CanvasCourse],
    selection: int | str,
) -> CanvasCourse | None:
    if isinstance(selection, int):
        if 1 <= selection <= len(courses):
            return courses[selection - 1]
        for course in courses:
            if course.course_id == selection:
                return course
        return None

    normalized = selection.strip()
    if not normalized:
        return None

    if normalized.isdigit():
        return resolve_course_selection(courses, int(normalized))

    lowered = normalized.lower()
    exact_matches = [course for course in courses if course.name.lower() == lowered]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [course for course in courses if lowered in course.name.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def sort_modules_for_selection(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        modules,
        key=lambda module: (
            module.get("position") if isinstance(module.get("position"), int) else 10_000,
            module_name_from_payload(module).lower(),
        ),
    )


def resolve_module_selection(
    modules: list[dict[str, Any]],
    selection: int | str,
) -> dict[str, Any] | None:
    if isinstance(selection, int):
        if 1 <= selection <= len(modules):
            return modules[selection - 1]
        for module in modules:
            module_id = module.get("id")
            if isinstance(module_id, int) and module_id == selection:
                return module
        return None

    normalized = selection.strip()
    if not normalized:
        return None

    if normalized.isdigit():
        return resolve_module_selection(modules, int(normalized))

    lowered = normalized.lower()
    exact_matches = [module for module in modules if module_name_from_payload(module).lower() == lowered]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [module for module in modules if lowered in module_name_from_payload(module).lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def prompt_course_selection(
    courses: list[CanvasCourse],
    *,
    input_fn: Callable[[str], str] = input,
) -> CanvasCourse:
    print(f"Classes ({len(courses)}):")
    for index, course in enumerate(courses, start=1):
        print(f"{index}. {course.name} (ID: {course.course_id})")

    while True:
        choice = input_fn("\nPick class by number, course ID, or class name: ").strip()
        selected_course = resolve_course_selection(courses, choice)
        if selected_course is not None:
            return selected_course
        print("Invalid class selection. Please try again.")


def prompt_module_selection(
    modules: list[dict[str, Any]],
    *,
    input_fn: Callable[[str], str] = input,
) -> dict[str, Any]:
    print(f"\nModules ({len(modules)}):")
    for index, module in enumerate(modules, start=1):
        module_name = module_name_from_payload(module)
        module_id = module.get("id")
        module_id_label = module_id if isinstance(module_id, int) else "N/A"
        print(f"{index}. {module_name} (ID: {module_id_label})")

    while True:
        choice = input_fn("\nPick module by number, module ID, or module name: ").strip()
        selected_module = resolve_module_selection(modules, choice)
        if selected_module is not None:
            return selected_module
        print("Invalid module selection. Please try again.")


def build_course_read_result(
    context: Any,
    course: CanvasCourse,
    *,
    modules: list[dict[str, Any]] | None = None,
) -> CourseReadResult:
    assignment = fetch_most_recent_assignment(context, course.course_id)
    if assignment is None:
        return CourseReadResult(
            course=course,
            assignment_name=None,
            assignment_url=None,
            created_at=None,
            due_at=None,
            module_label=None,
            module_name=None,
            module_id=None,
        )

    assignment_name = assignment_display_name(assignment)
    assignment_url = absolutize_url(
        assignment.get("html_url")
        if isinstance(assignment.get("html_url"), str)
        else None
    )
    created_at = format_dt(
        assignment.get("created_at")
        if isinstance(assignment.get("created_at"), str)
        else None
    )
    due_at = format_dt(
        assignment.get("due_at")
        if isinstance(assignment.get("due_at"), str)
        else None
    )
    module_label = "No matching module found"
    module_name: str | None = None
    module_id: int | None = None
    try:
        module_rows = modules if modules is not None else fetch_course_modules_with_items(context, course.course_id)
        matched_modules = find_modules_for_assignment(module_rows, assignment)
        if matched_modules:
            module_label = ", ".join(matched_modules)
            module_name = matched_modules[0]
            module_id = resolve_module_id_by_name(module_rows, module_name)
    except Exception:
        module_label = "Unavailable (could not read course modules)"

    return CourseReadResult(
        course=course,
        assignment_name=assignment_name,
        assignment_url=assignment_url,
        created_at=created_at,
        due_at=due_at,
        module_label=module_label,
        module_name=module_name,
        module_id=module_id,
    )


def print_course_read_result(result: CourseReadResult) -> None:
    print(f"Selected class: {result.course.name} (ID: {result.course.course_id})")
    if result.assignment_name is None:
        print("Most recent assignment: No assignments found.")
    else:
        print(f"Most recent assignment: {result.assignment_name}")
        print(f"Created: {result.created_at} | Due: {result.due_at}")
        print(f"URL: {result.assignment_url}")

    module_label = result.module_label or "N/A"
    print(f"Module/Unit: {module_label}")
    if result.saved_assignments_path:
        print(f"Saved module items: {result.module_assignment_count}")
        print(f"Cache file: {result.saved_assignments_path}")
    if result.saved_submitted_assignments_path:
        print(f"Submitted assignments: {result.submitted_assignment_count}")
        print(f"Submitted cache file: {result.saved_submitted_assignments_path}")
    if result.saved_submitted_module_path:
        print(
            "Submitted assignments in module: "
            f"{result.submitted_module_assignment_count}"
        )
        print(
            "Downloaded submitted files in module: "
            f"{result.submitted_module_files_count}"
        )
        print(f"Submitted module cache file: {result.saved_submitted_module_path}")
    if result.saved_tasked_module_path:
        print(
            "Tasked items in module: "
            f"{result.tasked_module_item_count}"
        )
        print(
            "Downloaded tasked files in module: "
            f"{result.tasked_module_files_count}"
        )
        print(f"Tasked module cache file: {result.saved_tasked_module_path}")


def run_canvas(
    *,
    course_selection: int | str | None = None,
    prompt_for_selection: bool = True,
    module_selection: int | str | None = None,
    prompt_for_module_selection: bool = True,
    headless: bool | None = None,
    input_fn: Callable[[str], str] = input,
    save_module_assignments: bool = False,
    save_submitted_assignments: bool = False,
    save_submitted_module_files: bool = False,
    save_tasked_module_files: bool = False,
    cache_dir: str | Path = "Cache",
    print_output: bool = True,
) -> CourseReadResult | None:
    load_dotenv()

    username = pick_env("LCDS_USERNAME", "CANVAS_USERNAME", "USERNAME")
    password = pick_env("LCDS_PASSWORD", "CANVAS_PASSWORD", "PASSWORD")
    resolved_headless = as_bool(os.getenv("HEADLESS"), default=True) if headless is None else headless

    if not username or not password:
        raise RuntimeError(
            "Missing credentials. Add one of these pairs to .env:\n"
            "- LCDS_USERNAME + LCDS_PASSWORD (preferred)\n"
            "- CANVAS_USERNAME + CANVAS_PASSWORD\n"
            "- USERNAME + PASSWORD"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=resolved_headless)
        try:
            context = browser.new_context()
            page = context.new_page()

            login_to_canvas(page, username, password)
            courses = fetch_active_courses(context)
            if not courses:
                if print_output:
                    print("No active classes found.")
                return None

            if course_selection is not None:
                selected_course = resolve_course_selection(courses, course_selection)
                if selected_course is None:
                    raise ValueError(f"Could not find class from selection: {course_selection!r}")
            elif prompt_for_selection:
                selected_course = prompt_course_selection(courses, input_fn=input_fn)
            else:
                raise ValueError("course_selection is required when prompt_for_selection=False.")

            selected_module: dict[str, Any] | None = None
            modules: list[dict[str, Any]] | None = None
            if module_selection is not None or prompt_for_module_selection:
                module_rows = fetch_course_modules_with_items(context, selected_course.course_id)
                modules = sort_modules_for_selection(module_rows)
                if module_selection is not None:
                    selected_module = resolve_module_selection(modules, module_selection)
                    if selected_module is None:
                        raise ValueError(f"Could not find module from selection: {module_selection!r}")
                elif prompt_for_module_selection and modules:
                    selected_module = prompt_module_selection(modules, input_fn=input_fn)

            result = build_course_read_result(context, selected_course, modules=modules)
            if selected_module is not None:
                selected_module_name = module_name_from_payload(selected_module)
                selected_module_id = selected_module.get("id")
                result.module_name = selected_module_name
                result.module_id = selected_module_id if isinstance(selected_module_id, int) else None
                result.module_label = selected_module_name

            if save_module_assignments:
                result = save_module_assignments_to_cache(
                    context,
                    result,
                    cache_dir=cache_dir,
                )
            if save_submitted_assignments:
                result = save_submitted_assignments_to_cache(
                    context,
                    result,
                    cache_dir=cache_dir,
                )
            if save_submitted_module_files:
                result = save_submitted_module_files_to_cache(
                    context,
                    result,
                    cache_dir=cache_dir,
                )
            if save_tasked_module_files:
                result = save_tasked_module_files_to_cache(
                    context,
                    result,
                    cache_dir=cache_dir,
                )
            if print_output:
                print()
                print_course_read_result(result)
            return result
        finally:
            browser.close()
