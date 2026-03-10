from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SUPPORTED_LOCAL_SOURCE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".docx"}


@dataclass
class LocalSourceCacheResult:
    study_set_name: str
    source_dir: str
    module_dir: str
    tasked_dir: str
    summary_path: str
    discovered_files_count: int
    staged_file_count: int


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned[:120] if cleaned else fallback


def _safe_stage_name(filename: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip()).strip("._")
    if not cleaned:
        cleaned = f"source_{index:02d}"
    return f"{index:02d}_{cleaned}"


def discover_local_source_files(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Directory not found: {root}")
    if not root.is_dir():
        raise RuntimeError(f"Path is not a directory: {root}")

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_LOCAL_SOURCE_EXTENSIONS
    ]
    candidates.sort(key=lambda path: str(path.relative_to(root)).lower())
    return candidates


def _paths_overlap(a: Path, b: Path) -> bool:
    try:
        return a == b or a.is_relative_to(b) or b.is_relative_to(a)
    except AttributeError:
        a_parts = a.parts
        b_parts = b.parts
        return a == b or a_parts[: len(b_parts)] == b_parts or b_parts[: len(a_parts)] == a_parts


def build_local_source_cache(
    source_dir: str | Path,
    study_set_name: str,
    *,
    cache_dir: str | Path = "Cache",
) -> LocalSourceCacheResult:
    resolved_source_dir = Path(source_dir).expanduser().resolve()
    study_label = study_set_name.strip()
    if not study_label:
        raise RuntimeError("Study-set name cannot be empty.")

    source_files = discover_local_source_files(resolved_source_dir)
    if not source_files:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_SOURCE_EXTENSIONS))
        raise RuntimeError(f"No supported files were found. Supported types: {supported}")

    cache_root = Path(cache_dir).expanduser().resolve()
    module_slug = _slugify(study_label, "custom_upload")
    module_dir = cache_root / "Custom" / module_slug
    tasked_dir = module_dir / "tasked"

    if _paths_overlap(resolved_source_dir, module_dir):
        raise RuntimeError(
            "The selected source directory overlaps with the generated cache directory. "
            "Choose a different source folder."
        )

    if module_dir.exists():
        shutil.rmtree(module_dir)
    tasked_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, object]] = []
    for index, source_file in enumerate(source_files, start=1):
        item_dir = tasked_dir / _safe_stage_name(source_file.name, index)
        item_dir.mkdir(parents=True, exist_ok=True)
        staged_path = item_dir / source_file.name
        shutil.copy2(source_file, staged_path)

        relative_title = str(source_file.relative_to(resolved_source_dir))
        items.append(
            {
                "position": index,
                "type": "LocalFile",
                "title": relative_title,
                "item_dir": str(item_dir),
                "downloaded_file_count": 1,
            }
        )

    summary_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "course": {"id": None, "name": "Custom Upload"},
        "module": {"id": None, "name": study_label},
        "tasked_item_count": len(items),
        "downloaded_file_count": len(source_files),
        "source_dir": str(resolved_source_dir),
        "items": items,
    }
    summary_path = tasked_dir / "custom_items.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return LocalSourceCacheResult(
        study_set_name=study_label,
        source_dir=str(resolved_source_dir),
        module_dir=str(module_dir),
        tasked_dir=str(tasked_dir),
        summary_path=str(summary_path),
        discovered_files_count=len(source_files),
        staged_file_count=len(source_files),
    )
