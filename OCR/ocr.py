from __future__ import annotations

import os
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path
from typing import Any, Optional

import ocrmypdf
from pypdf import PdfReader

from .PDF.pdf_utils import split_wide_pages

try:
    from doctr.io import DocumentFile as DoctrDocumentFile
except ImportError:
    DoctrDocumentFile = None  # type: ignore[assignment]

_DEFAULT_TESSERACT_PATHS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
    "/usr/bin",
)


def _discover_bundled_tesseract() -> Optional[Path]:
    """Return path to bundled tesseract directory if present."""

    candidates: list[Path] = []
    exe = None
    try:
        exe = Path(sys.executable).resolve()
    except Exception:
        exe = None

    if getattr(sys, "frozen", False):
        if exe is not None:
            mac_resources = exe.parent.parent / "Resources" / "tesseract"
            candidates.append(mac_resources)
            candidates.append(exe.parent / "tesseract")
            candidates.append(exe.parent / "_internal" / "tesseract")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "tesseract")

    # Fallback for local development
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "third_party" / "tesseract-macos")

    for base in candidates:
        try:
            bin_path = base / "bin" / "tesseract"
            if bin_path.exists():
                return base
        except Exception:
            continue
    return None


def ensure_tesseract_available(custom_tesseract_path: str | None = None) -> None:
    bundle_root = _discover_bundled_tesseract()

    if custom_tesseract_path:
        p = Path(custom_tesseract_path)
        if not p.exists():
            raise FileNotFoundError(f"Tesseract not found at: {custom_tesseract_path}")
        os.environ["PATH"] = str(p.parent) + os.pathsep + os.environ.get("PATH", "")
    elif bundle_root is not None:
        bin_dir = bundle_root / "bin"
        lib_dir = bundle_root / "lib"
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("TESSDATA_PREFIX", str(bundle_root / "share"))
        dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + dyld if dyld else str(lib_dir)
    else:
        # When launched as a macOS .app, PATH is short; add common Homebrew locations.
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        missing = [p for p in _DEFAULT_TESSERACT_PATHS if p and p not in path_entries and Path(p).exists()]
        if missing:
            os.environ["PATH"] = os.pathsep.join(missing + path_entries)
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract is not available on PATH.\n\n"
            "Install Tesseract (e.g., UB Mannheim build on Windows) or pick its path "
            "in the 'Tesseract path' field."
        )


def run_ocr(
    input_pdf: str,
    output_pdf: Optional[str] = None,
    languages: str = "eng",
    force: bool = False,
    optimize: int = 0,
    deskew: bool = True,
    clean: bool = False,
    custom_tesseract_path: str | None = None,
) -> str:
    ensure_tesseract_available(custom_tesseract_path)
    out_path = output_pdf or str(Path(input_pdf).with_suffix(".ocr.pdf"))

    CREATE_NO_WINDOW = 0x08000000

    def _wrap_subprocess_call(fn):
        def _wrapped(*args, **kwargs):
            if sys.platform.startswith("win"):
                try:
                    cf = int(kwargs.get("creationflags", 0))
                except Exception:
                    cf = 0
                kwargs["creationflags"] = cf | CREATE_NO_WINDOW
                try:
                    si = kwargs.get("startupinfo") or subprocess.STARTUPINFO()
                    si.dwFlags |= 0x00000001  # STARTF_USESHOWWINDOW
                    si.wShowWindow = 0  # SW_HIDE
                    kwargs["startupinfo"] = si
                except Exception:
                    pass
            return fn(*args, **kwargs)
        return _wrapped

    _orig = {
        "Popen": subprocess.Popen,
        "run": getattr(subprocess, "run", None),
        "call": getattr(subprocess, "call", None),
        "check_call": getattr(subprocess, "check_call", None),
        "check_output": getattr(subprocess, "check_output", None),
    }

    subprocess.Popen = _wrap_subprocess_call(subprocess.Popen)  # type: ignore[assignment]
    if _orig["run"]:
        subprocess.run = _wrap_subprocess_call(_orig["run"])  # type: ignore[assignment]
    if _orig["call"]:
        subprocess.call = _wrap_subprocess_call(_orig["call"])  # type: ignore[assignment]
    if _orig["check_call"]:
        subprocess.check_call = _wrap_subprocess_call(_orig["check_call"])  # type: ignore[assignment]
    if _orig["check_output"]:
        subprocess.check_output = _wrap_subprocess_call(_orig["check_output"])  # type: ignore[assignment]

    patched_modules: list[tuple[object, str, object]] = []
    try:
        for name, mod in list(sys.modules.items()):
            if not name or not name.startswith("ocrmypdf"):
                continue
            try:
                for attr in ("Popen", "run", "call", "check_call", "check_output"):
                    if hasattr(mod, attr):
                        orig = getattr(mod, attr)
                        wrapper = _wrap_subprocess_call(orig)
                        setattr(mod, attr, wrapper)
                        patched_modules.append((mod, attr, orig))
                subm = getattr(mod, "subprocess", None)
                if subm is not None:
                    for attr in ("Popen", "run", "call", "check_call", "check_output"):
                        if hasattr(subm, attr):
                            orig = getattr(subm, attr)
                            wrapper = _wrap_subprocess_call(orig)
                            setattr(subm, attr, wrapper)
                            patched_modules.append((subm, attr, orig))
            except Exception:
                continue

        ocrmypdf.ocr(
            input_pdf,
            out_path,
            language=languages,
            force_ocr=force,
            skip_text=not force,
            optimize=optimize,
            deskew=deskew,
            remove_background=clean,
            progress_bar=False,
        )
    finally:
        try:
            subprocess.Popen = _orig["Popen"]  # type: ignore[assignment]
            if _orig["run"]:
                subprocess.run = _orig["run"]  # type: ignore[assignment]
            if _orig["call"]:
                subprocess.call = _orig["call"]  # type: ignore[assignment]
            if _orig["check_call"]:
                subprocess.check_call = _orig["check_call"]  # type: ignore[assignment]
            if _orig["check_output"]:
                subprocess.check_output = _orig["check_output"]  # type: ignore[assignment]
        except Exception:
            pass
        for target, attr, orig in patched_modules:
            try:
                setattr(target, attr, orig)
            except Exception:
                pass
    return out_path


def load_pdf_ocr_app():
    """Dynamically import the doctr-based OCR app module."""
    module_name = "pdf_ocr_app"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = Path(__file__).parent / "PDF" / "app.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Unable to locate pdf-ocr app at {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module_dir = str(module_path.parent)
    added_to_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_to_path = True

    try:
        spec.loader.exec_module(module)  # type: ignore[call-arg]
    finally:
        if added_to_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass

    return module


def run_pdf_ocr(input_file: str | Path, *, split_pages: bool = True) -> dict[str, Any]:
    """Run the doctr OCR pipeline, optionally splitting wide pages first."""
    source_pdf = Path(input_file)

    prepared_pdf: Path = source_pdf
    if split_pages:
        cache_dir = Path(__file__).resolve().parents[1] / "Cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        prepared_pdf_path = cache_dir / f"{source_pdf.stem}_split{source_pdf.suffix}"
        prepared_pdf = split_wide_pages(source_pdf, output_path=prepared_pdf_path)
    app = load_pdf_ocr_app()
    result = app.convert_pdf(prepared_pdf)
    print(f"OCR text saved to: {result['output_path']}")
    pdf_path = result.get("pdf_output_path")
    if pdf_path:
        print(f"OCR PDF saved to: {pdf_path}")
    return result


def classify_pdf(
    input_file: str | Path,
    *,
    total_char_threshold: int = 200,
    per_page_threshold: int = 30,
    ocr_char_threshold: int = 80,
    ocr_high_char_threshold: int = 1200,
    ocr_confidence_threshold: float = 0.75,
    sample_pages: int = 1,
) -> str:
    """Heuristically classify PDFs as digital or handwritten scans."""
    pdf_path = Path(input_file)
    reader = PdfReader(str(pdf_path))

    text_lengths: list[int] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_lengths.append(len(text.strip()))

    if not text_lengths:
        return "handwritten"

    total_text = sum(text_lengths)
    text_heavy_pages = sum(1 for length in text_lengths if length >= per_page_threshold)
    majority_threshold = max(1, len(text_lengths) // 2)

    if total_text >= total_char_threshold or text_heavy_pages >= majority_threshold:
        return "digital"

    # Fall back to running a lightweight doctr pass on the first page(s)
    try:
        if DoctrDocumentFile is None:
            raise ImportError("doctr is not available")
        app_module = load_pdf_ocr_app()
        get_model = getattr(app_module, "_get_ocr_model", None)
        if get_model is None:
            raise AttributeError("doctr model loader unavailable for classification")

        doc = DoctrDocumentFile.from_pdf(str(pdf_path))
        if sample_pages > 0 and len(doc) > sample_pages:
            doc = doc[:sample_pages]

        ocr_model = get_model()
        result = ocr_model(doc)

        recognised_chars = 0
        confidences: list[float] = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        word_text = getattr(word, "value", "")
                        recognised_chars += len(word_text.strip())
                        conf = getattr(word, "confidence", None)
                        if isinstance(conf, (int, float)):
                            confidences.append(float(conf))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if recognised_chars >= ocr_high_char_threshold:
            return "digital"
        if recognised_chars >= ocr_char_threshold and avg_confidence >= ocr_confidence_threshold:
            return "digital"
    except Exception:
        # If we cannot evaluate with doctr, fall through to handwritten classification.
        pass

    return "handwritten"

# from OCR.ocr import classify_pdf, run_pdf_ocr

# path = "HN.pdf"
# split_pages = True  # Toggle to False to skip splitting wide pages.

# run_pdf_ocr(path, split_pages=split_pages) if classify_pdf(path) == "digital" else print(
#     "Handwritten note detected – skipping OCR."
# )