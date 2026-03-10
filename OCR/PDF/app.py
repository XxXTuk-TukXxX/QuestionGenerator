import logging
import time
from pathlib import Path
import contextlib
import ssl
import argparse

import torch
from doctr.models import ocr_predictor

from OCR.PDF.pdf2text import convert_PDF_to_Text, rm_local_text_files

try:
    from OCR.ocr import run_ocr
except ModuleNotFoundError:
    import sys

    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from OCR.ocr import run_ocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# macOS site Python often lacks certs; force unverified context for model downloads.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


_here = Path(__file__).parent
_OCR_MODEL = None


def _get_ocr_model():
    global _OCR_MODEL
    if _OCR_MODEL is None:
        logging.info("Loading OCR model")
        with contextlib.redirect_stdout(None):
            _OCR_MODEL = ocr_predictor(
                "db_resnet50",
                "crnn_mobilenet_v3_large",
                pretrained=True,
                assume_straight_pages=True,
            )
    return _OCR_MODEL


def convert_pdf(pdf_path, output_path=None, max_pages=20, clear_temp=True):
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path}")

    if clear_temp:
        rm_local_text_files()

    ocr_model = _get_ocr_model()
    start = time.perf_counter()
    stats = convert_PDF_to_Text(pdf_path, ocr_model=ocr_model, max_pages=max_pages)
    runtime_minutes = round((time.perf_counter() - start) / 60, 2)

    destination = Path(output_path) if output_path else pdf_path.with_name(
        f"{pdf_path.stem}_OCR.txt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", errors="ignore") as f:
        f.write(stats["converted_text"])

    pdf_output = pdf_path.with_name(f"{pdf_path.stem}_OCR.pdf")
    ocrmypdf_result = run_ocr(
        str(pdf_path),
        output_pdf=str(pdf_output),
        languages="eng",
        force=False,
        deskew=True,
    )

    logging.info(
        "OCR complete: pages=%s truncated=%s runtime=%s minutes text=%s pdf=%s",
        stats["num_pages"],
        stats["truncated"],
        runtime_minutes,
        destination,
        ocrmypdf_result,
    )
    return {
        "output_path": str(destination),
        "pdf_output_path": ocrmypdf_result,
        "runtime_minutes": runtime_minutes,
        **stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PDF to text using doctr OCR")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=str((_here / "test_split.pdf").resolve()),
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=str((_here / "test_split_ocr.txt").resolve()),
        help="Destination file for the OCR output",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of PDF pages to OCR",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Skip clearing cached OCR text files before running",
    )

    args = parser.parse_args()
    logging.info("Using GPU status: %s", torch.cuda.is_available())
    result = convert_pdf(
        args.pdf_path,
        output_path=args.output_path,
        max_pages=args.max_pages,
        clear_temp=not args.keep_temp,
    )
    logging.info("Saved OCR output to %s", result["output_path"])
    logging.info("Saved OCR PDF to %s", result["pdf_output_path"])
