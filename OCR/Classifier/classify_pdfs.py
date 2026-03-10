from __future__ import annotations

import argparse
from pathlib import Path

from OCR.ocr import classify_pdf, run_pdf_ocr


def process_documents(pdf_paths: list[str | Path]) -> tuple[list[Path], list[Path]]:
    """Classify each PDF and run OCR on digital documents."""
    digital_docs: list[Path] = []
    handwritten_docs: list[Path] = []

    for pdf in pdf_paths:
        pdf_path = Path(pdf).resolve()
        if not pdf_path.is_file():
            print(f"Skipping missing file: {pdf_path}")
            continue

        classification = classify_pdf(pdf_path)
        if classification == "digital":
            print(f"[digital] {pdf_path.name}")
            digital_docs.append(pdf_path)
            run_pdf_ocr(pdf_path)
        else:
            print(f"[handwritten] {pdf_path.name} – no action taken")
            handwritten_docs.append(pdf_path)

    return digital_docs, handwritten_docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify PDFs as digital or handwritten and OCR the digital ones.",
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="One or more PDF files to classify.",
    )
    args = parser.parse_args()

    digital, handwritten = process_documents(args.pdfs)
    print("\nSummary:")
    print(f"  Digital PDFs processed: {len(digital)}")
    print(f"  Handwritten notes skipped: {len(handwritten)}")


if __name__ == "__main__":
    main()
