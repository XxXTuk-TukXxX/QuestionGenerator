"""PDF utilities for preparing documents before OCR."""

from __future__ import annotations

import copy
import pathlib

from pypdf import PdfReader, PdfWriter, Transformation

PathLike = str | pathlib.Path


def split_wide_pages(
    pdf_path: PathLike,
    output_path: PathLike | None = None,
    *,
    ratio_threshold: float = 1.25,
) -> pathlib.Path:
    """Split landscape/double-page spreads into individual portrait pages."""

    src = pathlib.Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"PDF does not exist: {src}")

    reader = PdfReader(str(src))
    writer = PdfWriter()

    split_occurred = False

    for page in reader.pages:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if height == 0:
            continue
        if width / height >= ratio_threshold:
            split_occurred = True
            half_width = width / 2
            left_page = writer.add_blank_page(width=half_width, height=height)
            left_page.merge_transformed_page(
                copy.deepcopy(page),
                Transformation().translate(-float(page.mediabox.left), 0),
            )

            right_page = writer.add_blank_page(width=half_width, height=height)
            right_page.merge_transformed_page(
                copy.deepcopy(page),
                Transformation().translate(-(float(page.mediabox.left) + half_width), 0),
            )
        else:
            writer.add_page(copy.deepcopy(page))

    if split_occurred or output_path is not None:
        destination = pathlib.Path(output_path) if output_path else src.with_name(src.stem + "_split" + src.suffix)
        with destination.open("wb") as fh:
            writer.write(fh)
        return destination.resolve()

    return src.resolve()


__all__ = ["split_wide_pages"]
