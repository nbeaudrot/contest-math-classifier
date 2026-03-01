#!/usr/bin/env python3
"""
Preprocess a Mathcounts chapter sprint PDF:
1. Remove the entire first page (cover/instructions).
2. Extract content from remaining pages excluding border and copyright footer.
3. Build Question objects, render figure regions to PNGs, and write question_texts.json.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pymupdf

# DPI for rendering figure regions to PNG
FIGURE_DPI = 150

# Margin to exclude page border (points) - same as extract_questions
PAGE_BORDER_MARGIN = 55
# Bottom margin to exclude copyright notice (points)
BOTTOM_COPYRIGHT_MARGIN = 55

# Pattern: "1. _____________" or "10.\t_____________"
ANSWER_BLANK_PATTERN = re.compile(r"^(\d+)\.\s+_+\s*$", re.MULTILINE)

# Minimum width (points) for a block to be treated as main question text (not figure label)
QUESTION_TEXT_MIN_WIDTH = 170
# Left margin: blocks with x0 >= this are in the "question column"
QUESTION_COLUMN_X0 = 150
# Tolerance for "inside blank bbox" (float comparison)
UNITS_BBOX_TOLERANCE = 0.5

# Superscript/subscript: PyMuPDF span flags (same as extract_questions.py)
SUPERSCRIPT_FLAG = 1   # bit 0
SUBSCRIPT_FLAG = 32    # bit 5

# Fraction detection: only treat as fraction when numerator/denominator are short (e.g. 1/2, 3/4).
FRACTION_MAX_NUMERATOR_LEN = 6   # max chars for numerator text
FRACTION_MAX_DENOMINATOR_LEN = 6 # max chars for denominator text
FRACTION_VERTICAL_GAP_MAX = 8    # pts; numerator span's y1 must be <= denominator's y0 + this
FRACTION_X_OVERLAP_MIN = 2       # pts; spans must overlap or be this close in x


@dataclass
class Question:
    """A single extracted question with optional diagram region and embedded images."""

    question_number: int
    page_number: int  # 0-based page index in the PDF
    text: str
    answer_units: Optional[str] = None
    text_bounding_box: Optional[pymupdf.Rect] = None
    figure_bounding_box: Optional[pymupdf.Rect] = None
    figure_image_filename: Optional[str] = None  # e.g. "question_4_figure.png" when saved


def rect_to_list(rect: pymupdf.Rect) -> list[float]:
    """Convert PyMuPDF Rect to a JSON-serializable list [x0, y0, x1, y1]."""
    return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]


def _spans_form_stacked_fraction(span_a: dict, span_b: dict) -> bool:
    """
    True if span_a is drawn above span_b with similar x and both are short (numerator/denominator).
    Only short spans (e.g. digits) are treated as fractions; paragraph lines are not.
    """
    text_a = (span_a.get("text") or "").strip()
    text_b = (span_b.get("text") or "").strip()
    if len(text_a) > FRACTION_MAX_NUMERATOR_LEN or len(text_b) > FRACTION_MAX_DENOMINATOR_LEN:
        return False
    bbox_a = span_a.get("bbox", (0, 0, 0, 0))
    bbox_b = span_b.get("bbox", (0, 0, 0, 0))
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return False
    x0_a, y0_a, x1_a, y1_a = bbox_a
    x0_b, y0_b, x1_b, y1_b = bbox_b
    # A is above B: A's bottom is at or above B's top (with small gap)
    if y1_a > y0_b + FRACTION_VERTICAL_GAP_MAX:
        return False
    # Similar x: overlap or nearly adjacent
    x_overlap = min(x1_a, x1_b) - max(x0_a, x0_b)
    return x_overlap >= -FRACTION_X_OVERLAP_MIN


def _block_spans_with_fractions(block: dict) -> list[tuple[str, int]]:
    """
    Flatten block to (text_segment, line_index) in reading order. Apply
    superscript (^) and subscript (_) from span flags. Merge any two
    consecutive spans that form a stacked fraction into \\frac{num}{denom}.
    """
    segments: list[tuple[str, dict, int]] = []  # (text, span, line_index)
    for line_i, line in enumerate(block.get("lines", [])):
        for span in line.get("spans", []):
            t = span.get("text", "")
            flags = span.get("flags", 0)
            if (
                segments
                and t
                and segments[-1][0].strip()
                and _spans_form_stacked_fraction(segments[-1][1], span)
            ):
                num_raw = segments[-1][1].get("text", "").strip()
                segments.pop()
                segments.append(
                    (f"\\frac{{{num_raw}}}{{{t.strip()}}}", span, line_i)
                )
            else:
                if flags & SUPERSCRIPT_FLAG:
                    t = f"^{t}"
                elif flags & SUBSCRIPT_FLAG:
                    t = f"_{t}_"
                segments.append((t, span, line_i))
    return [(seg[0], seg[2]) for seg in segments]


def block_text_with_notation(block: dict) -> str:
    """
    Extract text from a text block with superscripts as ^... (no trailing ^) and
    subscripts as _..._. Detects fractions: when two spans are vertically stacked
    (numerator above denominator by bbox), merge as \\frac{num}{denom}. Then
    convert unicode roots to fractional exponents (e.g. √x -> (x)^(1/2)).
    """
    if block.get("type") != 0:
        return ""
    segments_with_line = _block_spans_with_fractions(block)
    lines_out = []
    current = []
    prev_line = None
    for text, line_i in segments_with_line:
        if prev_line is not None and line_i != prev_line:
            lines_out.append("".join(current))
            current = []
        current.append(text)
        prev_line = line_i
    if current:
        lines_out.append("".join(current))
    text = "\n".join(lines_out)
    text = convert_roots_to_exponents(text)
    return clean_superscript_artifacts(text)


def convert_roots_to_exponents(text: str) -> str:
    """
    Convert unicode root symbols to fractional exponents.
    √x -> (x)^(1/2), ∛x -> (x)^(1/3), ∜x -> (x)^(1/4).
    Radicand is taken to be parenthesized expression first, else a single token.
    """
    def repl_sq(m):
        radicand = (m.group(2) or m.group(1) or "").strip()
        return f"({radicand})^(1/2)" if radicand else "^(1/2)"
    def repl_cb(m):
        radicand = (m.group(2) or m.group(1) or "").strip()
        return f"({radicand})^(1/3)" if radicand else "^(1/3)"
    def repl_four(m):
        radicand = (m.group(2) or m.group(1) or "").strip()
        return f"({radicand})^(1/4)" if radicand else "^(1/4)"
    # Square root: √(expr) or √token
    text = re.sub(r"√\s*\(([^)]+)\)|√\s*([^\s^\[\]{}()]+)", repl_sq, text)
    text = re.sub(r"∛\s*\(([^)]+)\)|∛\s*([^\s^\[\]{}()]+)", repl_cb, text)
    text = re.sub(r"∜\s*\(([^)]+)\)|∜\s*([^\s^\[\]{}()]+)", repl_four, text)
    return text


def clean_superscript_artifacts(text: str) -> str:
    """
    Fix spurious ^ from PDF superscript styling that is not a real exponent.
    - ^y^ or ^xy^ (letters only in carets) -> y, xy (variable/product, not exponent).
    - Lone ^ after \\frac{...} or space -> remove (stray superscript span).
    Keeps real exponents: cm^2, m^2, x^2, ^(1/2), etc. (no closing ^ or digit after ^).
    """
    # Unwrap carets around letters only: ^y^ -> y, 12^xy^ -> 12xy
    text = re.sub(r"\^([a-zA-Z]+)\^", r"\1", text)
    # Remove lone ^ (space-caret-space or } ^)
    text = re.sub(r" \^ ", " ", text)
    text = re.sub(r"\} \^", "}", text)
    # Collapse any double space left after removals
    text = re.sub(r"  +", " ", text)
    return text


def block_to_dict(block: dict) -> dict:
    """Convert a get_text('dict') block to a JSON-serializable dict."""
    result = {
        "type": block.get("type", 0),
        "bbox": rect_to_list(pymupdf.Rect(block["bbox"])),
    }
    if result["type"] == 0:  # text block
        lines = []
        for line in block.get("lines", []):
            line_text_parts = []
            for span in line.get("spans", []):
                line_text_parts.append(span.get("text", ""))
            lines.append("".join(line_text_parts))
        result["lines"] = lines
        result["text"] = "\n".join(lines)
    return result


def extract_questions_from_page(
    blocks_raw: list[dict], page_index: int, content_rect: pymupdf.Rect
) -> dict[int, Question]:
    """
    Build a dict of question_number -> Question for all questions on the page.
    Uses answer-blank blocks to find question numbers; assigns text/units/figure by position.
    """
    # 1) Find answer-blank blocks: "N. _____________"
    blanks: list[tuple[int, pymupdf.Rect]] = []
    for block in blocks_raw:
        if block.get("type") != 0:
            continue
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "")
        m = ANSWER_BLANK_PATTERN.match(text.strip())
        if m:
            qnum = int(m.group(1))
            bbox = pymupdf.Rect(block["bbox"])
            blanks.append((qnum, bbox))
    blanks.sort(key=lambda x: x[1].y0)

    # 2) For each consecutive pair of blanks, assign blocks to that question
    result: dict[int, Question] = {}
    for i, (qnum, blank_rect) in enumerate(blanks):
        y_blank = blank_rect.y0
        y_next = blanks[i + 1][1].y0 if i + 1 < len(blanks) else content_rect.y1

        # Units: short text block whose bbox is inside blank's x and y close to blank
        answer_units: Optional[str] = None
        question_text_parts: list[tuple[float, str, pymupdf.Rect]] = []  # (y0, text, bbox) for ordering
        figure_rects: list[pymupdf.Rect] = []

        for block in blocks_raw:
            bbox = pymupdf.Rect(block["bbox"])
            # Only consider blocks in this question's vertical band (by vertical center)
            mid_y = (bbox.y0 + bbox.y1) / 2
            if mid_y <= y_blank or mid_y >= y_next:
                continue
            block_type = block.get("type", 0)

            # Skip the blank block itself
            if block_type == 0:
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                if ANSWER_BLANK_PATTERN.match(text.strip()):
                    continue

            # Units: short text, x inside blank bbox, y very close to blank line
            if block_type == 0:
                line_text = block_text_with_notation(block).strip()
                if len(line_text) < 25 and line_text and not line_text.isdigit():
                    if (
                        blank_rect.x0 <= bbox.x0 + UNITS_BBOX_TOLERANCE
                        and bbox.x1 <= blank_rect.x1 + UNITS_BBOX_TOLERANCE
                        and abs(bbox.y0 - blank_rect.y0) < 25
                    ):
                        if answer_units is None:
                            answer_units = line_text
                            continue

            # Question text: type 0, in right column, wide enough to be paragraph (not table cell)
            if block_type == 0:
                width = bbox.x1 - bbox.x0
                if bbox.x0 >= QUESTION_COLUMN_X0 and width >= QUESTION_TEXT_MIN_WIDTH:
                    question_text_parts.append((bbox.y0, block_text_with_notation(block), bbox))
                    continue

            # Everything else in the band (images, labels, table cells) -> figure
            if block_type == 1 or (block_type == 0 and (bbox.x0 < QUESTION_COLUMN_X0 or (bbox.x1 - bbox.x0) < QUESTION_TEXT_MIN_WIDTH)):
                figure_rects.append(bbox)

        question_text_parts.sort(key=lambda x: x[0])
        text_str = " ".join(
            s.strip() for _, s, _ in question_text_parts if s.strip()
        ).strip()
        # Remove all newlines (replace with space and collapse runs of spaces)
        text_str = " ".join(text_str.replace("\n", " ").split())
        text_bbox: Optional[pymupdf.Rect] = None
        if question_text_parts:
            text_bbox = question_text_parts[0][2]
            for _, _, r in question_text_parts[1:]:
                text_bbox = text_bbox | r
        figure_bbox: Optional[pymupdf.Rect] = None
        if figure_rects:
            figure_bbox = figure_rects[0]
            for r in figure_rects[1:]:
                figure_bbox = figure_bbox | r

        result[qnum] = Question(
            question_number=qnum,
            page_number=page_index,
            text=text_str,
            answer_units=answer_units,
            text_bounding_box=text_bbox,
            figure_bounding_box=figure_bbox,
        )

    return result


def preprocess_pdf(path: Path, output_dir: Optional[Path] = None) -> list[Question]:
    """
    Open PDF, skip first page, extract content from remaining pages
    inside content rect (no border, no copyright footer). If output_dir is set,
    render each question's figure region to PNG (question_n_figure.png).
    Return a list of Question objects for all 30 questions.
    """
    doc = pymupdf.open(path)
    try:
        all_questions: dict[int, Question] = {}
        # Page 0 is the cover; process from page 1 onward
        for page_index in range(1, len(doc)):
            page = doc[page_index]
            rect = page.rect
            width = rect.width
            height = rect.height

            # Content area: strip border and bottom copyright
            content_rect = pymupdf.Rect(
                PAGE_BORDER_MARGIN,
                PAGE_BORDER_MARGIN,
                width - PAGE_BORDER_MARGIN,
                height - BOTTOM_COPYRIGHT_MARGIN,
            )

            try:
                blocks_raw = page.get_text(
                    "dict", clip=content_rect, flags=pymupdf.TEXT_PRESERVE_WHITESPACE
                )["blocks"]
            except TypeError:
                blocks_raw = page.get_text("dict", clip=content_rect)["blocks"]

            page_questions = extract_questions_from_page(
                blocks_raw, page_index, content_rect
            )

            # Render figure region to PNG when requested
            if output_dir is not None:
                for qnum, q in page_questions.items():
                    if q.figure_bounding_box is not None:
                        clip = q.figure_bounding_box
                        if clip.width > 0 and clip.height > 0:
                            pix = page.get_pixmap(dpi=FIGURE_DPI, clip=clip)
                            filename = f"question_{qnum}_figure.png"
                            pix.save(str(output_dir / filename))
                            page_questions[qnum] = replace(
                                q, figure_image_filename=filename
                            )

            all_questions.update(page_questions)

        return [all_questions[n] for n in sorted(all_questions)]
    finally:
        doc.close()


def question_to_dict(q: Question) -> dict:
    """Convert a Question to a JSON-serializable dict."""
    d: dict = {
        "question_number": q.question_number,
        "page_number": q.page_number,
        "text": q.text,
        "answer_units": q.answer_units,
    }
    if q.text_bounding_box is not None:
        d["text_bounding_box"] = rect_to_list(q.text_bounding_box)
    if q.figure_bounding_box is not None:
        d["figure_bounding_box"] = rect_to_list(q.figure_bounding_box)
    if q.figure_image_filename is not None:
        d["figure_image_filename"] = q.figure_image_filename
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a Mathcounts chapter sprint PDF into Question objects and figure PNGs."
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for question_texts.json and question_n_figure.png files",
    )
    args = parser.parse_args()
    path = args.pdf
    output_dir = args.output_dir

    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = preprocess_pdf(path, output_dir=output_dir)
    payload = [question_to_dict(q) for q in questions]
    out_json = output_dir / "question_texts.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json} and {sum(1 for q in questions if q.figure_image_filename)} figure image(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
