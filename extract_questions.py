#!/usr/bin/env python3
"""
Extract numbered math problems from a PDF file, along with any associated diagrams.
Outputs each question to a separate text file with LLM-friendly math notation,
and saves diagram images as PNG files.
"""

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf


@dataclass
class Question:
    """A single extracted question with optional diagram region and embedded images."""

    number: int
    page_number: int  # 0-based page index in the PDF
    text: str
    y0: float
    y1: float
    image_rects: Optional[list[tuple[int, pymupdf.Rect]]] = None  # (page_num, rect) per diagram
    images: list[tuple[bytes, str]] = field(default_factory=list)  # (bytes, ext) embedded images


# Superscript flag in PyMuPDF span flags (bit 0)
SUPERSCRIPT_FLAG = 1
# Subscript flag (bit 5)
SUBSCRIPT_FLAG = 32

# Questions that have diagrams (vector graphics) that must be rendered
DIAGRAM_QUESTIONS = {4, 7, 12, 19, 20, 23, 25, 28}
# Padding around diagram bbox to include nearby labels (in points)
DIAGRAM_PADDING = 8
# Margin to exclude page border (in points)
PAGE_BORDER_MARGIN = 55


def transform_math_for_llm(text: str) -> str:
    """
    Transform math notation for LLM consumption.
    - Normalize whitespace (multiple spaces/newlines -> single space)
    - Ensure proper spacing around ^ for exponents
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\^\s*', '^', text)
    text = re.sub(r'\^(\d+)([a-zA-Z])', r'^\1 \2', text)
    return text.strip()


def extract_text_with_notation(page: pymupdf.Page) -> str:
    """
    Extract text from a page, converting superscripts to ^ and subscripts to _.
    """
    parts = []
    try:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except TypeError:
        blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if block.get("type") == 1:
            continue
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                t = span.get("text", "")
                flags = span.get("flags", 0)
                if flags & SUPERSCRIPT_FLAG:
                    line_parts.append(f"^{t}^")
                elif flags & SUBSCRIPT_FLAG:
                    line_parts.append(f"_{t}_")
                else:
                    line_parts.append(t)
            if line_parts:
                parts.append("".join(line_parts))
    return "\n".join(parts)


def get_page_items(page: pymupdf.Page, doc: pymupdf.Document, page_num: int) -> list:
    """
    Get all items (text and images) from a page in reading order.
    Returns list of (y0, y1, "text", text_str) or (y0, y1, "image", (bytes, ext)).
    """
    items = []
    try:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except TypeError:
        blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if block.get("type") == 1:
            bbox = block.get("bbox", (0, 0, 0, 0))
            x0, y0, x1, y1 = bbox
            # skip image blocks that are part of the border
            if x0 < PAGE_BORDER_MARGIN or x0 > doc[page_num].rect.width - PAGE_BORDER_MARGIN:
                logging.debug(f"Skipping image block at x0={x0}, y0={y0}, x1={x1}, y1={y1} because it is part of the border")
                continue
            if y0 < PAGE_BORDER_MARGIN or y0 > doc[page_num].rect.height - PAGE_BORDER_MARGIN:
                logging.debug(f"Skipping image block at x0={x0}, y0={y0}, x1={x1}, y1={y1} because it is part of the border")
                continue
            img_data = block.get("image")
            if img_data:
                ext = block.get("ext", "png")
                items.append((y0, y1, "image", (img_data, ext)))
        elif block.get("type") == 0:
            # skip text blocks that are part of the footer
            bbox = block.get("bbox", (0, 0, 0, 0))
            x0, y0, x1, y1 = bbox # might not need all of these ...
            if y1 > doc[page_num].rect.height - PAGE_BORDER_MARGIN:
                logging.debug(f"Skipping text block at y1={y1} because it is part of the footer")
                continue
            text_parts = []
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    flags = span.get("flags", 0)
                    if flags & SUPERSCRIPT_FLAG:
                        line_parts.append(f"^{t}^")
                    elif flags & SUBSCRIPT_FLAG:
                        line_parts.append(f"_{t}_")
                    else:
                        line_parts.append(t)
                text_parts.append("".join(line_parts))
            text = "\n".join(text_parts)
            if text.strip():
                items.append((y0, y1, "text", text))
        else:
            pass

    # Also get images via get_images() - dict blocks may not include all
    seen_xrefs = set()
    for img in page.get_images():
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            img_info = doc.extract_image(xref)
            rects = page.get_image_rects(xref) if hasattr(page, "get_image_rects") else []
            y0 = rects[0][1] if rects else 0
            y1 = rects[0][3] if rects else y0 + 50
            items.append((y0, y1, "image", (img_info["image"], img_info["ext"])))
        except Exception:
            pass

    items.sort(key=lambda x: x[0])
    return items


QUESTION_PATTERN = re.compile(r'^(\d+)[.)\]:]\s+', re.MULTILINE)


def find_question_splits(full_text: str) -> list[tuple[int, int, int]]:
    """Find (question_num, start_pos, end_pos) for each question."""
    matches = list(QUESTION_PATTERN.finditer(full_text))
    if not matches:
        return [(1, 0, len(full_text))]
    result = []
    for i, m in enumerate(matches):
        qnum = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        result.append((qnum, start, end))
    return result


# Full-page frame drawings (e.g. page border) to exclude from diagram detection
FULL_PAGE_FRAME_HEIGHT = 700


def _compute_diagram_rects(
    qnum: int,
    question_rects: dict[int, list[tuple[int, float, float]]],
    qnums_sorted: list[int],
    doc: pymupdf.Document,
) -> Optional[list[tuple[int, pymupdf.Rect]]]:
    """
    Compute diagram region for a question if it is in DIAGRAM_QUESTIONS.
    Uses vector graphics (get_drawings) to find diagram bounding boxes between
    this question's text and the next question's y0. Falls back to geometric
    heuristic for diagram-as-table (e.g. Q4).
    """
    if qnum not in DIAGRAM_QUESTIONS:
        return None
    rects = question_rects.get(qnum, [])
    if not rects:
        return None
    page_num = rects[0][0]
    q_y0 = min(r[1] for r in rects)
    q_y1 = max(r[2] for r in rects)

    idx = qnums_sorted.index(qnum)
    next_on_page = None
    for nq in qnums_sorted[idx + 1:]:
        nrects = question_rects.get(nq, [])
        if nrects and any(r[0] == page_num for r in nrects):
            next_on_page = nq
            break
    if next_on_page:
        nr = [r for r in question_rects[next_on_page] if r[0] == page_num]
        next_y0 = min(r[1] for r in nr)
        band_y1 = next_y0 - 5
    else:
        band_y1 = doc[page_num].rect.height - PAGE_BORDER_MARGIN

    margin = PAGE_BORDER_MARGIN
    band_y0 = max(margin, q_y1 - 20)  # allow slight overlap with question text
    page = doc[page_num]
    page_height = page.rect.height

    # Find vector graphics in the band between question and next question
    diagram_rect = None
    try:
        drawings = page.get_drawings()
        relevant_rects = []
        for d in drawings:
            r = d.get("rect")
            if r is None:
                continue
            dy0 = r.y0 if hasattr(r, "y0") else r[1]
            dy1 = r.y1 if hasattr(r, "y1") else r[3]
            height = dy1 - dy0
            # Exclude full-page frames
            if height > FULL_PAGE_FRAME_HEIGHT:
                continue
            # Must overlap the band
            if dy1 < band_y0 or dy0 > band_y1:
                continue
            relevant_rects.append(pymupdf.Rect(r))
        if relevant_rects:
            diagram_rect = relevant_rects[0]
            for r in relevant_rects[1:]:
                diagram_rect |= r
            diagram_rect = pymupdf.Rect(
                max(page.rect.x0 + margin, diagram_rect.x0 - DIAGRAM_PADDING),
                max(margin, diagram_rect.y0 - DIAGRAM_PADDING),
                min(page.rect.x1 - margin, diagram_rect.x1 + DIAGRAM_PADDING),
                min(page_height - margin, diagram_rect.y1 + DIAGRAM_PADDING),
            )
    except Exception:
        pass

    # Fallback: geometric heuristic (e.g. for diagram-as-table like Q4)
    if diagram_rect is None or diagram_rect.width <= 0 or diagram_rect.height <= 0:
        diagram_y0 = max(margin, q_y1 + 5)
        diagram_y1 = band_y1
        min_height = 30
        if diagram_y1 <= diagram_y0:
            diagram_y1 = min(q_y1 + 120, page_height - margin)
        if diagram_y1 - diagram_y0 < min_height:
            diagram_y1 = diagram_y0 + min(150, page_height - margin - diagram_y0)
        diagram_y1 = max(diagram_y1, diagram_y0 + min_height)
        diagram_rect = pymupdf.Rect(
            page.rect.x0 + margin, diagram_y0,
            page.rect.x1 - margin, diagram_y1
        )

    if diagram_rect.width <= 0 or diagram_rect.height <= 0:
        return None
    return [(page_num, diagram_rect)]


def _print_questions_summary(questions: list[Question]) -> None:
    """Print a summary of each question: number, page, y0/y1, and image rect if present."""
    for q in questions:
        page_display = q.page_number + 1  # 1-based for user
        line = f"Question {q.number}: page {page_display}, y0={q.y0:.1f}, y1={q.y1:.1f}"
        if q.image_rects:
            _page_num, rect = q.image_rects[0]
            line += f", image_rect=({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f})"
        print(line)

def extract_questions_from_page(doc: pymupdf.Document, page_num: int) -> list[Question]:
    """Extract questions from a single page."""
    page = doc[page_num]
    full_text_parts = []
    all_items = []
    for y0, y1, item_type, content in get_page_items(page, doc, page_num):
        all_items.append((page_num, y0, y1, item_type, content))
        if item_type == "text":
            full_text_parts.append(content)
    full_text = "\n".join(full_text_parts)
    question_splits = find_question_splits(full_text)
    questions_text = {qnum: full_text[start:end] for qnum, start, end in question_splits}


    # Single pass: assign text blocks and images to questions by current text position
    question_rects = {qnum: [] for qnum, _, _ in question_splits}
    q_images = {qnum: [] for qnum, _, _ in question_splits}
    current_q = question_splits[0][0] if question_splits else 1
    text_pos = 0
    seen_image_hashes: set[int] = set()

    for page_num, y0, y1, item_type, content in all_items:
        if item_type == "text":
            for qnum, start, end in question_splits:
                if start <= text_pos < end:
                    current_q = qnum
                    break
            question_rects[current_q].append((page_num, y0, y1))
            text_pos += len(content) + 1
        elif item_type == "image":
            img_data, ext = content
            h = hash(img_data) if len(img_data) < 10000 else hash(img_data[:1000])
            if h not in seen_image_hashes:
                seen_image_hashes.add(h)
                if current_q in q_images:
                    q_images[current_q].append((img_data, ext))

    # Build each question as we go: compute diagram rect for this question, then create Question
    qnums_sorted = [qnum for qnum, _, _ in question_splits]
    result: list[Question] = []
    for qnum, start, end in question_splits:
        rects = question_rects.get(qnum, [])
        page_num = rects[0][0] if rects else 0
        y0 = min(r[1] for r in rects) if rects else 0.0
        y1 = max(r[2] for r in rects) if rects else 0.0
        image_rects = _compute_diagram_rects(qnum, question_rects, qnums_sorted, doc)
        if image_rects:
            d_y0 = min(r.y0 for _, r in image_rects)
            d_y1 = max(r.y1 for _, r in image_rects)
            y0 = min(y0, d_y0)
            y1 = max(y1, d_y1)
        result.append(
            Question(
                number=qnum,
                page_number=page_num,
                text=transform_math_for_llm(questions_text[qnum]),
                y0=y0,
                y1=y1,
                image_rects=image_rects,
                images=q_images.get(qnum, []),
            )
        )

    # Validate top-to-bottom order on this page: each question's y0 must be >= previous question's y1
    """
    for i in range(1, len(result)):
        prev_y1 = result[i - 1].y1
        curr_y0 = result[i].y0
        if curr_y0 < prev_y1:
            raise ValueError(
                f"Page {page_num + 1}: question {result[i].number} y0={curr_y0} is less than "
                f"previous question {result[i - 1].number} y1={prev_y1}; questions must be ordered top-to-bottom."
            )
    """
    return result

def extract_questions_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    summary: bool = False,
) -> None:
    """Main extraction logic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)

    # Build full text and ordered list of (page, y0, y1, type, content)
    # Skip page 0 (title page)
    all_items = []
    full_text_parts = []
    questions: list[Question] = []
    for page_num in range(1, len(doc)):
        this_page_questions = extract_questions_from_page(doc, page_num)
        questions.extend(this_page_questions)

    # Collect all output paths we would write
    output_paths = []
    for q in questions:
        qname = f"question_{q.number:03d}"
        output_paths.append(output_dir / f"{qname}.txt")
        for i in range(len(q.images)):
            img_data, ext = q.images[i]
            out_ext = ext if ext in ("png", "jpg", "jpeg") else "png"
            output_paths.append(output_dir / f"{qname}_diagram_{i+1:02d}.{out_ext}")
        if q.image_rects:
            output_paths.append(output_dir / f"{qname}.png")

    if not overwrite:
        existing = [p for p in output_paths if p.exists()]
        if existing:
            logging.error("Error: the following output files already exist (use --overwrite to overwrite):", file=sys.stderr)
            for p in existing:
                logging.error(f"  {p}", file=sys.stderr)
            raise SystemExit(1)

    if summary:
        _print_questions_summary(questions)

    # Write output files
    for q in questions:
        qname = f"question_{q.number:03d}"
        txt_path = output_dir / f"{qname}.txt"
        txt_path.write_text(q.text, encoding="utf-8")
        logging.debug(f"Wrote {txt_path}")

        for i, (img_data, ext) in enumerate(q.images):
            out_ext = ext if ext in ("png", "jpg", "jpeg") else "png"
            img_path = output_dir / f"{qname}_diagram_{i+1:02d}.{out_ext}"
            img_path.write_bytes(img_data)
            logging.debug(f"  Wrote diagram {img_path}")

        if q.image_rects:
            page_num, diagram_rect = q.image_rects[0]
            diagram_rect = pymupdf.Rect(diagram_rect)  # copy before merging
            for _, r in q.image_rects[1:]:
                diagram_rect |= r
            if diagram_rect.width > 0 and diagram_rect.height > 0:
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150, clip=diagram_rect)
                diagram_path = output_dir / f"{qname}.png"
                pix.save(str(diagram_path))
                logging.debug(f"  Wrote diagram {diagram_path}")

    doc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract math problems and diagrams from a PDF file."
    )
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=Path("test-assets/2020_mathcounts_chapter_sprint.pdf"),
        help="Path to the PDF file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("test-output/extracted_questions"),
        help="Output directory for extracted files",
    )
    parser.add_argument(
        "-w", "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files if present (default: do not overwrite)",
    )
    parser.add_argument(
        "-r", "--clean",
        default=False,
        action="store_true",
        help="Clean output directory before extracting questions",
    )
    parser.add_argument(
        "-s", "--summary",
        default=False,
        action="store_true",
        help="Print a summary of the extracted questions",
    )
    parser.add_argument(
        "-l", "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
    )
    args = parser.parse_args()

    level = getattr(logging, args.loglevel)
    logging.basicConfig(level=level)

    if not args.pdf.exists():
        print(f"Error: PDF file not found: {args.pdf}")
        return 1

    if args.clean:
        if args.output.exists():
            for p in args.output.glob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)

    extract_questions_from_pdf(
        args.pdf,
        args.output,
        overwrite=args.overwrite,
        summary=args.summary,
    )
    return 0


if __name__ == "__main__":
    exit(main())
