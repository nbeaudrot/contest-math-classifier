#!/usr/bin/env python3
"""
Extract numbered math problems from a PDF file, along with any associated diagrams.
Outputs each question to a separate text file with LLM-friendly math notation,
and saves diagram images as PNG files.
"""

import argparse
import re
from pathlib import Path

import pymupdf


# Superscript flag in PyMuPDF span flags (bit 0)
SUPERSCRIPT_FLAG = 1
# Subscript flag (bit 5)
SUBSCRIPT_FLAG = 32


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
    Returns list of (y0, "text", text_str) or (y0, "image", (bytes, ext)).
    """
    items = []
    try:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except TypeError:
        blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        bbox = block.get("bbox", (0, 0, 0, 0))
        y0 = bbox[1]
        if block.get("type") == 1:
            img_data = block.get("image")
            if img_data:
                ext = block.get("ext", "png")
                items.append((y0, "image", (img_data, ext)))
        else:
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
                items.append((y0, "text", text))

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
            items.append((y0, "image", (img_info["image"], img_info["ext"])))
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


def extract_questions_from_pdf(pdf_path: Path, output_dir: Path) -> None:
    """Main extraction logic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)

    # Build full text and ordered list of (page, y0, type, content)
    all_items = []
    full_text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for y0, item_type, content in get_page_items(page, doc, page_num):
            all_items.append((page_num, y0, item_type, content))
            if item_type == "text":
                full_text_parts.append(content)

    full_text = "\n".join(full_text_parts)
    question_splits = find_question_splits(full_text)

    # Extract question text
    questions_text = {qnum: full_text[start:end] for qnum, start, end in question_splits}

    # Associate images with questions: process items in order, track text position
    q_images = {qnum: [] for qnum, _, _ in question_splits}
    current_q = question_splits[0][0] if question_splits else 1
    text_pos = 0
    seen_image_hashes = set()

    for page_num, y0, item_type, content in all_items:
        if item_type == "text":
            for qnum, start, end in question_splits:
                if start <= text_pos < end:
                    current_q = qnum
                    break
            text_pos += len(content) + 1
        elif item_type == "image":
            img_data, ext = content
            # Deduplicate by content hash (same image may appear in dict and get_images)
            h = hash(img_data) if len(img_data) < 10000 else hash(img_data[:1000])
            if h in seen_image_hashes:
                continue
            seen_image_hashes.add(h)
            if current_q in q_images:
                q_images[current_q].append((img_data, ext))

    # Write output files
    for qnum, start, end in question_splits:
        text = transform_math_for_llm(questions_text[qnum])
        qname = f"question_{qnum:03d}"
        txt_path = output_dir / f"{qname}.txt"
        txt_path.write_text(text, encoding="utf-8")
        print(f"Wrote {txt_path}")

        for i, (img_data, ext) in enumerate(q_images.get(qnum, [])):
            out_ext = ext if ext in ("png", "jpg", "jpeg") else "png"
            img_path = output_dir / f"{qname}_diagram_{i+1:02d}.{out_ext}"
            img_path.write_bytes(img_data)
            print(f"  Wrote diagram {img_path}")

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
        required=True,
        help="Output directory for extracted files",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: PDF file not found: {args.pdf}")
        return 1

    extract_questions_from_pdf(args.pdf, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
