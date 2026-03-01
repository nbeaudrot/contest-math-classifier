#!/usr/bin/env python3
"""
Diagnostic script to inspect PDF structure: text blocks, images, and vector graphics
and their y-coordinates. Use this to understand where diagrams sit relative to question text.
"""

import re
import sys
from pathlib import Path

import pymupdf

# Same constants as extract_questions
DIAGRAM_QUESTIONS = {4, 7, 12, 19, 20, 23, 25, 28}
PAGE_BORDER_MARGIN = 55
QUESTION_PATTERN = re.compile(r"^(\d+)[.)\]:]\s+", re.MULTILINE)


def inspect_page(doc: pymupdf.Document, page_num: int, page_label: str = "") -> None:
    """Dump structure of a single page: text blocks, images, drawings."""
    page = doc[page_num]
    rect = page.rect
    print(f"\n{'='*60}")
    print(f"PAGE {page_num + 1} (0-based: {page_num}) {page_label}")
    print(f"  Page size: {rect.width:.1f} x {rect.height:.1f} pts")
    print(f"  Margin (excluded): {PAGE_BORDER_MARGIN} pts from edges")

    # 1. Text blocks from get_text("dict")
    try:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except TypeError:
        blocks = page.get_text("dict")["blocks"]

    print("\n  --- TEXT BLOCKS (from get_text dict) ---")
    for i, block in enumerate(blocks):
        bbox = block.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = bbox
        if block.get("type") == 1:
            print(f"    [{i}] type=image  bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})  y0={y0:.1f} y1={y1:.1f}")
        else:
            lines = block.get("lines", [])
            line_count = len(lines)
            first_line = ""
            if lines and lines[0].get("spans"):
                first_line = (lines[0]["spans"][0].get("text", ""))[:50].encode("ascii", errors="replace").decode()
            print(f"    [{i}] type=text   bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})  y0={y0:.1f} y1={y1:.1f}  lines={line_count}  \"{first_line}...\"")

    # 2. Images from get_images() + get_image_rects()
    print("\n  --- EMBEDDED IMAGES (get_images + get_image_rects) ---")
    seen_xrefs = set()
    for img in page.get_images():
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            rects = page.get_image_rects(xref) if hasattr(page, "get_image_rects") else []
            for j, r in enumerate(rects):
                if hasattr(r, "y0"):
                    y0, y1 = r.y0, r.y1
                else:
                    y0, y1 = r[1], r[3]
                w = r.x1 - r.x0 if hasattr(r, "x0") else r[2] - r[0]
                h = y1 - y0
                print(f"    xref={xref} rect[{j}]: y0={y0:.1f} y1={y1:.1f}  size={w:.1f}x{h:.1f}")
        except Exception as e:
            print(f"    xref={xref} error: {e}")

    # 3. Vector graphics from get_drawings()
    print("\n  --- VECTOR GRAPHICS (get_drawings) ---")
    try:
        drawings = page.get_drawings()
        for i, d in enumerate(drawings):
            r = d.get("rect")
            if r is not None:
                y0 = r.y0 if hasattr(r, "y0") else r[1]
                y1 = r.y1 if hasattr(r, "y1") else r[3]
                items = d.get("items", [])
                n_items = len(items)
                print(f"    [{i}] rect y0={y0:.1f} y1={y1:.1f}  items={n_items}")
    except Exception as e:
        print(f"    (get_drawings failed: {e})")

    # 4. cluster_drawings (aggregated vector graphics bboxes)
    print("\n  --- CLUSTER DRAWINGS (vector graphics bboxes) ---")
    try:
        clusters = page.cluster_drawings() if hasattr(page, "cluster_drawings") else []
        for i, r in enumerate(clusters):
            y0 = r.y0 if hasattr(r, "y0") else r[1]
            y1 = r.y1 if hasattr(r, "y1") else r[3]
            print(f"    [{i}] y0={y0:.1f} y1={y1:.1f}")
    except Exception as e:
        print(f"    (cluster_drawings failed: {e})")


def main():
    pdf_path = Path("test-assets/2020_mathcounts_chapter_sprint.pdf")
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    doc = pymupdf.open(pdf_path)
    # Skip page 0 (title)
    for page_num in range(1, len(doc)):
        page = doc[page_num]
        # Build question-to-y mapping from text
        try:
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except TypeError:
            blocks = page.get_text("dict")["blocks"]

        question_blocks = []
        for block in blocks:
            if block.get("type") == 1:
                continue
            text_parts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
            text = "".join(text_parts)
            m = QUESTION_PATTERN.search(text)
            if m:
                qnum = int(m.group(1))
                bbox = block.get("bbox", (0, 0, 0, 0))
                question_blocks.append((qnum, bbox[1], bbox[3]))

        label = ""
        if question_blocks:
            qnums = [q[0] for q in question_blocks]
            diag = [q for q in qnums if q in DIAGRAM_QUESTIONS]
            if diag:
                label = f"[diagram questions: {diag}]"
            else:
                label = f"[questions: {qnums[0]}..{qnums[-1]}]"

        inspect_page(doc, page_num, label)

    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
