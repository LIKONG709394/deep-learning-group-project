"""
report_pdf.py — Build teaching feedback PDF for Catcher
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def build_teaching_feedback_pdf(
    output_path: str,
    board_lines: List[str],
    clarity: Dict[str, Any],
    alignment: Dict[str, Any],
    speech_text: str,
    module_errors: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create final PDF report.

    Returns output_path.
    """
    register_cjk_font()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4

    font_name = "STSong-Light"
    title_size = 18
    heading_size = 12
    body_size = 10
    line_gap = 14

    y = h - 48

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(font_name, body_size)
        y = h - 48

    def ensure_space(lines_needed: int = 1):
        nonlocal y
        if y < 60 + lines_needed * line_gap:
            new_page()

    def draw_title(text: str):
        nonlocal y
        ensure_space(2)
        c.setFont(font_name, title_size)
        c.drawString(48, y, text)
        y -= 28

    def draw_heading(text: str):
        nonlocal y
        ensure_space(2)
        c.setFont(font_name, heading_size)
        c.drawString(48, y, text)
        y -= 18
        c.setFont(font_name, body_size)

    def draw_line(text: str):
        nonlocal y
        ensure_space(1)
        c.drawString(56, y, safe_text(text))
        y -= line_gap

    def draw_paragraph(text: str, width_chars: int = 78):
        nonlocal y
        parts = wrap_text(safe_text(text), width_chars)
        for part in parts:
            draw_line(part)

    draw_title("Catcher Teaching Feedback Report")

    draw_heading("1. Blackboard OCR Summary")
    if board_lines:
        for i, line in enumerate(board_lines, start=1):
            draw_paragraph(f"{i}. {line}")
    else:
        draw_line("No board text detected.")

    y -= 6

    draw_heading("2. Handwriting Clarity")
    clarity_label = str(clarity.get("label", "unknown"))
    clarity_score = clarity.get("score", "N/A")
    clarity_suggestion = str(clarity.get("suggestion", "No suggestion available."))
    laplacian = clarity.get("laplacian", "N/A")
    stroke_var = clarity.get("stroke_width_variance", "N/A")

    draw_line(f"Clarity label: {clarity_label}")
    draw_line(f"Clarity score: {clarity_score}")
    draw_line(f"Laplacian variance: {laplacian}")
    draw_line(f"Stroke-width variance: {stroke_var}")
    draw_paragraph(f"Suggestion: {clarity_suggestion}")

    y -= 6

    draw_heading("3. Board–Speech Alignment")
    verdict = str(alignment.get("verdict", "unknown"))
    semantic_similarity = alignment.get("semantic_similarity", "N/A")
    keyword_overlap = alignment.get("keyword_overlap_rate", "N/A")
    matched_topics = alignment.get("matched_topics", []) or []
    board_only_topics = alignment.get("board_only_topics", []) or []
    speech_only_topics = alignment.get("speech_only_topics", []) or []

    draw_line(f"Verdict: {verdict}")
    draw_line(f"Semantic similarity: {semantic_similarity}")
    draw_line(f"Keyword overlap rate: {keyword_overlap}")

    draw_paragraph(
        "Matched topics: " + (", ".join(map(str, matched_topics[:15])) if matched_topics else "None")
    )
    draw_paragraph(
        "Board-only topics: " + (", ".join(map(str, board_only_topics[:15])) if board_only_topics else "None")
    )
    draw_paragraph(
        "Speech-only topics: " + (", ".join(map(str, speech_only_topics[:15])) if speech_only_topics else "None")
    )

    y -= 6

    draw_heading("4. Speech Transcript")
    if speech_text and speech_text.strip():
        draw_paragraph(speech_text, width_chars=82)
    else:
        draw_line("No speech transcript available.")

    y -= 6

    draw_heading("5. System Notes")
    if module_errors:
        for module_name, err in module_errors.items():
            draw_paragraph(f"{module_name}: {err}")
    else:
        draw_line("No module errors reported.")

    c.save()
    return str(path)


def register_cjk_font() -> None:
    """
    Use a built-in CID font so mixed Chinese/English text can render safely.
    """
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def wrap_text(text: str, width_chars: int = 78) -> List[str]:
    """
    Simple character-based wrap that also works reasonably for mixed CJK/English.
    """
    text = safe_text(text)
    if not text:
        return [""]

    words = text.split(" ")
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= width_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            if len(word) <= width_chars:
                current = word
            else:
                start = 0
                while start < len(word):
                    chunk = word[start : start + width_chars]
                    if len(chunk) == width_chars:
                        lines.append(chunk)
                    else:
                        current = chunk
                    start += width_chars
                if start >= len(word) and len(word) % width_chars == 0:
                    current = ""

    if current:
        lines.append(current)

    return lines