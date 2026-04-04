"""
Module E: PDF teaching feedback report (ReportLab + optional CJK font on Windows).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None  # type: ignore


def _register_cjk_font() -> str:
    """Return a font name that can render CJK if OCR/speech contains Chinese."""
    if canvas is None:
        return "Helvetica"
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for p in candidates:
        if not p.is_file():
            continue
        name = "CJKFont"
        try:
            pdfmetrics.registerFont(TTFont(name, str(p)))
            return name
        except Exception as e:
            logger.debug("Font register failed %s: %s", p, e)
    return "Helvetica"


def build_teaching_feedback_pdf(
    output_path: str,
    *,
    board_lines: List[str],
    clarity: Dict[str, Any],
    alignment: Optional[Dict[str, Any]],
    speech_text: str = "",
    module_errors: Optional[Dict[str, str]] = None,
) -> str:
    """Build PDF; returns absolute path written."""
    if canvas is None:
        raise RuntimeError("Install reportlab")

    font = _register_cjk_font()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 50
    line_h = 16

    def draw_line(text: str, indent: int = 0) -> None:
        nonlocal y
        if y < 60:
            c.showPage()
            y = h - 50
        c.setFont(font, 11)
        safe = text.encode("latin-1", "replace").decode("latin-1") if font == "Helvetica" else text
        try:
            c.drawString(50 + indent, y, safe[:120])
        except Exception:
            c.drawString(50 + indent, y, safe.encode("ascii", "replace").decode("ascii")[:120])
        y -= line_h

    draw_line("Classroom blackboard analytics — teaching feedback report")
    y -= 8
    draw_line(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    y -= 8

    draw_line("1. Board OCR")
    if board_lines:
        for t in board_lines:
            draw_line(f"- {t}", indent=10)
    else:
        draw_line("(no text recognized)", indent=10)
    y -= 8

    draw_line("2. Handwriting clarity")
    if clarity:
        draw_line(f"Level: {clarity.get('clarity', '')}  Score: {clarity.get('score', '')}", indent=10)
        draw_line(f"Suggestion: {clarity.get('suggestion', '')}", indent=10)
        draw_line(
            f"Laplacian variance: {clarity.get('laplacian_variance', '')}  "
            f"Stroke width variance (across components): {clarity.get('stroke_width_variance', '')}",
            indent=10,
        )
    y -= 8

    draw_line("3. Speech summary (Whisper)")
    st = speech_text or "(missing or transcription failed)"
    for chunk in range(0, len(st), 90):
        draw_line(st[chunk : chunk + 90], indent=10)
    y -= 8

    draw_line("4. Board vs speech alignment")
    if alignment:
        draw_line(f"Semantic similarity: {alignment.get('semantic_similarity')}", indent=10)
        draw_line(f"Keyword overlap (Jaccard): {alignment.get('keyword_overlap_rate')}", indent=10)
        draw_line(f"Verdict: {alignment.get('verdict')}", indent=10)
    else:
        draw_line("(skipped or unavailable)", indent=10)

    if module_errors:
        y -= 8
        draw_line("5. Module errors")
        for k, v in module_errors.items():
            if v:
                draw_line(f"{k}: {v}", indent=10)

    c.save()
    return str(path.resolve())


def run_module_e(
    output_path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"pdf_path": None, "error": None}
    try:
        out["pdf_path"] = build_teaching_feedback_pdf(
            output_path,
            board_lines=payload.get("board_lines") or [],
            clarity=payload.get("clarity") or {},
            alignment=payload.get("alignment"),
            speech_text=payload.get("speech_text") or "",
            module_errors=payload.get("module_errors"),
        )
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_e")
    return out
