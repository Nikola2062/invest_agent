"""Markdown → HTML → PDF rendering for the daily report.

Engine: `fpdf2` (pure Python, zero system-library dependencies). The previous
implementation used `weasyprint` which needs `libcairo`/`libpango` installed
via the OS package manager — not feasible on managed cloud hosts. fpdf2 ships
as a single wheel and works on any platform where pip works.

Pipeline:
  1. `markdown` package converts the report's GFM-ish markdown to HTML
     (tables, fenced code, basic lists).
  2. ASCII-fold the HTML — fpdf2's default Helvetica font is Latin-1 only,
     so we replace emoji and special characters (—, ✓, ✗, ⚠, ⭐, etc.)
     with ASCII equivalents BEFORE passing to fpdf2. The MARKDOWN report on
     disk keeps its full Unicode formatting; only the PDF is ASCII-folded.
  3. fpdf2's `pdf.write_html()` renders the simplified HTML.

If you want full Unicode in the PDF later, add a TTF font via
`pdf.add_font("DejaVu", fname=".../DejaVuSans.ttf", uni=True)` and bypass
the ASCII fold. See fpdf2 docs.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import markdown
from fpdf import FPDF

log = logging.getLogger(__name__)


# Translation table for the special characters our report uses. The MD report
# keeps full Unicode; this map is applied only on the PDF rendering path.
ASCII_SUBST: dict[str, str] = {
    # dashes / hyphens
    "—": "-",        # em dash
    "–": "-",        # en dash
    "−": "-",        # minus sign
    "‐": "-",        # hyphen
    # quotes
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    # bullets / list markers
    "•": "*",
    "·": ".",
    # arrows
    "→": "->", "←": "<-", "↑": "^", "↓": "v",
    # math / formula symbols
    "×": "x", "÷": "/", "±": "+/-",
    "≥": ">=", "≤": "<=", "≈": "~", "≠": "!=",
    "√": "sqrt",
    # Greek letters used in the metrics docs
    "σ": "sigma", "Σ": "Sum", "μ": "mu", "Δ": "D", "δ": "d",
    "α": "a", "β": "b", "π": "pi",
    # superscripts (Δ²RS = D2RS in ASCII fallback)
    "²": "2", "³": "3", "¹": "1", "⁴": "4",
    # block-drawing chars used by the Phase A Flow Map bars
    "█": "#", "▓": "#", "▒": "=", "░": "-",
    # report-specific glyphs
    "✓": "[OK]",      # consistent direction
    "✗": "[X]",       # contradicting
    "⚠️": "[!]",       # warning (compound emoji)
    "⚠": "[!]",       # warning sign (BMP)
    "⏳": "[pending]",
    "⭐": "*",
    "📊": "",
    "🤖": "",
    "✅": "[PASS]",
    # whitespace
    " ": " ",     # nbsp
    " ": " ",     # thin space
    " ": " ",     # narrow nbsp
}


_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def _ascii_fold(text: str) -> str:
    """Apply ASCII_SUBST first, then drop any remaining non-ASCII char with '?'.
    Logs a sample of stripped characters once so the operator notices if the
    table needs expansion."""
    for src, dst in ASCII_SUBST.items():
        text = text.replace(src, dst)
    leftovers = _NON_ASCII_RE.findall(text)
    if leftovers:
        unique = sorted(set(leftovers))[:10]
        log.info("PDF render: stripping %d non-ASCII chars (sample: %r); "
                 "add these to ASCII_SUBST if you want them preserved", len(leftovers), unique)
        text = _NON_ASCII_RE.sub("?", text)
    return text


def markdown_to_pdf(md_text: str, output_path: Path) -> Path:
    """Render the report markdown to PDF at `output_path`. Returns the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    html_ascii = _ascii_fold(html_body)

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    pdf.write_html(html_ascii, table_line_separators=True)

    pdf.output(str(output_path))
    return output_path
