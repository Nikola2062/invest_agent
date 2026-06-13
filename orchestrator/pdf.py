"""Render markdown reports into web-grade PDFs via WeasyPrint (HTML + CSS).

Real CSS gives us a proper cover page, a table of contents with live page
numbers (target-counter), filled + zebra-striped tables, callout blockquotes,
running headers and "Page X / Y" footers. Traditional Chinese renders natively
through fontconfig (no manual font embedding).

Two entry points (signatures unchanged from the old fpdf2 renderer):
  * markdown_to_pdf(md, ..., title, subtitle)      — single document (on-demand stock)
  * render_report(sections, ..., title, subtitle)  — cover + TOC + one page per section
"""
from __future__ import annotations

import html as _html
from pathlib import Path

import markdown as mdlib
from weasyprint import HTML

_MD_EXT = ["tables", "fenced_code", "sane_lists"]

# Emoji that don't render cleanly in print → readable substitutes.
_EMOJI = {"⭐": "★", "⏳": "", "✅": "[ok]", "❌": "[x]", "⚠️": "[!]",
          "🔎": "", "📊": "", "🟢": "●", "🔴": "●", "🟡": "●", "🔄": ""}


def _sanitize(md: str) -> str:
    for k, v in _EMOJI.items():
        md = md.replace(k, v)
    return md


def _strip_first_heading(md: str) -> str:
    lines, out, removed = md.split("\n"), [], False
    for ln in lines:
        if not removed and ln.lstrip().startswith("#"):
            removed = True
            continue
        out.append(ln)
    return "\n".join(out).lstrip("\n")


def _md_html(md: str) -> str:
    return mdlib.markdown(_sanitize(md), extensions=_MD_EXT)


def _css(running_title: str) -> str:
    rt = running_title.replace('"', "'")
    return f"""
@page {{
  size: A4;
  margin: 22mm 16mm 16mm 16mm;
  @top-right {{ content: "{rt}"; font-family: 'Helvetica Neue', sans-serif;
               font-size: 8pt; color: #9aa1ab; }}
  @bottom-left {{ content: "{rt}"; font-family: 'Helvetica Neue', sans-serif;
                 font-size: 8pt; color: #9aa1ab; }}
  @bottom-right {{ content: "Page " counter(page) " / " counter(pages);
                  font-family: 'Helvetica Neue', sans-serif; font-size: 8pt; color: #9aa1ab; }}
}}
@page :first {{ margin: 0; @top-right {{ content: none; }}
                @bottom-left {{ content: none; }} @bottom-right {{ content: none; }} }}

body {{ font-family: 'Helvetica Neue', 'PingFang TC', 'Arial Unicode MS', system-ui, sans-serif;
        color: #222831; font-size: 10.5pt; line-height: 1.45; }}
h1 {{ color: #112d4e; font-size: 17pt; margin: 14px 0 6px; }}
h2 {{ color: #1a477a; font-size: 13.5pt; margin: 12px 0 5px;
      border-bottom: 1px solid #e0e4e9; padding-bottom: 3px; }}
h3 {{ color: #37598a; font-size: 11.5pt; margin: 10px 0 4px; }}
h4 {{ color: #5a6068; font-size: 10.5pt; margin: 8px 0 3px; }}
p, li {{ margin: 3px 0; }}
a {{ color: #2196f3; text-decoration: none; }}
strong, b {{ color: #1a2230; }}

/* tables — filled header, zebra rows */
table {{ width: 100%; border-collapse: collapse; font-size: 9pt; margin: 8px 0 12px;
         table-layout: auto; }}
thead th {{ background: #112d4e; color: #fff; text-align: left; font-weight: 600;
            padding: 6px 8px; }}
tbody td {{ padding: 5px 8px; border-bottom: 1px solid #e6e9ee; vertical-align: top; }}
tbody tr:nth-child(even) {{ background: #f5f7fa; }}

/* callouts + code */
blockquote {{ background: #edf4fb; border-left: 4px solid #2196f3; margin: 8px 0;
              padding: 7px 12px; color: #37485a; }}
code {{ font-family: 'SF Mono', Menlo, Consolas, monospace; background: #f1f4f8;
        padding: 1px 4px; border-radius: 3px; font-size: 8.5pt; }}
pre {{ background: #f4f6f9; border: 1px solid #e6e9ee; border-radius: 6px; padding: 9px 11px;
       white-space: pre-wrap; word-break: break-word; font-size: 8pt; line-height: 1.3; }}
pre code {{ background: none; padding: 0; }}

/* cover */
.cover {{ page-break-after: always; }}
.cover-band {{ background: #112d4e; color: #fff; border-bottom: 3px solid #2196f3;
               padding: 15px 16mm; }}
.cover-eyebrow {{ font-size: 10pt; font-weight: 600; letter-spacing: 3px; }}
.cover-body {{ padding: 78mm 16mm 0; }}
.cover-title {{ color: #112d4e; font-size: 32pt; font-weight: 700; margin: 0; line-height: 1.1; }}
.cover-rule {{ width: 64px; height: 3px; background: #2196f3; margin: 12px 0 16px; }}
.cover-sub {{ color: #7c848e; font-size: 13pt; }}

/* table of contents */
.toc {{ page-break-after: always; }}
.toc h1 {{ border-bottom: 2px solid #2196f3; padding-bottom: 5px; }}
.toc ol {{ list-style: none; padding: 0; margin: 10px 0 0; }}
.toc li {{ padding: 6px 0; border-bottom: 1px dotted #d7dce3; font-size: 11pt; }}
.toc a {{ color: #222831; }}
.toc a::after {{ content: target-counter(attr(href), page); float: right; color: #7c848e; }}

/* sections */
.section {{ page-break-before: always; }}
.section > .section-title {{ color: #112d4e; font-size: 20pt; font-weight: 700;
                             border-bottom: 2px solid #2196f3; padding-bottom: 6px; margin: 0 0 10px; }}
"""


def _cover_html(title: str, subtitle: str) -> str:
    sub = f'<div class="cover-sub">{_html.escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="cover">'
        '<div class="cover-band"><span class="cover-eyebrow">INVESTOR REPORT</span></div>'
        '<div class="cover-body">'
        f'<div class="cover-title">{_html.escape(title)}</div>'
        '<div class="cover-rule"></div>'
        f'{sub}</div></div>'
    )


def _write(html_body: str, running_title: str, out_path: Path) -> Path:
    doc = f"<html><head><meta charset='utf-8'><style>{_css(running_title)}</style></head><body>{html_body}</body></html>"
    HTML(string=doc).write_pdf(str(out_path))
    return out_path


def markdown_to_pdf(md_text: str, out_path, *, title: str, subtitle: str = "") -> Path:
    out_path = Path(out_path)
    body = _cover_html(title, subtitle) + f'<div class="content">{_md_html(_strip_first_heading(md_text))}</div>'
    return _write(body, title, out_path)


def render_report(sections: list[dict], out_path, *, title: str, subtitle: str = "",
                  toc_title: str = "Contents") -> Path:
    out_path = Path(out_path)
    parts = [_cover_html(title, subtitle)]

    toc_items = "".join(
        f'<li><a href="#sec-{i}">{_html.escape(sec["title"])}</a></li>'
        for i, sec in enumerate(sections)
    )
    parts.append(f'<div class="toc"><h1>{_html.escape(toc_title)}</h1><ol>{toc_items}</ol></div>')

    for i, sec in enumerate(sections):
        parts.append(
            f'<section class="section" id="sec-{i}">'
            f'<h1 class="section-title">{_html.escape(sec["title"])}</h1>'
            f'{_md_html(_strip_first_heading(sec["md"]))}</section>'
        )
    return _write("".join(parts), title, out_path)
