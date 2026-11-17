
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import html

def _highlight_words(text: str, spans: List[Tuple[int,int]]) -> str:
    words = text.split()
    marks = [0] * len(words)
    for (l, r) in spans:
        l = max(0, l); r = min(len(words)-1, r)
        for i in range(l, r+1):
            marks[i] = 1
    out = []
    for w, m in zip(words, marks):
        w = html.escape(w)
        out.append(f"<mark>{w}</mark>" if m else w)
    return " ".join(out)

def render_cases_to_html(cases: List[Dict[str, Any]], title: str = "SCOUT case-based explanations") -> str:
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             "<style>body{font-family:system-ui,Segoe UI,Arial;line-height:1.4} "
             "details{margin:8px 0} .card{border:1px solid #ddd;padding:12px;border-radius:8px;margin:12px 0} "
             "code{background:#f6f8fa;padding:2px 4px;border-radius:4px}</style>"
             "</head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1>")

    for c in cases:
        text = c.get("text", "")
        spans = [(e["word_start"], e["word_end"]) for e in c.get("evidence_spans", [])]
        highlighted = _highlight_words(text, spans) if text else ""
        parts.append("<div class='card'>")
        parts.append(f"<div><b>idx</b>: {c.get('idx')} &nbsp; <b>gold</b>: {c.get('gold')} "
                     f"&nbsp; <b>pred</b>: {c.get('pred')} &nbsp; <b>explained</b>: {c.get('explained_class')}</div>")
        parts.append(f"<div style='margin-top:6px'>{highlighted}</div>")

        parts.append("<h3>Evidence spans</h3><ul>")
        for e in c.get("evidence_spans", []):
            parts.append(f"<li><code>mass={e.get('mass'):.4f}</code> "
                         f"[{e.get('word_start')},{e.get('word_end')}] — {html.escape(e.get('span_text',''))}</li>")
        parts.append("</ul>")

        parts.append("<h3>Top prototypes (with Readable Anchors)</h3>")
        for p in c.get("top_prototypes", []):
            parts.append("<details><summary>"
                         f"prototype <b>{p['prototype']}</b> "
                         f"(responsibility={p['responsibility']:.4f})</summary>")
            parts.append("<ul>")
            for a in p.get("anchors", []):
                parts.append(f"<li><code>support={a.get('support')}</code> "
                             f"[{a.get('word_start')},{a.get('word_end')}] "
                             f"— {html.escape(a.get('span_text',''))}</li>")
            parts.append("</ul></details>")

        parts.append("<h3>Span → Anchor links</h3>")
        for item in c.get("span_support_links", []):
            parts.append(f"<details><summary>evidence span id {item.get('evidence_span')}</summary><ul>")
            for lk in item.get("top", []):
                parts.append(f"<li>proto {lk.get('prototype')} support {lk.get('support')}: "
                             f"<code>mass={lk.get('mass'):.6f}</code></li>")
            parts.append("</ul></details>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)
