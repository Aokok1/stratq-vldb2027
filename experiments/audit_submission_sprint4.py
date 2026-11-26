"""Check verbatim CMT title/abstract and manuscript-backed SAR excerpts."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "4_LaTeX_Source/vldb_template"


def plain(text: str, *, latex_comments: bool = True) -> str:
    if latex_comments:
        text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_").replace(r"\,", " ")
    text = text.replace("~", " ")

    def math_plain(match):
        value = match.group(1)
        value = value.replace(r"\times", "x").replace(r"\delta", "delta")
        value = value.replace(r"\epsilon", "epsilon").replace(r"\leq", "<=")
        value = value.replace(r"\geq", ">=").replace(r"\pm", "+/-")
        value = value.replace(r"\infty", "inf")
        value = re.sub(r"\\[A-Za-z]+", "", value)
        return value

    text = re.sub(r"\$(.*?)\$", math_plain, text, flags=re.S)
    for _ in range(4):
        text = re.sub(r"\\(?:texttt|textsc|emph|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:ref|eqref|label)\{[^}]*\}", "", text)
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def section(text: str, name: str, next_name: str) -> str:
    match = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## {re.escape(next_name)}\s*$)", text, re.M | re.S)
    if not match:
        raise ValueError(f"Missing CMT section: {name}")
    return match.group(1).strip()


def main():
    main_tex = (PAPER / "stratq.tex").read_text(encoding="utf-8")
    all_tex = "\n".join(p.read_text(encoding="utf-8") for p in sorted(PAPER.glob("stratq*.tex")))
    cmt_path = ROOT / "CMT_REGISTRATION_VLDB2027.md"
    cmt = cmt_path.read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]*)\}", main_tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S).group(1)
    cmt_title = section(cmt, "Title", "Abstract")
    cmt_abstract = section(cmt, "Abstract", "Self-Assessment of Relevance (SAR)")
    sar = section(cmt, "Self-Assessment of Relevance (SAR)", "Suggested topics")
    checks = {
        "title_verbatim": plain(title) == plain(cmt_title, latex_comments=False),
        "abstract_verbatim": plain(abstract) == plain(cmt_abstract, latex_comments=False),
        "author_placeholder_preserved": "\\author{Author information pending}" in main_tex,
        "availability_url_blank_preserved": "\\renewcommand\\vldbavailabilityurl{}" in main_tex,
    }
    sar_paragraphs = [plain(p, latex_comments=False) for p in re.split(r"\n\s*\n", sar) if p.strip()]
    all_plain = plain(all_tex)
    checks["sar_paragraph_count"] = len(sar_paragraphs) == 3
    checks["sar_paragraphs_verbatim_from_manuscript"] = all(p in all_plain for p in sar_paragraphs)
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(v is True for v in checks.values()),
        "manuscript_sha256": hashlib.sha256(main_tex.encode()).hexdigest(),
        "cmt_sha256": hashlib.sha256(cmt.encode()).hexdigest(),
        "normalized_title": plain(title),
        "normalized_abstract": plain(abstract),
        "sar_paragraphs": sar_paragraphs,
    }
    target = ROOT / "results/submission_consistency_sprint4.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not out["passed"]:
        if not checks["abstract_verbatim"]:
            print("MANUSCRIPT ABSTRACT:", out["normalized_abstract"])
            print("CMT ABSTRACT:", plain(cmt_abstract, latex_comments=False))
        for i, paragraph in enumerate(sar_paragraphs, 1):
            if paragraph not in all_plain:
                print("SAR paragraph not found verbatim:", i, paragraph)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
