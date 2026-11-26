"""Offline structural gate for the Sprint 4 bibliography and citation context."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "4_LaTeX_Source/vldb_template"
BIB = PAPER / "stratq_refs.bib"
AUDIT = ROOT / "BIBLIOGRAPHY_AUDIT.md"


def main():
    bib = BIB.read_text(encoding="utf-8")
    entries = {m.group(2): m.group(1) for m in re.finditer(r"@(\w+)\{([^,]+),", bib)}
    tex = "\n".join(p.read_text(encoding="utf-8") for p in sorted(PAPER.glob("stratq*.tex")))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    audit = AUDIT.read_text(encoding="utf-8")
    preprints = {
        "zhou2026celeus", "spertus2026sequential", "hait2026target", "wu2026faq",
        "shekhar2026rates", "xu2026fppi", "dou2026conformalcascade", "khosravi2026rankcs",
    }
    forbidden = [
        "excludes queries whose aggregate appears",
        "global baselines fail",
        "94.7\\%",
        "incremental query RSS",
        "eliminate weak-proxy degradation",
        r"$\gamma_{N_g} N_g^2$",
        "restarting to $D_g(t_g)$ only if empty",
        "whenever weak proxy predictions would inflate",
        "only \\textsc{StratQ} holds valid anytime",
    ]
    proof = (ROOT / "PROOF_PACKAGE.md").read_text(encoding="utf-8")
    checks = {
        "62_active_entries": len(entries) == 62,
        "62_unique_cited_keys": len(cited) == 62,
        "no_undefined_citation": not (cited - entries.keys()),
        "no_uncited_entry": not (entries.keys() - cited),
        "all_keys_in_online_audit": all(key in audit for key in entries),
        "tao_pages_11664_11673": bool(re.search(r"@inproceedings\{tao2019threshold,.*?pages=\{11664--11673\}", bib, re.S)),
        "jourdan_formal_jmlr": bool(re.search(r"@article\{jourdan2026apgai,.*?volume=\{27\}.*?number=\{19\}.*?pages=\{1--90\}", bib, re.S)),
        "sfyraki_formal_iclr": bool(re.search(r"@inproceedings\{sfyraki2026active,.*?Fourteenth International Conference on Learning Representations", bib, re.S)),
        "eight_preprints_remain_misc": all(entries.get(k) == "misc" for k in preprints),
        "spertus_not_marked_accepted": "submitted to Annals of Applied Statistics" in bib and "forthcoming" not in bib.lower(),
        "no_pending_review_markers": "[待复核]" not in bib and "[to verify]" not in bib,
        "proof_lower_endpoint_uses_log_upper_bound": r"B_t^--(\ell^++V_t^+)" in proof and r"B_t^--(\ell^-+V_t^+)" not in proof,
        "forbidden_context_phrases_absent": all(phrase not in tex for phrase in forbidden),
    }
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bib_sha256": hashlib.sha256(BIB.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "active_entries": len(entries),
        "cited_keys": len(cited),
        "preprint_keys": sorted(preprints),
        "checks": checks,
        "passed": all(checks.values()),
    }
    target = ROOT / "results/bibliography_static_sprint4.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(target)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
