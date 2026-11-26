"""Sprint 3/4 end-to-end Claim-to-JSON audit for the 12-page StratQ manuscript.

Verifies load-bearing numerical claims against saved raw/summary JSON, then
checks Sprint 4 statistics, proof/code consistency, bibliography, CMT, PDF,
and LaTeX hard gates.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import fitz

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "4_LaTeX_Source" / "vldb_template"
RESULTS = ROOT / "results"


def main() -> int:
    inputs: dict[str, str] = {}
    checks: list[dict] = []

    def read_text(path: Path) -> str:
        raw = path.read_bytes()
        rel = str(path.relative_to(ROOT))
        inputs[rel] = "sha256:" + hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8")

    def load_json(name: str):
        return json.loads(read_text(RESULTS / f"{name}.json"))

    def record(name: str, ok: bool, detail: str = ""):
        checks.append({"claim": name, "status": "match" if ok else "mismatch", "detail": detail})

    tex = {p.name: read_text(p) for p in sorted(PAPER.glob("stratq*.tex"))}
    all_tex = "\n".join(tex.values())

    def expect_in(fname: str, frag: str, label: str):
        record(f"{label} ({fname})", frag in tex[fname], f"expected fragment: {frag!r}")

    # 1. Load raw verified JSONs
    s1 = load_json("sprint1_safe_gating_ablation")
    gpu = load_json("sprint2_gpu_workloads")
    sys_d = load_json("sprint2_systems")
    mem_d = load_json("sprint2_memory_10m")
    fb = load_json("adult_fixed_budget")
    bl = load_json("adult_blazeit")
    sn = load_json("adult_senate")
    oa = load_json("adult_online_aggregation")
    lw = load_json("adult_lws")
    vd = load_json("adult_verdict")
    hy = load_json("adult_hypergeom")
    db = load_json("decision_boundaries")
    rc = load_json("adult_routing_capped")
    stats = load_json("sprint4_summary_statistics")
    gate = load_json("sprint4_yelp_gate_diagnostic")
    code_validation = load_json("sprint4_code_change_validation")
    bib_static = load_json("bibliography_static_sprint4")
    submission = load_json("submission_consistency_sprint4")
    proof = read_text(ROOT / "PROOF_PACKAGE.md")

    # 2. Sprint 1 Safe-LOO ablation claims
    f_s1 = s1["fashion_mnist"]["summary"]
    f_pct = 100.0 * (f_s1["safe_loo"]["mean_calls"] / f_s1["no_proxy"]["mean_calls"] - 1.0)
    record("Fashion safe_loo mean calls = 1360.0", abs(f_s1["safe_loo"]["mean_calls"] - 1360.0) < 1e-6)
    record("Fashion safe_loo delta = -61.26%", abs(f_pct - (-61.26)) < 0.01, f"{f_pct:.4f}%")
    expect_in("stratq_evaluation.tex", "1,360.0", "Fashion safe_loo calls in prose/table")
    expect_in("stratq_evaluation.tex", "61.26\\%", "Fashion -61.26% in prose/table")

    sms_c = s1["sms"]["workloads"]["COUNT"]["summary"]
    sms_h = s1["sms"]["workloads"]["HAVING"]["summary"]
    sms_t = s1["sms"]["workloads"]["TOP-2"]["summary"]
    record("SMS COUNT safe_loo = 702.4 (-14.01%)",
           abs(sms_c["safe_loo"]["mean_calls"] - 702.4) < 1e-6
           and abs(100 * (sms_c["safe_loo"]["mean_calls"] / sms_c["no_proxy"]["mean_calls"] - 1) + 14.01) < 0.01)
    record("SMS HAVING safe_loo = 848.7 (-1.67%)",
           abs(sms_h["safe_loo"]["mean_calls"] - 848.7) < 1e-6
           and abs(100 * (sms_h["safe_loo"]["mean_calls"] / sms_h["no_proxy"]["mean_calls"] - 1) + 1.67) < 0.01)
    record("SMS TOP-2 safe_loo = 858.1 (-0.90%)",
           abs(sms_t["safe_loo"]["mean_calls"] - 858.1) < 1e-6
           and abs(100 * (sms_t["safe_loo"]["mean_calls"] / sms_t["no_proxy"]["mean_calls"] - 1) + 0.90) < 0.01)
    expect_in("stratq_evaluation.tex", "702.4", "SMS COUNT 702.4 in tex")
    expect_in("stratq_evaluation.tex", "14.01\\%", "SMS COUNT -14.01% in tex")

    ad_s1 = s1["adult"]["summary"]
    record("Adult safe_loo = 4378.0 (+0.93%)",
           abs(ad_s1["safe_loo"]["mean_calls"] - 4378.0) < 1e-6
           and abs(100 * (ad_s1["safe_loo"]["mean_calls"] / ad_s1["no_proxy"]["mean_calls"] - 1) - 0.93) < 0.01)
    expect_in("stratq_evaluation.tex", "4,378.0", "Adult safe_loo calls in tex")
    expect_in("stratq_evaluation.tex", "+0.93\\%", "Adult +0.93% honest disclosure in tex")

    # 3. Sprint 2 GPU workloads (Yelp + CIFAR-10)
    wmap = {w["workload"]: w for w in gpu["workloads"]}
    yelp = wmap["Yelp-DistilBERT-SST2"]
    cifar = wmap["CIFAR10-ResNet34-GPU"]

    y_sp_f = yelp["latency"]["median_exact_ms"] / yelp["latency"]["median_fresh_ms"]
    y_sp_r = yelp["latency"]["median_exact_ms"] / yelp["latency"]["median_reused_query_ms"]
    c_sp_f = cifar["latency"]["median_exact_ms"] / cifar["latency"]["median_fresh_ms"]
    c_sp_r = cifar["latency"]["median_exact_ms"] / cifar["latency"]["median_reused_query_ms"]
    record("Yelp fresh/reused speedups = 3.61x / 3.67x",
           abs(y_sp_f - 3.61) < 0.01 and abs(y_sp_r - 3.67) < 0.01, f"{y_sp_f:.3f}x / {y_sp_r:.3f}x")
    record("CIFAR-10 fresh/reused speedups = 4.58x / 4.78x",
           abs(c_sp_f - 4.58) < 0.01 and abs(c_sp_r - 4.78) < 0.01, f"{c_sp_f:.3f}x / {c_sp_r:.3f}x")
    expect_in("stratq.tex", "3.61\\times$--$4.78\\times", "GPU speedup range in abstract")
    expect_in("stratq_evaluation.tex", "77.99", "Yelp exact 77.99s in tex")
    expect_in("stratq_evaluation.tex", "18.22", "CIFAR exact 18.22s in tex")

    y_top2 = mean(r["calls"] for r in yelp["decision_runs"] if r["query"] == "TOP-2" and r["method"] == "StratQ-safe_loo")
    c_top2 = mean(r["calls"] for r in cifar["decision_runs"] if r["query"] == "TOP-2" and r["method"] == "StratQ-safe_loo")
    record("Yelp StratQ TOP-2 mean calls = 594.7 (3.0%)",
           abs(y_top2 - 594.6667) < 0.1 and abs(100 * y_top2 / 20000 - 3.0) < 0.05)
    record("CIFAR StratQ COUNT = 1936.0 and TOP-2 = 4186.7 (3/3 cert)",
           abs(gpu["summary"]["CIFAR10-ResNet34-GPU"]["StratQ-safe_loo"]["mean_calls"] - 1936.0) < 1e-6
           and abs(c_top2 - 4186.6667) < 0.1)
    expect_in("stratq_evaluation.tex", "594.7", "Yelp TOP-2 calls in tex")
    expect_in("stratq_evaluation.tex", "1,936.0", "CIFAR COUNT calls in tex")
    expect_in("stratq_evaluation.tex", "4,186.7", "CIFAR TOP-2 calls in tex")

    # 4. Sprint 2 Systems: Proxy Sweep, Zipf Stress, Amortization, 10M Memory
    ps = sys_d["proxy_quality_sweep"]["summary"]
    record("Proxy sweep AUC 0.95 safe_loo=4064 vs no_proxy=5152",
           abs(ps["0.95|safe_loo"]["mean_calls"] - 4064) < 1e-6
           and abs(ps["0.95|no_proxy"]["mean_calls"] - 5152) < 1e-6)
    record("Proxy sweep AUC 0.50 safe_loo=5163 (+0.21%) vs CELEUS=9579",
           abs(ps["0.50|safe_loo"]["mean_calls"] - 5163) < 0.5
           and abs(ps["0.50|CELEUS-FixedProxy"]["mean_calls"] - 9579) < 0.5)
    expect_in("stratq_evaluation.tex", "4,064", "Proxy sweep 4,064 in tex")
    expect_in("stratq_evaluation.tex", "5,163", "Proxy sweep 5,163 in tex")
    expect_in("stratq_evaluation.tex", "9,579", "CELEUS 9,579 in tex")

    zs = sys_d["zipf_stress"]["runs"]
    for m, expected_cov in [("StratQ-safe_loo", 30), ("BARGAIN-Global", 0), ("PilotDB-Asymp", 16)]:
        sub = [r for r in zs if r["method"] == m]
        cov = sum(r["all_covered"] for r in sub)
        record(f"Zipf 30-run coverage {m} = {expected_cov}/30", cov == expected_cov, f"actual={cov}/{len(sub)}")
    p100_cov = sum(r["all_covered"] for r in zs if r["method"] == "PilotDB-Asymp" and r["G"] == 100)
    record("PilotDB-Asymp G=100 coverage = 0/6", p100_cov == 0)

    rec10 = [r for r in sys_d["profile_amortization"]["records"] if r["rows"] == 10000000][0]
    record("10M profile=8.47s, warm=2.83s, Q*=2",
           abs(rec10["profile_once_ms"] / 1000 - 8.47) < 0.01
           and abs(rec10["median_reused_ms"] / 1000 - 2.83) < 0.01
           and rec10["breakeven_vs_fresh"] == 2)
    qstage = [s for s in mem_d["stages"] if s["stage"] == "query"][0]
    record("10M online query delta RSS = +21.69 MB",
           abs(qstage["rss_query_delta_bytes"] / 1e6 - 21.69) < 0.02,
           f"{qstage['rss_query_delta_bytes']/1e6:.2f} MB")
    expect_in("stratq_evaluation.tex", "21.69", "10M +21.69 MB in tex")

    # 5. Classical Adult baselines & decision boundaries & capped routing
    tr = fb["truth"]
    def mhw(runs, method=None):
        sub = [r for r in runs if r["budget"] == 4096 and (method is None or r.get("method") == method)]
        return mean(100 * max((v["upper"] - v["lower"]) / (2 * tr[g]["size"]) for g, v in r["groups"].items()) for r in sub)

    for label, val, exp in [
        ("Adult StratQ 4096 = 5.29pp", mhw(fb["runs"], "stratq"), 5.29),
        ("Adult BlazeIt 4096 = 11.92pp", mhw(bl["runs"]), 11.92),
        ("Adult ABae-insp 4096 = 12.62pp", mhw(fb["runs"], "proxy_stratified_fixed"), 12.62),
        ("Adult Senate 4096 = 4.19pp", mhw(sn["runs"]), 4.19),
        ("Adult OLA 4096 = 44.52pp", mhw(oa["runs"]), 44.52),
        ("Adult LWS 4096 = 4.45pp", mhw(lw["runs"]), 4.45),
        ("Adult VerdictDB 4096 = 3.88pp", mhw(vd["runs"]), 3.88),
    ]:
        record(label, abs(val - exp) < 0.01, f"actual={val:.2f}")
    record("Decision boundaries total runs = 960", len(db["runs"]) == 960)
    record("Adult capped routing runs = 360 (180 pairs)", len(rc["runs"]) == 360)

    # 6. Sprint 4 raw-JSON statistics, semantics, proof, and submission gates
    yelp_stats = stats["gpu_count"]["Yelp-DistilBERT-SST2"]
    record("Yelp COUNT mean/sample SD matches raw runs",
           abs(yelp_stats["StratQ-safe_loo"]["calls"]["std_sample"] - 73.9008344562721) < 1e-6
           and abs(yelp_stats["Spertus-WoR"]["calls"]["std_sample"] - 97.76161482572459) < 1e-6)
    expect_in("stratq_evaluation.tex", r"4{,}349\pm74", "Yelp COUNT sample SD in tex")
    record("Yelp gate trace reproduces frozen call means",
           abs(gate["summary"]["stratq_mean_calls"] - 4349.333333333333) < 1e-6
           and abs(gate["summary"]["spertus_mean_calls"] - 3858.6666666666665) < 1e-6)
    record("Yelp range-penalty explanation rejected by trace",
           gate["summary"]["selected_proxy_calls"] == 112
           and gate["summary"]["selected_with_range_factor_above_1_01"] == 0
           and gate["summary"]["range_penalty_explanation_supported"] is False)
    expect_in("stratq_evaluation.tex", "range-penalty explanation is therefore unsupported", "Yelp mechanism correction in tex")

    ylat = stats["gpu_latency"]["Yelp-DistilBERT-SST2"]
    record("Yelp proxy-scan and first-query sensitivity match JSON",
           abs(ylat["speedup_fresh_including_proxy_scan"] - 3.047477441144744) < 1e-9
           and abs(ylat["speedup_first_query_including_proxy_train_and_scan"] - 2.4112035204240447) < 1e-9)
    expect_in("stratq_evaluation.tex", "3.05$\\times$/3.09$\\times$", "Yelp proxy-scan sensitivity in tex")
    expect_in("stratq_evaluation.tex", "2.41$\\times$", "Yelp first-query setup sensitivity in tex")
    cpu = stats["cpu_latency_aligned"]
    record("Aligned CPU timing exact/fresh/reused values match JSON",
           abs(cpu["fashion"]["exact_best_median_wall_ms"] - 1430.47) < 0.02
           and abs(cpu["fashion"]["fresh_proxy_median_wall_ms"] - 577.32) < 0.02
           and abs(cpu["sms"]["exact_best_median_wall_ms"] - 519.24) < 0.02
           and abs(cpu["sms"]["reused_proxy_median_wall_ms"] - 834.74475) < 0.02)
    expect_in("stratq_evaluation.tex", "1,430.5", "Aligned Fashion exact timing in tex")
    expect_in("stratq_evaluation.tex", "519.2", "Aligned SMS exact timing in tex")

    record("Zipf design is 15 configurations x 2 seeds = 30 runs",
           stats["zipf"]["configurations"] == 15 and stats["zipf"]["runs_per_method"] == 30)
    record("PilotDB optional-stop tail coverage = 215/222",
           stats["zipf"]["methods"]["PilotDB-Asymp"]["tail_covered_instances"] == 215
           and stats["zipf"]["methods"]["PilotDB-Asymp"]["tail_total_instances"] == 222)
    expect_in("stratq_evaluation.tex", "215/222=96.85", "Correct PilotDB tail coverage in tex")
    record("BARGAIN global guarantee is not evaluated by per-group JSON",
           stats["zipf"]["methods"]["BARGAIN-Global"]["all_group_covered_runs"] == 0
           and "global interval" in tex["stratq_evaluation.tex"])

    record("Decision-boundary audit finds zero wrong certificates",
           stats["decision_boundaries"]["wrong_certificates"] == 0
           and stats["decision_boundaries"]["interval_coverage_failures"] == 0
           and stats["decision_boundaries"]["topk_certified_correct_runs"] == 288)
    record("Decision-boundary positive-gap and tie denominators match JSON",
           stats["decision_boundaries"]["positive_gap_topk_budget768"] == {"certified": 120, "total": 120}
           and stats["decision_boundaries"]["tied_topk_census"] == {"certified": 40, "total": 40})
    expect_in("stratq_evaluation.tex", "40/40", "Tie census denominator in tex")
    record("Capped routing improves 180/180 matched pairs",
           stats["capped_routing"]["matched_pairs"] == 180
           and stats["capped_routing"]["strict_width_improvements"] == 180)

    mem = stats["memory_10m"]
    record("10M query stage wall and query-only wall are distinct",
           abs(mem["query_subprocess_total_wall_ms"] - 187849.9344) < 0.01
           and abs(mem["query_only_wall_ms"] - 37857.2799) < 0.01)
    record("10M immediate and final RSS deltas have opposite signs",
           mem["rss_query_delta_bytes"] == 21688320
           and mem["rss_final_minus_after_profile_bytes"] == -679936
           and mem["query_tracemalloc_peak_bytes"] == 85583716)
    expect_in("stratq_evaluation.tex", "187.85", "Query subprocess wall in tex")
    expect_in("stratq_evaluation.tex", "$-0.68$", "Final RSS delta in tex")
    expect_in("stratq_evaluation.tex", "85.6", "Query traced-allocation peak in tex")

    sms_count = stats["sprint1_ablation"]["SMS COUNT"]
    record("SMS legacy arm is reported and does not beat safe_loo claim",
           abs(sms_count["legacy"]["calls"]["mean"] - 689.6) < 1e-6
           and sms_count["legacy"]["calls"]["mean"] < sms_count["safe_loo"]["calls"]["mean"])
    expect_in("stratq_evaluation.tex", "689.6", "SMS legacy COUNT calls in tex")
    expect_in("stratq_evaluation.tex", "beat the legacy selector", "SMS legacy caveat in tex")
    record("Independent hypergeometric CS statistics match raw runs",
           abs(stats["adult_hypergeometric"]["calls"]["std_sample"] - 178.76) < 0.01
           and stats["adult_hypergeometric"]["covered_runs"] == 30)
    expect_in("stratq_evaluation.tex", r"4{,}581.7\pm178.8", "Hypergeometric SD in tex")

    record("Comparator inventory states 11 adaptations plus independent CS",
           "eleven database/modern operator adaptations plus an independent grouped hypergeometric CS" in tex["stratq_evaluation.tex"])
    record("Proof lower endpoint subtracts log upper bound",
           r"B_t^--(\ell^++V_t^+)" in proof and r"B_t^--(\ell^-+V_t^+)" not in proof)
    record("Higham feature-total bound has correct N scaling",
           r"\gamma_{N_g}N_g" in tex["stratq_method.tex"] and r"\gamma_{N_g} N_g^2" not in tex["stratq_method.tex"])
    record("Safe restart rule matches Theorem 2 and code",
           "only when the conservative previous and deterministic intervals are disjoint" in tex["stratq_method.tex"])
    record("PilotDB GROUP BY limitation is accurately scoped",
           "aggregates used as \\texttt{GROUP BY} keys" in tex["stratq_problem.tex"])
    record("BlazeIt local arm is labeled inspired fixed-look",
           "BlazeIt-inspired" in tex["stratq_evaluation.tex"])
    record("Safe-LOO no longer claims worst-case elimination",
           "eliminate weak-proxy degradation" not in all_tex and "worst-case call guarantee" in all_tex)
    record("Sprint 4 code replay preserves calls/decisions and widens outward",
           code_validation["calls_equal"] and code_validation["decisions_equal"]
           and code_validation["new_intervals_are_outward_of_old"])
    record("Bibliography static gate passes", bib_static["passed"])
    record("CMT title/abstract/SAR and placeholders pass", submission["passed"])

    # 7. PDF & BibTeX structural compliance
    doc = fitz.open(PAPER / "stratq.pdf")
    ref_page = next((i for i, p in enumerate(doc, 1) if "REFERENCES" in p.get_text()), None)
    if ref_page is None:
        body_pages = None
    else:
        before_refs = doc[ref_page - 1].get_text().split("REFERENCES", 1)[0].strip()
        body_pages = ref_page - 1 if not before_refs else ref_page
    record("PVLDB body <= 12 pages (references excluded)", body_pages is not None and body_pages <= 12,
           f"body_pages={body_pages}, total={len(doc)}, refs_page={ref_page}")
    bbl = read_text(PAPER / "stratq.bbl")
    n_bib = len(re.findall(r"\\bibitem", bbl))
    record("Bibliography cites all 62 verified references", n_bib == 62, f"cited={n_bib}")
    record("Zero markdown ** residue in stratq*.tex", "**" not in all_tex)
    log_txt = (PAPER / "stratq.log").read_text(encoding="utf-8", errors="ignore")
    over5 = []
    for line in log_txt.splitlines():
        if "Overfull \\hbox" in line:
            match = re.search(r"\(([0-9.]+)pt", line)
            if match and float(match.group(1)) >= 5.0:
                over5.append(line)
    record("Zero Overfull hbox >= 5pt", len(over5) == 0, f"count={len(over5)}")
    latex_errors = [line for line in log_txt.splitlines() if line.startswith("!")]
    record("Zero LaTeX errors", len(latex_errors) == 0, f"count={len(latex_errors)}")
    undefined_refs = [line for line in log_txt.splitlines()
                      if re.search(r"(Citation|Reference).*undefined|There were undefined references", line, re.I)]
    record("Zero undefined citations/references", len(undefined_refs) == 0, f"count={len(undefined_refs)}")

    passed = sum(c["status"] == "match" for c in checks)
    failed = len(checks) - passed
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "inputs": inputs,
        "checks": checks,
    }
    out_path = PAPER / "PAPER_CLAIM_AUDIT_SPRINT3.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Sprint 3/4 Claim Audit: {passed}/{len(checks)} passed, {failed} failed -> {out_path}")
    for c in checks:
        if c["status"] != "match":
            print("  MISMATCH:", c)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
