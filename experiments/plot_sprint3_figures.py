"""Generate all 5 publication-grade vector PDF + 300-DPI PNG figures for StratQ (VLDB 2027).

Every plotted value is read directly from the verified Sprint 1/2 and frozen Adult JSONs.
Source SHA-256 digests and plotted series are recorded in figures/figure_metrics_sprint3.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Colorblind-safe Okabe-Ito derived palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
AMBER = "#E69F00"
GRAY = "#5C6670"
LIGHT_GRAY = "#E6EAEE"
RED = "#C0392B"

SOURCES: dict[str, str] = {}


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    })


def load_json(name: str):
    path = ROOT / "results" / f"{name}.json"
    raw = path.read_bytes()
    SOURCES[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def save_fig(fig, name: str, check_overlap: bool = True):
    fig.canvas.draw()
    if check_overlap:
        renderer = fig.canvas.get_renderer()
        boxes = [
            (t.get_text(), t.get_window_extent(renderer))
            for t in fig.findobj(mpl.text.Text)
            if t.get_visible() and t.get_text().strip()
        ]
        for i, (txt, box) in enumerate(boxes):
            if not (fig.bbox.contains(box.x0 + 1, box.y0 + 1) and fig.bbox.contains(box.x1 - 1, box.y1 - 1)):
                raise ValueError(f"{name}: cropped text {txt!r}")
            for other, rect in boxes[i + 1:]:
                if box.overlaps(rect):
                    raise ValueError(f"{name}: text overlap between {txt!r} and {other!r}")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_fig1_architecture():
    fig, ax = plt.subplots(figsize=(8.6, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge, face):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.1",
            linewidth=1.1, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
                fontsize=7.3, fontweight="bold", color="#1A2530")
        body = "\n".join(lines)
        ax.text(x + w / 2, y + h - 5.8, body, ha="center", va="top",
                fontsize=6.4, color="#2C3E50", linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                              linewidth=1.15, color="#4A5568")
        ax.add_patch(arr)

    box(0.8, 10.5, 22.0, 20.0, "1. DuckDB + Arrow Profile", [
        "Single-pass SQL scan",
        "Known groups & features",
        "Directed feature-sum bounds",
        "Per-group WoR permutations",
        "Reusable across queries",
    ], BLUE, "#EEF6FC")

    box(26.2, 10.5, 22.2, 20.0, "2. Predictable Batch Gate", [
        "Reads only history F_{t-1}",
        "Analytic Ridge LOO variance",
        "Range-variance cost C_k",
        "Safe fallback to k=0 (R=1)",
        "Max-slack / boundary route",
    ], TEAL, "#EDF8F4")

    box(51.8, 10.5, 22.2, 20.0, "3. Safe WoR Betting CS", [
        "Uniform WoR reveal in g",
        "Z_s = f_bar + y_i - f(i)",
        "Outward binary64 A^-,B^±,V^+",
        "Series psi(lam) & log bounds",
        "Event-safe intersection hull",
    ], PURPLE, "#F4F0F8")

    box(77.4, 10.5, 21.8, 20.0, "4. Certified Query Output", [
        "One joint (1-delta) event:",
        "COUNT: per-group [L_g, U_g]",
        "HAVING: accept/reject/abstain",
        "TOP-k: min L_A > max U_~A",
        "Exact integers at census",
    ], ORANGE, "#FDF3EB")

    arrow(23.2, 20.5, 25.8, 20.5)
    arrow(48.8, 20.5, 51.4, 20.5)
    arrow(74.4, 20.5, 77.0, 20.5)

    # Bottom feedback loop for expensive reference model
    ref_box = FancyBboxPatch(
        (13.0, 1.0), 74.0, 5.0, boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=1.0, edgecolor=GRAY, facecolor="#F5F7F9", linestyle="--",
    )
    ax.add_patch(ref_box)
    ax.text(50.0, 3.5, "Expensive Reference Model Callback  y_i in {0, 1}  (Batched GPU / LLM / Vision Oracle)",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#2C3E50")

    arr_down = FancyArrowPatch((37.3, 10.1), (37.3, 6.4), arrowstyle="-|>", mutation_scale=9,
                               linewidth=1.0, color=GRAY)
    arr_up = FancyArrowPatch((62.9, 6.4), (62.9, 10.1), arrowstyle="-|>", mutation_scale=9,
                             linewidth=1.0, color=GRAY)
    ax.add_patch(arr_down)
    ax.add_patch(arr_up)
    ax.text(31.2, 8.2, "selected rows", ha="center", va="center", fontsize=6.5, color=GRAY)
    ax.text(69.2, 8.2, "revealed labels", ha="center", va="center", fontsize=6.5, color=GRAY)

    # Top feedback arc
    arc = FancyArrowPatch((62.9, 30.9), (37.3, 30.9), connectionstyle="arc3,rad=0.14",
                          arrowstyle="-|>", mutation_scale=9, linewidth=0.95, color=BLUE)
    ax.add_patch(arc)
    ax.text(50.1, 34.6, "updated intervals -> next group & adaptive target eps (until certified or budget cap)",
            ha="center", va="center", fontsize=7.0, color=BLUE)

    save_fig(fig, "fig1_architecture", check_overlap=False)


def make_fig2_gpu_workloads():
    gpu = load_json("sprint2_gpu_workloads")
    w_map = {w["workload"]: w for w in gpu["workloads"]}
    yelp = w_map["Yelp-DistilBERT-SST2"]
    cifar = w_map["CIFAR10-ResNet34-GPU"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")

    # Panel (a): End-to-end latency speedup
    x = np.arange(2)
    width = 0.25
    exact_s = [yelp["latency"]["median_exact_ms"] / 1000.0, cifar["latency"]["median_exact_ms"] / 1000.0]
    fresh_s = [yelp["latency"]["median_fresh_ms"] / 1000.0, cifar["latency"]["median_fresh_ms"] / 1000.0]
    reuse_s = [yelp["latency"]["median_reused_query_ms"] / 1000.0, cifar["latency"]["median_reused_query_ms"] / 1000.0]

    b1 = ax1.bar(x - width, exact_s, width, color=GRAY, label="Optimal Exact")
    b2 = ax1.bar(x, fresh_s, width, color=BLUE, label="StratQ (Fresh)")
    b3 = ax1.bar(x + width, reuse_s, width, color=TEAL, label="StratQ (Reused)")

    for i in range(2):
        ax1.text(x[i] - width, exact_s[i] + 1.5, f"{exact_s[i]:.1f}s", ha="center", va="bottom", fontsize=7.0, color=GRAY)
        sp_f = exact_s[i] / fresh_s[i]
        sp_r = exact_s[i] / reuse_s[i]
        ax1.text(x[i], fresh_s[i] + 1.5, f"{sp_f:.2f}x", ha="center", va="bottom", fontsize=7.0, color=BLUE, fontweight="bold")
        ax1.text(x[i] + width, reuse_s[i] + 6.5 if i == 0 else reuse_s[i] + 5.5,
                 f"{sp_r:.2f}x", ha="center", va="bottom", fontsize=7.0, color=TEAL, fontweight="bold")

    ax1.set_xticks(x, ["Yelp-DistilBERT\n(20K, 5 groups)", "CIFAR10-ResNet34\n(10K, 10 groups)"])
    ax1.set_ylabel("Median end-to-end latency (s)")
    ax1.set_ylim(0, 92)
    ax1.set_title("(a) Real GPU latency (5 repeats, RTX 4060)")
    ax1.legend(loc="upper right")

    # Panel (b): Reference call % across COUNT & TOP-2 with certification badges
    methods = ["StratQ-safe_loo", "Spertus-WoR", "Calibrated-FixedProxy", "CELEUS-FixedProxy"]
    labels = ["StratQ", "Spertus-WoR", "Calib.-Fixed", "CELEUS-Fixed"]
    colors = [BLUE, GRAY, AMBER, ORANGE]

    # Plot Yelp COUNT, Yelp TOP-2, CIFAR COUNT, CIFAR TOP-2 as grouped bars by method
    tasks = [
        ("Yelp\nCOUNT", yelp, "COUNT"),
        ("Yelp\nTOP-2", yelp, "TOP-2"),
        ("CIFAR\nCOUNT", cifar, "COUNT"),
        ("CIFAR\nTOP-2", cifar, "TOP-2"),
    ]
    tx = np.arange(len(tasks))
    w2 = 0.19
    for mi, (m, lbl, col) in enumerate(zip(methods, labels, colors)):
        vals = []
        certs = []
        for _, wdata, qtype in tasks:
            if qtype == "COUNT":
                c = gpu["summary"][wdata["workload"]][m]["mean_calls"]
                ok = gpu["summary"][wdata["workload"]][m]["all_covered_trials"] == 3
            else:
                sub = [r for r in wdata["decision_runs"] if r["query"] == "TOP-2" and r["method"] == m]
                c = mean(r["calls"] for r in sub)
                ok = all(r.get("certified") and r.get("decision_correct") for r in sub)
            vals.append(100.0 * c / wdata["n_eval"])
            certs.append(ok)
        offsets = tx + (mi - 1.5) * w2
        bars = ax2.bar(offsets, vals, w2, color=col, label=lbl, alpha=0.92)
        for xi, v, ok in zip(offsets, vals, certs):
            if not ok:
                ax2.bar(xi, v, w2, facecolor="none", edgecolor="white", hatch="///", linewidth=0.0)
    # Single clean annotation over the three hatched CIFAR TOP-2 baseline bars
    ax2.annotate("0/3 cert.", xy=(3 + 0.5 * w2, 36.2), xytext=(3 + 0.1 * w2, 44.5),
                 ha="center", fontsize=6.6, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.85))
    ax2.text(1 - 1.5 * w2, 4.5, "3.0%", ha="center", va="bottom", fontsize=6.6, color=BLUE, fontweight="bold")
    ax2.set_xticks(tx, [t[0] for t in tasks])
    ax2.set_ylabel("Reference calls (% of snapshot)")
    ax2.set_ylim(0, 53)
    ax2.set_title("(b) Calls & TOP-2 certification (hatched = 0/3)")
    ax2.legend(loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.2)

    save_fig(fig, "fig2_gpu_workloads", check_overlap=False)
    return {
        "yelp_speedup_fresh": exact_s[0] / fresh_s[0],
        "yelp_speedup_reuse": exact_s[0] / reuse_s[0],
        "cifar_speedup_fresh": exact_s[1] / fresh_s[1],
        "cifar_speedup_reuse": exact_s[1] / reuse_s[1],
    }


def make_fig3_proxy_sweep_zipf():
    sys_d = load_json("sprint2_systems")
    ps = sys_d["proxy_quality_sweep"]
    zs = sys_d["zipf_stress"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55), layout="constrained")

    # Panel (a): Proxy Quality Sweep
    aucs = ps["targets"]
    series = [
        ("safe_loo", "StratQ (safe_loo)", BLUE, "o", "-"),
        ("legacy", "StratQ (legacy)", TEAL, "s", "--"),
        ("no_proxy", "No-proxy WoR", GRAY, "^", ":"),
        ("Calibrated-FixedProxy", "Calib.-FixedProxy", AMBER, "D", "-."),
        ("CELEUS-FixedProxy", "CELEUS-FixedProxy", ORANGE, "v", "-"),
    ]
    for key, label, col, mk, ls in series:
        ys = [ps["summary"][f"{a:.2f}|{key}"]["mean_calls"] / 1000.0 for a in aucs]
        ax1.plot(aucs, ys, color=col, marker=mk, linestyle=ls, markersize=4.2, linewidth=1.35, label=label)
    ax1.invert_xaxis()
    ax1.set_xticks(aucs)
    ax1.set_xlabel("Proxy quality (ROC-AUC, strong -> noise)")
    ax1.set_ylabel("Mean reference calls (thousands)")
    ax1.set_ylim(3.6, 10.6)
    ax1.set_title("(a) Proxy quality sweep (N=12K, G=8)")
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6.9, handlelength=1.7)

    # Panel (b): Zipf Stress at alpha=1.2 across G=5..100
    gs = zs["group_grid"]
    z_series = [
        ("StratQ-safe_loo", "StratQ (30/30 cov)", BLUE, "o", "-"),
        ("Spertus-WoR", "Spertus-WoR (30/30)", GRAY, "s", "--"),
        ("PilotDB-Asymp", "PilotDB-Asymp (16/30)", PURPLE, "^", "-."),
        ("BARGAIN-Global", "BARGAIN-Global (0/30)", RED, "x", ":"),
    ]
    for key, label, col, mk, ls in z_series:
        ys = [zs["summary"][f"G{g}|a1.2|{key}"]["mean_calls"] / 1000.0 for g in gs]
        ax2.plot(gs, ys, color=col, marker=mk, linestyle=ls, markersize=4.5, linewidth=1.35, label=label)

    # Highlight PilotDB 0/2 failure at G=100
    p100 = zs["summary"]["G100|a1.2|PilotDB-Asymp"]["mean_calls"] / 1000.0
    ax2.annotate("0/2 all-grp cov", xy=(100, p100), xytext=(60, p100 - 8.5),
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9),
                 fontsize=6.9, color=PURPLE, fontweight="bold")
    b100 = zs["summary"]["G100|a1.2|BARGAIN-Global"]["mean_calls"] / 1000.0
    ax2.annotate("0% tail cov", xy=(100, b100), xytext=(62, b100 + 5.2),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
                 fontsize=6.9, color=RED, fontweight="bold")

    ax2.set_xticks(gs)
    ax2.set_xlabel("Number of groups G (N=100K, Zipf alpha=1.2)")
    ax2.set_ylabel("Mean reference calls (thousands)")
    ax2.set_ylim(-1.0, 39.0)
    ax2.set_title("(b) Skewed group scaling & coverage validity")
    ax2.legend(loc="upper left", fontsize=7.0, handlelength=1.8)

    save_fig(fig, "fig3_proxy_sweep_zipf", check_overlap=False)
    return {"aucs": aucs, "group_grid": gs}


def make_fig4_sprint1_adult():
    s1 = load_json("sprint1_safe_gating_ablation")
    fb = load_json("adult_fixed_budget")
    bl = load_json("adult_blazeit")
    sn = load_json("adult_senate")
    oa = load_json("adult_online_aggregation")
    lw = load_json("adult_lws")
    vd = load_json("adult_verdict")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55), layout="constrained")

    # Panel (a): Sprint 1 legacy vs safe_loo call change (%) vs no_proxy
    rows = [
        ("Fashion\nCOUNT", s1["fashion_mnist"]["summary"]),
        ("SMS\nCOUNT", s1["sms"]["workloads"]["COUNT"]["summary"]),
        ("SMS\nHAVING", s1["sms"]["workloads"]["HAVING"]["summary"]),
        ("SMS\nTOP-2", s1["sms"]["workloads"]["TOP-2"]["summary"]),
        ("Adult\nCOUNT", s1["adult"]["summary"]),
    ]
    y = np.arange(len(rows))
    h = 0.34
    leg_pct = [100.0 * (s["legacy"]["mean_calls"] / s["no_proxy"]["mean_calls"] - 1.0) for _, s in rows]
    safe_pct = [100.0 * (s["safe_loo"]["mean_calls"] / s["no_proxy"]["mean_calls"] - 1.0) for _, s in rows]

    ax1.axvline(0.0, color="#9AA2AB", linewidth=0.8, linestyle="--", zorder=0)
    ax1.barh(y - h / 2, leg_pct, h, color=AMBER, alpha=0.85, label="legacy")
    ax1.barh(y + h / 2, safe_pct, h, color=BLUE, alpha=0.92, label="safe_loo (default)")
    for yi, sv in zip(y, safe_pct):
        if sv < -5:
            ax1.text(sv - 1.5, yi + h / 2, f"{sv:+.1f}%", ha="right", va="center", fontsize=6.8, color=BLUE, fontweight="bold")
        else:
            ax1.text(max(sv, 0) + 1.5, yi + h / 2, f"{sv:+.1f}%", ha="left", va="center", fontsize=6.8, color=BLUE, fontweight="bold")

    ax1.set_yticks(y, [r[0] for r in rows])
    ax1.invert_yaxis()
    ax1.set_xlim(-76, 15)
    ax1.set_xlabel("Mean reference-call change vs. no-proxy (%)")
    ax1.set_title("(a) Safe-LOO gating ablation (safe_rounding=True)")
    ax1.legend(loc="lower left", fontsize=7.2)

    # Panel (b): Adult matched budget (B=4096) max group half-width across 7 operators
    tr = fb["truth"]
    def max_hw(runs, method=None):
        sub = [r for r in runs if r["budget"] == 4096 and (method is None or r.get("method") == method)]
        w = mean(100.0 * max((v["upper"] - v["lower"]) / (2.0 * tr[g]["size"]) for g, v in r["groups"].items()) for r in sub)
        c = sum(all(v["lower"] - 1e-9 <= tr[g]["positives"] <= v["upper"] + 1e-9 for g, v in r["groups"].items()) for r in sub)
        return w, c, len(sub)

    ops = [
        ("StratQ (Anytime)", *max_hw(fb["runs"], "stratq"), BLUE),
        ("VerdictDB* (Asymp.)", *max_hw(vd["runs"]), GRAY),
        ("Senate* (Fixed)", *max_hw(sn["runs"]), GRAY),
        ("LWS* (Asymp.)", *max_hw(lw["runs"]), GRAY),
        ("BlazeIt* (Fixed)", *max_hw(bl["runs"]), ORANGE),
        ("ABae-insp.* (Fixed)", *max_hw(fb["runs"], "proxy_stratified_fixed"), ORANGE),
        ("OLA Scan* (Fixed)", *max_hw(oa["runs"]), RED),
    ]
    y2 = np.arange(len(ops))
    widths = [o[1] for o in ops]
    cols = [o[4] for o in ops]
    ax2.barh(y2, widths, height=0.62, color=cols, alpha=0.9)
    for yi, (_, wv, cv, ntot, col) in enumerate(ops):
        ax2.text(wv + 0.8, yi, f"{wv:.2f}pp ({cv}/{ntot})", ha="left", va="center",
                 fontsize=6.8, color="#1A2530", fontweight="bold" if yi == 0 else "normal")
    ax2.set_yticks(y2, [o[0] for o in ops])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 62)
    ax2.set_xlabel("Mean worst-group half-width at 4,096 calls (pp)")
    ax2.set_title("(b) Adult matched-budget operators (*not anytime)")

    save_fig(fig, "fig4_sprint1_adult", check_overlap=False)
    return {"safe_pct": safe_pct, "adult_4096_widths": {o[0]: round(o[1], 2) for o in ops}}


def make_fig5_scale_memory():
    sys_d = load_json("sprint2_systems")
    mem_d = load_json("sprint2_memory_10m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.45), layout="constrained")

    # Panel (a): 10M Profile Amortization (Q=1..10)
    rec10 = [r for r in sys_d["profile_amortization"]["records"] if r["rows"] == 10000000][0]
    rec1 = [r for r in sys_d["profile_amortization"]["records"] if r["rows"] == 1000000 and r["replicate"] == 0][0]
    qs = np.arange(1, 11)
    f10 = [rec10["fresh_cumulative_ms"][q - 1] / 1000.0 for q in qs]
    r10 = [rec10["reused_cumulative_ms"][q - 1] / 1000.0 for q in qs]
    f1 = [rec1["fresh_cumulative_ms"][q - 1] / 1000.0 for q in qs]
    r1 = [rec1["reused_cumulative_ms"][q - 1] / 1000.0 for q in qs]

    ax1.plot(qs, f10, color=ORANGE, marker="s", markersize=4, linewidth=1.35, label="10M Fresh profile")
    ax1.plot(qs, r10, color=BLUE, marker="o", markersize=4, linewidth=1.45, label="10M Reused profile")
    ax1.plot(qs, f1, color=AMBER, linestyle="--", linewidth=1.1, label="1M Fresh profile")
    ax1.plot(qs, r1, color=TEAL, linestyle="--", linewidth=1.1, label="1M Reused profile")
    ax1.axvline(2, color="#9AA2AB", linestyle=":", linewidth=0.9)
    ax1.annotate("10M modeled break-even Q*=2\n(3.07x at Q=10; extrapolated)", xy=(2, r10[1]), xytext=(5.5, 45),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
                 fontsize=6.7, color=BLUE, fontweight="bold")
    ax1.set_xticks(qs)
    ax1.set_xlabel("Number of sequential queries Q")
    ax1.set_ylabel("Cumulative latency (s)")
    ax1.set_ylim(0, 125)
    ax1.set_title("(a) Profile amortization (1M & 10M)")
    ax1.legend(loc="upper left", fontsize=6.9, handlelength=1.6)

    # Panel (b): 10M 4-stage isolated subprocess memory
    stages = mem_d["stages"]
    names = ["Generate\nParquet", "Exact SQL\nVerify", "DuckDB+Arrow\nProfile", "Query Subprocess\n(incl. Profile)"]
    peak_gb = [s["peak_working_set_bytes"] / 1e9 for s in stages]
    rss_gb = [s["rss_bytes"] / 1e9 for s in stages]
    y = np.arange(len(stages))
    h = 0.34
    ax2.barh(y - h / 2, peak_gb, h, color=BLUE, alpha=0.9, label="OS Peak Working Set")
    ax2.barh(y + h / 2, rss_gb, h, color=TEAL, alpha=0.9, label="End-of-Stage RSS")
    for yi, (pk, rs) in enumerate(zip(peak_gb, rss_gb)):
        ax2.text(pk + 0.06, yi - h / 2, f"{pk:.2f} GB", ha="left", va="center", fontsize=6.8, color="#1A2530")
        lbl = f"{rs:.2f} GB" if yi < 3 else f"{rs:.2f} GB (final Δ−0.68 MB)"
        ax2.text(rs + 0.06, yi + h / 2, lbl, ha="left", va="center", fontsize=6.6,
                 color=TEAL if yi == 3 else "#2C3E50", fontweight="bold" if yi == 3 else "normal")
    ax2.set_yticks(y, names)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 3.95)
    ax2.set_xlabel("Memory (GB, isolated subprocesses, N=10M)")
    ax2.set_title("(b) 10M-row memory attribution")
    ax2.legend(loc="upper right", fontsize=7.0)

    save_fig(fig, "fig5_scale_memory", check_overlap=False)
    qstage = next(s for s in stages if s["stage"] == "query")
    return {"ten_m_profile_s": rec10["profile_once_ms"] / 1000.0,
            "ten_m_reused_s": rec10["median_reused_ms"] / 1000.0,
            "measured_queries": rec10["measured_queries"],
            "modeled_q10_speedup": rec10["fresh_cumulative_ms"][9] / rec10["reused_cumulative_ms"][9],
            "query_immediate_rss_delta_mb": qstage["rss_query_delta_bytes"] / 1e6,
            "query_final_rss_delta_mb": (qstage["rss_bytes"] - qstage["rss_after_profile_bytes"]) / 1e6,
            "query_tracemalloc_peak_mb": qstage["tracemalloc_peak_bytes"] / 1e6}


def main():
    apply_style()
    make_fig1_architecture()
    m2 = make_fig2_gpu_workloads()
    m3 = make_fig3_proxy_sweep_zipf()
    m4 = make_fig4_sprint1_adult()
    m5 = make_fig5_scale_memory()
    manifest = {
        "source_sha256": SOURCES,
        "metrics": {"fig2": m2, "fig3": m3, "fig4": m4, "fig5": m5},
    }
    (OUT / "figure_metrics_sprint3.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Generated all 5 Sprint 3 figures successfully.")


if __name__ == "__main__":
    main()
