"""All V3 figure/table artifacts. Missing results are explicitly marked missing."""
from __future__ import annotations
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from .utils import save_json

FIG_NAMES = {
    "Fig1": "Equal-K masks, unequal two-hop evidence (schematic)",
    "Fig2": "P1/P2 information boundaries (protocol diagram)",
    "Fig3": "RIMES teacher-student and normalized KL (method diagram)",
    "Fig4": "P1 fixed-test MRR under missingness (RQ1 empirical)",
    "Fig5": "Evidence survival vs ranking degradation (RQ2 empirical)",
    "Fig6": "P2 held-out family robustness (RQ3 empirical)",
    "Fig7": "Ablation and computational cost tradeoffs (efficiency frontier)"
}

TABLE_NAMES = {
    "Table0": "Prior-work novelty boundary",
    "Table1": "Datasets and protocol metadata",
    "Table2": "Mask profiles and matched null residuals",
    "Table3": "P1 training-observation effects and paired comparisons",
    "Table4": "RQ2 matched evidence and association analysis",
    "Table5": "P2 robust results, clean/held-out tradeoffs and GPU cost",
    "Table6": "RIMES ablation and evidence allocation controls"
}


def _node(ax, x, y, text, w=0.25, h=0.15):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", edgecolor="black", facecolor="white", linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def fig1(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    pos = {"h": (0, 0.50), "x": (1, 0.80), "y": (1, 0.20), "t": (2, 0.50), "u": (0, -0.22), "v": (1, -0.22), "w": (2, -0.22)}
    edges = [("h", "x"), ("x", "t"), ("h", "y"), ("y", "t"), ("u", "v"), ("v", "w")]
    for i, (ax, removed, title) in enumerate(zip(axes, [{4, 5}, {0, 2}],
            ["Mask A: peripheral deletions", "Mask B: motif-supported deletions"])):
        for j, (a, b) in enumerate(edges):
            x1, y1 = pos[a]; x2, y2 = pos[b]
            c = "firebrick" if j in removed else "0.25"
            ls = "--" if j in removed else "-"
            ax.annotate("", xy=(x2 - 0.10, y2), xytext=(x1 + 0.10, y1), arrowprops=dict(arrowstyle="->", color=c, lw=2, linestyle=ls))
        for n, (x, y) in pos.items():
            ax.scatter([x], [y], s=700, c="white", edgecolors="black", zorder=3)
            ax.text(x, y, n, ha="center", va="center", fontsize=11, zorder=4)
        ax.set_xlim(-.4, 2.4); ax.set_ylim(-.55, 1.2); ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.text(0.5, -.10, f"K=2; surviving h->t witnesses: {2 if i==0 else 0}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
    fig.suptitle("Fig. 1  Same edge-removal budget does not imply same support survival", fontsize=13)
    fig.text(.5, .015, "SCHEMATIC ONLY. The nodes and counts illustrate the definition; they are not experimental measurements.",
             ha="center", fontsize=8)
    _save(fig, out, "Fig1")


def fig2(out):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _node(ax, .04, .68, "P1: Ttr -> Tobs(m)", .24, .14)
    _node(ax, .37, .68, "Fresh theta(m,seed)", .24, .14)
    _node(ax, .71, .68, "Ranking same Qtest", .24, .14)
    _arrow(ax, (.30, .75), (.36, .75)); _arrow(ax, (.63, .75), (.70, .75))
    _node(ax, .04, .31, "P2: Ttr = C + L", .24, .14)
    _node(ax, .37, .37, "Train theta on L", .23, .12)
    _node(ax, .37, .18, "Context C -> Cm", .23, .12)
    _node(ax, .71, .31, "Frozen theta, Qtest", .24, .14)
    _arrow(ax, (.30, .38), (.36, .43)); _arrow(ax, (.30, .34), (.36, .24))
    _arrow(ax, (.62, .43), (.70, .39)); _arrow(ax, (.62, .25), (.70, .37))
    ax.text(.50, .93, "P1 = training-observation perturbation; P2 = fixed-supervision inference-context shift", ha="center", fontsize=12)
    ax.plot([.025, .97], [.57, .57], ls="--", color="0.6", lw=1.2)
    ax.text(.50, .07, "Tva: development/validation only. Tte: sealed until evaluation; full truth pool only in final evaluator.", ha="center", fontsize=10)
    _save(fig, out, "Fig2")


def fig3(out):
    fig, ax = plt.subplots(figsize=(12, 5.1))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _node(ax, .04, .66, "C, same Kq", .21, .13); _node(ax, .37, .66, "Teacher (stopgrad)", .25, .13)
    _node(ax, .04, .28, "Cm, same Kq", .21, .13); _node(ax, .37, .28, "Student (gradient)", .25, .13)
    _node(ax, .72, .66, "Gate g(q)", .22, .13); _node(ax, .72, .27, "Rank CE on L", .22, .13)
    _arrow(ax, (.27, .72), (.36, .72)); _arrow(ax, (.27, .35), (.36, .35))
    _arrow(ax, (.64, .72), (.71, .72)); _arrow(ax, (.64, .35), (.71, .35))
    _node(ax, .37, .025, "d(q): W(C) -> W(Cm); normalize sum(w)*KL + rank CE", .56, .11)
    _arrow(ax, (.50, .27), (.50, .15)); _arrow(ax, (.83, .26), (.83, .15)); _arrow(ax, (.83, .65), (.64, .14))
    ax.text(.5, .94, "Fig. 3  Query-aligned teacher/student distributions with evidence survival and KL mass control", ha="center", fontsize=12)
    _save(fig, out, "Fig3")


def fig4(input_dir, out):
    """Fig 4: P1 Fixed-Test MRR under Missingness (RQ1 empirical)."""
    inp = Path(input_dir) / "results.csv"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    if not inp.exists() or inp.stat().st_size == 0:
        ax.text(.5, .5, "NOT COMPUTED\nRun P1 on CoDEx-S/M.", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        df = pd.read_csv(inp)
        p1 = df[df.protocol == "P1"]
        if not len(p1):
            ax.text(.5, .5, "NOT COMPUTED: no P1 records", ha="center", transform=ax.transAxes)
        else:
            fam_order = ["clean", "M1", "M2", "M4", "Null-M4"]
            colors = {"clean": "#2ca02c", "M1": "#1f77b4", "M2": "#ff7f0e", "M4": "#d62728", "Null-M4": "#9467bd"}
            p1_sub = p1[(p1.dataset == "s") & (p1.method == "complex")]
            if not len(p1_sub): p1_sub = p1
            agg = p1_sub.groupby(["rho", "family"])["MRR"].agg(["mean", "std", "count"]).reset_index()
            rhos = sorted(agg.rho.unique())
            x = np.arange(len(fam_order))
            width = 0.35 if len(rhos) > 1 else 0.5
            for i, r in enumerate(rhos):
                r_agg = agg[agg.rho == r].set_index("family")
                means = [r_agg.loc[f, "mean"] if f in r_agg.index else 0 for f in fam_order]
                stds = [r_agg.loc[f, "std"] if (f in r_agg.index and not np.isnan(r_agg.loc[f, "std"])) else 0 for f in fam_order]
                pos = x + (i - (len(rhos) - 1) / 2) * width
                ax.bar(pos, means, width, yerr=stds, capsize=4, label=f"rho = {r:.2f}",
                       color=[colors.get(f, "grey") for f in fam_order], alpha=0.85 if i==0 else 0.65)
                for px, m in zip(pos, means):
                    if m > 0: ax.text(px, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(fam_order, fontsize=10)
            ax.set_ylabel("Filtered Test MRR (Head/Tail)", fontsize=11)
            ax.set_title("Fig. 4  RQ1: P1 Retrained Model Performance under Missingness Families (CoDEx-S ComplEx)", fontsize=11)
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
    _save(fig, out, "Fig4")


def _rq2_adjusted(df):
    rows = df[(df["family"] == "M4") & (df["method"] == ("erm" if "erm" in set(df.method) else "rimes"))].copy()
    if len(rows) == 0: return rows, None
    rows = rows[~rows["no_support"].astype(bool)].dropna(subset=["damage", "rr_change", "W0"])
    if len(rows) < 15 or rows.damage.nunique() < 3: return rows, None
    controls = pd.get_dummies(rows["r"].astype(str), prefix="r", dtype=float)
    controls["log_head_deg"] = np.log1p(rows["head_degree"].astype(float))
    controls["log_tail_deg"] = np.log1p(rows["tail_degree"].astype(float))
    controls["log_W0"] = np.log1p(rows["W0"].astype(float))
    c = controls.to_numpy(); y = rows["rr_change"].to_numpy(dtype=float)
    regression = Ridge(alpha=1., fit_intercept=True).fit(c, y)
    rows["adjusted_rr_change"] = y - regression.predict(c) + float(y.mean())
    return rows, regression


def fig5(input_dir, out):
    """Fig 5: Evidence Survival vs Ranking Degradation (RQ2 empirical)."""
    inp = Path(input_dir) / "rq2_queries.csv"
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    if not inp.exists() or inp.stat().st_size == 0:
        ax.text(.5, .5, "NOT COMPUTED\nRun P2 on CoDEx-S/M and export per-query ranks.", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        df = pd.read_csv(inp)
        if not len(df) or "damage" not in df.columns:
            ax.text(.5, .5, "NOT COMPUTED: no RQ2 measurements", ha="center", transform=ax.transAxes)
        else:
            rows, regression = _rq2_adjusted(df)
            if regression is None:
                ax.text(.5, .5, f"NOT ESTIMABLE: insufficient supported variation\nSupported query-direction pairs: {len(rows)}",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                bins = [0, 1.e-10, .2, .4, .6, .8, 1.0000001]
                labels = ["0", "(0,.2]", "(.2,.4]", "(.4,.6]", "(.6,.8]", "(.8,1]"]
                rows["bucket"] = pd.cut(rows.damage, bins=bins, labels=labels, include_lowest=True)
                grouped = rows.groupby("bucket", observed=False).agg(value=("adjusted_rr_change", "mean"), n=("adjusted_rr_change", "count")).reset_index()
                grouped = grouped[grouped.n > 0]
                ax.plot(grouped.bucket.astype(str), grouped.value, marker="o", lw=1.8, color="#1f77b4")
                for x, (_, item) in enumerate(grouped.iterrows()):
                    ax.annotate(f"n={int(item.n)}", (x, item.value), xytext=(2, 8), textcoords="offset points", fontsize=8)
                ax.axhline(0, linestyle="--", color="0.5")
                ax.set_ylabel("Degree/relation/W0-adjusted RR change (shift - clean)", fontsize=10)
                ax.set_xlabel("Two-hop evidence damage d (train-visible templates)", fontsize=10)
                ax.tick_params(axis="x", labelrotation=20)
                ax.text(.01, .01, "Descriptive association only; query pairs are not independent runs.", transform=ax.transAxes, fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Fig. 5  RQ2: Evidence Damage versus Adjusted Ranking Change", fontsize=11)
    _save(fig, out, "Fig5")


def fig6(input_dir, out):
    """Fig 6: P2 Held-out Family Robustness (RQ3 empirical)."""
    inp = Path(input_dir) / "results.csv"
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if not inp.exists() or inp.stat().st_size == 0:
        ax.text(.5, .5, "NOT COMPUTED\nRun P2 on CoDEx-S/M.", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        df = pd.read_csv(inp)
        p2 = df[df.protocol == "P2"]
        if not len(p2):
            ax.text(.5, .5, "NOT COMPUTED: no P2 records", ha="center", transform=ax.transAxes)
        else:
            p2_s = p2[p2.dataset == "s"]
            if not len(p2_s): p2_s = p2
            methods = ["erm", "dropedge", "uniform", "gate_only", "degree", "relation", "shuffled_d", "group_dro", "rimes", "graph_off"]
            methods = [m for m in methods if m in p2_s.method.unique()]
            if not methods: methods = sorted(p2_s.method.unique())
            piv = p2_s.groupby(["method", "family"])["MRR"].mean().unstack(fill_value=0)
            x = np.arange(len(methods))
            width = 0.26
            for idx, (fam, col, alpha) in enumerate([("clean", "#2ca02c", 0.9), ("M3", "#ff7f0e", 0.8), ("M4", "#d62728", 0.8)]):
                vals = [piv.loc[m, fam] if (m in piv.index and fam in piv.columns) else 0 for m in methods]
                ax.bar(x + (idx - 1) * width, vals, width, label=fam, color=col, alpha=alpha)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel("Filtered Test MRR", fontsize=10)
            ax.set_title("Fig. 6  RQ3: P2 Model Robustness across Clean and Held-Out Missingness Families (M3/M4)", fontsize=11)
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
    _save(fig, out, "Fig6")


def fig7(input_dir, out):
    """Fig 7: Ablation and Computational Cost Tradeoffs."""
    inp = Path(input_dir) / "results.csv"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    if not inp.exists() or inp.stat().st_size == 0:
        ax.text(.5, .5, "NOT COMPUTED\nRun P2 on CoDEx-S/M.", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        df = pd.read_csv(inp)
        p2 = df[df.protocol == "P2"]
        if not len(p2):
            ax.text(.5, .5, "NOT COMPUTED: no P2 records", ha="center", transform=ax.transAxes)
        else:
            agg = p2.groupby("method").agg(
                train_time=("train_time_s", "mean"),
                clean_mrr=("MRR", lambda s: s[p2.loc[s.index, "family"] == "clean"].mean() if (p2.loc[s.index, "family"] == "clean").any() else 0),
                m4_mrr=("MRR", lambda s: s[p2.loc[s.index, "family"] == "M4"].mean() if (p2.loc[s.index, "family"] == "M4").any() else 0)
            ).reset_index()
            for _, row in agg.iterrows():
                m = row["method"]
                color = "red" if m == "rimes" else ("green" if m == "uniform" else "blue")
                ax.scatter(row["train_time"], row["m4_mrr"], s=100, color=color, alpha=0.8, edgecolors="black")
                ax.annotate(m, (row["train_time"], row["m4_mrr"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
            ax.set_xlabel("Training Time (seconds per run)", fontsize=10)
            ax.set_ylabel("Held-Out M4 MRR", fontsize=10)
            ax.set_title("Fig. 7  Ablation Efficiency Frontier: Computational Budget vs. Robustness", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.6)
    _save(fig, out, "Fig7")


def _fmt_sd(vals):
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if not len(vals): return "NOT RUN"
    return f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}" if len(vals) > 1 else f"{vals.mean():.4f} (n=1)"


def _paired_ci(vals):
    vals = np.asarray(vals, dtype=float)
    if not len(vals): return "NOT RUN"
    if len(vals) < 3: return f"{vals.mean():+.4f} (n={len(vals)}; CI not established)"
    mean = vals.mean(); half = student_t.ppf(.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    return f"{mean:+.4f} [{mean-half:+.4f},{mean+half:+.4f}]"


def _table0():
    return pd.DataFrame([
      ["Dobreva & Sagi (ISWC 2026)", "Property graph missingness, node classification", "Fixed-query KG entity ranking"],
      ["InferWiki (ACL 2021)", "Inference evidence in KGC benchmarks", "Paired missingness-family stress test"],
      ["SRP-KGC (IJCAI 2025)", "KGC with unavailable explicit reasoning paths", "Budget/degree-matched train-only evidence weights"],
      ["StructKGC (EMNLP 2024)", "Structure-aware contrastive consistency", "Evidence-damage-specific KL allocation"],
      ["PAAL (IJCAI 2026)", "Continual KG embedding under structural shift", "Frozen static predictor with deployment context masks"],
      ["RACA (Sci Rep 2026)", "Sparse relation-aware aggregation", "Held-out mask families and fixed-L test"]],
      columns=["prior_work", "already_studied", "specific_scope_to_test"])


def _table1(input_dir):
    fn = Path(input_dir) / "protocol.json"
    multi = Path(input_dir) / "dataset_protocols.json"
    if multi.exists(): protos = json.loads(multi.read_text(encoding="utf8"))
    elif fn.exists(): protos = [json.loads(fn.read_text(encoding="utf8"))]
    else: return pd.DataFrame([{"status": "NOT RUN: use `rimes run`"}])
    rows = []
    for x in protos:
        c = x["official_counts"]
        rows.append({"dataset": c["source"].split("/")[-1],
           **{k: c[k] for k in ("entities", "relations", "train", "valid", "test")},
           "C": x["C_size"], "L": x["L_size"], "scope": x["scope"],
           "P1": "independent retraining", "P2": "fixed labels"})
    return pd.DataFrame(rows)


def _table2(input_dir):
    fn = Path(input_dir) / "mask_profiles.csv"
    if not fn.exists() or not fn.stat().st_size: return pd.DataFrame([{"status": "NOT RUN"}])
    df = pd.read_csv(fn)
    keep = [x for x in ["dataset", "protocol", "rho", "family", "K", "relation_js_vs_graph",
        "head_degree_js_vs_graph", "tail_degree_js_vs_graph", "new_isolates",
        "removed_twohop_pair_count", "removed_twohop_lift_vs_independent_budget", "grouped_fraction",
        "residual_single_edges", "forced_edges_fraction", "relation_js_vs_m4", "head_degree_js_vs_m4",
        "tail_degree_js_vs_m4", "witness_retained_mean"] if x in df.columns]
    nums = [x for x in keep if x not in ("dataset", "protocol", "rho", "family")]
    return df[keep].groupby(["dataset", "protocol", "rho", "family"], dropna=False)[nums].mean().reset_index()


def _table3(input_dir):
    fn = Path(input_dir) / "results.csv"
    if not fn.exists() or not fn.stat().st_size: return pd.DataFrame([{"status": "NOT RUN"}])
    df = pd.read_csv(fn)
    df = df[df.protocol == "P1"]
    if not len(df): return pd.DataFrame([{"status": "NOT RUN: no P1 results"}])
    rows = []
    for (dataset, model, rho), g in df.groupby(["dataset", "method", "rho"]):
        row = {"dataset": dataset, "model": model, "rho": rho, "num_runs": len(g.seed.unique()), "scope": g.dataset_scope.iloc[0]}
        for fam in ("clean", "M1", "M2", "M4", "Null-M4"):
            row[f"{fam}_MRR"] = _fmt_sd(g.loc[g.family == fam, "MRR"])
            row[f"{fam}_macroMRR"] = _fmt_sd(g.loc[g.family == fam, "relation_macro_mrr"])
        piv = g.pivot_table(index="seed", columns="family", values="MRR")
        for a, b, name in (("M4", "M1", "M4_minus_M1"), ("M4", "Null-M4", "M4_minus_NullM4")):
            row[name] = _paired_ci((piv[a] - piv[b]).dropna().values) if a in piv and b in piv else "NOT RUN"
        row["avg_train_s"] = round(float(g.train_time_s.mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _table4(input_dir):
    """Table 4: RQ2 Matched Evidence and Association Analysis."""
    inp = Path(input_dir) / "rq2_queries.csv"
    if not inp.exists() or not inp.stat().st_size: return pd.DataFrame([{"status": "NOT RUN"}])
    df = pd.read_csv(inp)
    if not len(df): return pd.DataFrame([{"status": "NOT RUN: no RQ2 records"}])
    rows = []
    for (dataset, fam), g in df.groupby(["dataset", "family"]):
        n_total = len(g)
        n_nosup = int(g["no_support"].sum())
        sup = g[~g["no_support"].astype(bool)]
        row = {
            "dataset": dataset, "family": fam, "queries": n_total,
            "no_support_queries": n_nosup, "no_support_pct": f"{n_nosup/max(1,n_total)*100:.1f}%",
            "mean_W0_supported": f"{sup['W0'].mean():.2f}" if len(sup) else "N/A",
            "mean_damage_supported": f"{sup['damage'].mean():.4f}" if len(sup) else "N/A",
            "mean_rr_change_all": f"{g['rr_change'].mean():+.4f}",
            "mean_rr_change_nosup": f"{g.loc[g.no_support.astype(bool), 'rr_change'].mean():+.4f}" if n_nosup else "N/A",
            "mean_rr_change_sup": f"{sup['rr_change'].mean():+.4f}" if len(sup) else "N/A"
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _table5(input_dir):
    """Table 5: P2 Robust Results, clean/held-out tradeoffs and GPU cost."""
    fn = Path(input_dir) / "results.csv"
    if not fn.exists() or not fn.stat().st_size: return pd.DataFrame([{"status": "NOT RUN"}])
    df = pd.read_csv(fn); df = df[df.protocol == "P2"]
    if not len(df): return pd.DataFrame([{"status": "NOT RUN: no P2 results"}])
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "num_runs": len(g.seed.unique()), "scope": g.dataset_scope.iloc[0]}
        for fam in ("clean", "M3", "M4", "Null-M4"):
            row[f"{fam}_MRR"] = _fmt_sd(g.loc[g.family == fam, "MRR"])
        pivot = g.pivot_table(index="seed", columns="family", values="MRR")
        row["worst_family_MRR"] = _fmt_sd(pivot[[x for x in ("M3", "M4") if x in pivot]].min(axis=1))
        if all(x in pivot for x in ("clean", "M3", "M4")):
            row["worst_clean_drop"] = _paired_ci((pivot[["M3", "M4"]].min(axis=1) - pivot["clean"]).dropna())
        else: row["worst_clean_drop"] = "NOT RUN"
        row["train_s"] = _fmt_sd(g.groupby("seed").train_time_s.first())
        row["eval_latency_ms"] = _fmt_sd(g.loc[g.family == "M4", "fullrank_latency_ms_amortized"])
        row["train_GPU_hours"] = _fmt_sd(g.groupby("seed").train_gpu_hours_estimate.first()) if "train_gpu_hours_estimate" in g else "NOT RECORDED"
        row["steps"] = _fmt_sd(g.groupby("seed").steps.first())
        rows.append(row)
    return pd.DataFrame(rows)


def _table6(input_dir):
    """Table 6: RIMES Ablation and Evidence Allocation Controls."""
    t = _table5(input_dir)
    if "method" not in t: return pd.DataFrame([{"status": "NOT RUN"}])
    table = t[[x for x in ["dataset", "method", "clean_MRR", "M3_MRR", "M4_MRR", "worst_family_MRR", "train_s", "eval_latency_ms", "scope"] if x in t.columns]].copy()
    table["ablation_purpose"] = table.method.map({
        "erm": "no augmentation (clean baseline)",
        "dropedge": "random edge masking without consistency",
        "relation_dropedge": "relation-targeted masking alone",
        "uniform": "same eligible support, equal KL weights",
        "uniform_full": "all labels KL (unmatched eligibility)",
        "gate_only": "teacher margin gate without damage d",
        "degree": "endpoint degree-bucket inverse weights",
        "relation": "relation frequency inverse weights",
        "shuffled_d": "stratum-preserving damage shuffle (matched null)",
        "group_dro": "worst-relation group robust objective",
        "rimes": "evidence loss x teacher gate (proposed)",
        "graph_off": "no graph message passing (feature-only)"
    })
    return table


def render_outputs(input_dir, output_dir):
    inp = Path(input_dir); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"; tabs = out / "tables"
    figs.mkdir(exist_ok=True); tabs.mkdir(exist_ok=True)
    fig1(figs); fig2(figs); fig3(figs); fig4(inp, figs); fig5(inp, figs); fig6(inp, figs); fig7(inp, figs)
    tables = {
        "Table0": _table0(), "Table1": _table1(inp), "Table2": _table2(inp),
        "Table3": _table3(inp), "Table4": _table4(inp), "Table5": _table5(inp), "Table6": _table6(inp)
    }
    for name, df in tables.items():
        df.to_csv(tabs / f"{name}.csv", index=False)
        latex = df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
        (tabs / f"{name}.tex").write_text("\\begingroup\\scriptsize\n\\resizebox{\\linewidth}{!}{%\n" + latex + "}\n\\endgroup\n", encoding="utf8")
    scope = "NOT RUN"
    if (inp / "summary.json").exists():
        scope = json.loads((inp / "summary.json").read_text(encoding="utf8")).get("scope", "UNKNOWN")
    status = {
        "scope": scope,
        "figures": {k: {"source": v, "file": f"figures/{k}.png"} for k, v in FIG_NAMES.items()},
        "tables": {k: {"source": v, "file": f"tables/{k}.csv"} for k, v in TABLE_NAMES.items()},
        "warning": "Fig1-3 are schematic; Fig4-7 are empirical data-driven. Not-run tables are marked NOT RUN."
    }
    save_json(out / "artifact_manifest.json", status)
    text = [
        "# RIMES-KG automated figure/table report", "",
        f"Scope: **{scope}**", "",
        "All named outputs correspond to the V2/V3 proposal specifications.", "",
        "## Output inventory (Figures)",
    ]
    for k, v in FIG_NAMES.items(): text.append(f"- **{k}**: {v} (`figures/{k}.png`)")
    text.extend(["", "## Output inventory (Tables)"])
    for k, v in TABLE_NAMES.items(): text.append(f"- **{k}**: {v} (`tables/{k}.csv` & `.tex`)")
    text.extend([
        "", "## Interpretation boundary",
        "Official CoDEx-S/M counts and full ranking are required for formal claims.",
        "P1 and P2 use different supervision protocols; their scores must not be compared as a single leaderboard.",
        "No p-values or narrow confidence intervals are inferred from dependent triples."
    ])
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf8")
    return out
