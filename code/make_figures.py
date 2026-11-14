#!/usr/bin/env python3
"""make_figures.py -- figures, tables and the number registry for the technical report (paper).

    python code/make_figures.py            # from the project root, or any cwd

Reads ONLY result files under results/ (and the environment record results/environment.md):
    feasibility_matrix_detail.csv, metrics_accuracy.csv, latency_summary.csv, bench/latency/*/worker.json,
    raw/smoke/merge_parity_kev-0.8b.json, bench/accuracy/<combo>/rows.json,
    analysis/quant_metrics.csv, quant_paired_diffs.csv, quant_recalibration.csv, quant_decision_churn.csv,
    quant_rows_long.csv
Writes:
    paper/figures/fig{1..4}_*.{pdf,png}
    paper/tables/t*.{csv,tex}
    paper/tables/numbers.tex  (LaTeX macros: \\V{key} -> value)   and numbers.csv (key, value, source, locator)
    results/analysis/EXPLORATORY_churn_by_refconf.csv     (new, exploratory, not pre-registered)
    results/analysis/EXPLORATORY_reliability_bins.csv     (figure support, exploratory)
    results/analysis/EXPLORATORY_risk_coverage_points.csv (figure support, exploratory)
No number is typed by hand: every value in the figures/tables/macros is read or computed from the files above.
Requires: numpy, pandas, scipy, matplotlib.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
AN = RES / "analysis"
FIG = BASE / "paper" / "figures"
TAB = BASE / "paper" / "tables"
for d in (FIG, TAB, AN):
    d.mkdir(parents=True, exist_ok=True)

# Figure style: the rcParams set by the `figure-style` skill's apply_figure_style() (sizes 8/7/6), copied
# here so the script runs stand-alone.
STYLE = {"axes.labelsize": 8.0, "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
         "axes.titlelocation": "left", "axes.titlesize": 8.0, "figure.dpi": 200.0, "font.size": 8.0,
         "legend.fontsize": 7.0, "legend.frameon": False, "lines.linewidth": 1.2, "patch.linewidth": 0.6,
         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300.0,
         "xtick.labelsize": 6.0, "xtick.major.size": 3.0, "xtick.major.width": 0.6, "ytick.labelsize": 6.0,
         "ytick.major.size": 3.0, "ytick.major.width": 0.6}
plt.rcParams.update(STYLE)

# Colours. Precision (Fig. 2-4): reference dark grey, 8-bit blue, 4-bit orange (focal); bf16 variants light grey.
C_REF, C_8, C_4, C_BF = "#3a3a3a", "#3b75af", "#e1812c", "#a6a6a6"
# Backend (Fig. 1): MLX blue, PyTorch MPS purple, PyTorch CPU grey; alarm (guard / killed) red.
C_MLX, C_MPS, C_CPU, C_ALARM = "#3b75af", "#8c6bb1", "#8a8a8a", "#c44e52"

REFS = {"kev-0.8b": "kev-0.8b/fp32", "kev-4b": "kev-4b/full-bf16-merged"}
CC = {"kev-0.8b/fp32": "ks-fp", "kev-0.8b/bf16": "ks-bf", "kev-0.8b/full-bf16-merged": "ks-mg",
      "kev-0.8b/8bit": "ks-8", "kev-0.8b/4bit": "ks-4", "kev-4b/full-bf16-merged": "kl-mg",
      "kev-4b/8bit": "kl-8", "kev-4b/4bit": "kl-4"}
NICE = {"kev-0.8b/fp32": "Kev-0.8B fp32 (reference)", "kev-0.8b/bf16": "Kev-0.8B bf16 (adapter merged in memory)",
        "kev-0.8b/full-bf16-merged": "Kev-0.8B bf16 (merged checkpoint)", "kev-0.8b/8bit": "Kev-0.8B 8-bit",
        "kev-0.8b/4bit": "Kev-0.8B 4-bit", "kev-4b/full-bf16-merged": "Kev-4B bf16 merged (reference)",
        "kev-4b/8bit": "Kev-4B 8-bit", "kev-4b/4bit": "Kev-4B 4-bit"}
METRIC_NICE = {"acc": "Accuracy", "nll": "NLL", "brier": "Brier score", "ece15": "ECE (15 bins)",
               "ece10": "ECE (10 bins, Kev)", "mean_conf": "Mean confidence",
               "coverage_at_5pct_error": "Coverage at 5\\% error", "aurc": "AURC"}

# ----------------------------------------------------------------------------------------------------------
# number registry
REG = {}


def reg(key, value, source, locator):
    """Register a formatted value under `key`; it becomes \\V{key} in LaTeX and a row of numbers.csv."""
    if key in REG and REG[key]["value"] != value:
        raise ValueError(f"registry key {key} re-registered with a different value")
    REG[key] = {"key": key, "value": value, "source": source, "locator": locator}
    return value


def f4(x):
    return f"{x:.4f}"


def sgn4(x):
    """Signed 4-decimal value; a nonzero value that rounds to 0.0000 keeps its sign with 5 decimals."""
    if x == 0:
        return "0.0000"
    s = f"{abs(x):.4f}" if abs(x) >= 5e-5 else f"{abs(x):.5f}"
    return ("+" if x > 0 else "$-$") + s


def pfmt(p):
    """p-value text; bootstrap resolution is 1/10,000, Holm multiplies by <=10, so 0 -> '<0.001'."""
    if p == 0:
        return "$<$0.001"
    return f"{p:.4f}" if p < 0.1 else f"{p:.3f}"


def peq(p):
    """'= 0.135' or '< 0.001' for use after p / p_H."""
    t = pfmt(p)
    return t.replace("$<$", "$<$\\,") if t.startswith("$<$") else "\\,=\\," + t


def rel(p: Path):
    return str(p.relative_to(BASE))


# ----------------------------------------------------------------------------------------------------------
# load inputs
F_FZ = RES / "feasibility_matrix_detail.csv"
F_MA = RES / "metrics_accuracy.csv"
F_QM = AN / "quant_metrics.csv"
F_PD = AN / "quant_paired_diffs.csv"
F_RC = AN / "quant_recalibration.csv"
F_CH = AN / "quant_decision_churn.csv"
F_RL = AN / "quant_rows_long.csv"
F_LS = RES / "latency_summary.csv"
F_ENV = RES / "environment.md"
F_MP = RES / "raw" / "smoke" / "merge_parity_kev-0.8b.json"

fz = pd.read_csv(F_FZ)
ma = pd.read_csv(F_MA)
qm = pd.read_csv(F_QM).set_index("combo_id")
pdif = pd.read_csv(F_PD)
rc = pd.read_csv(F_RC).set_index("combo_id")
ch = pd.read_csv(F_CH).set_index("combo_id")
rl = pd.read_csv(F_RL)
ls = pd.read_csv(F_LS)
env_text = F_ENV.read_text()
mp = json.loads(F_MP.read_text())


def rows_json(combo):
    return json.loads((RES / "bench" / "accuracy" / combo.replace("/", "__") / "rows.json").read_text())


def population(rows):  # identical to code/analysis_quant.py (Kev's "clean" population)
    return {(r["id"], r["question"]): r for r in rows if r["variant"] == "clean" and r["source"] != "unknowable"}


# ----------------------------------------------------------------------------------------------------------
# feasibility numbers
FZ_ORDER = ["kev-0.8b/fp32", "kev-0.8b/fp32-cpu", "kev-0.8b/bf16", "kev-0.8b/full-bf16-merged", "kev-0.8b/8bit",
            "kev-0.8b/4bit", "kev-4b/adapter-bf16", "kev-4b/full-bf16-merged", "kev-4b/8bit", "kev-4b/4bit",
            "laya/fp32", "laya/fp16", "laya/fp32-cpu", "laya/fp16-laya-mlx", "laya-multilingual/fp32",
            "laya-multilingual/fp16", "laya-multilingual/fp16-laya-mlx", "nanojev/bf16-upstream", "nanojev/bf16",
            "nanojev/bf16-cpu", "anyjev-q0.8b/bf16", "anyjev-q0.8b/fp32", "anyjev-q0.8b/4bit", "anyjev-q4b/bf16",
            "anyjev-q4b/4bit", "jevmlx-q0.8b/bf16", "jevmlx-q0.8b/4bit", "jevmlx-q4b/bf16", "jevmlx-q4b/4bit"]
assert sorted(FZ_ORDER) == sorted(fz.combo_id), "feasibility combos changed; update FZ_ORDER"
fzi = fz.set_index("combo_id").loc[FZ_ORDER]
FAMILY = {"kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B", "laya": "Laya (ModernBERT-large)",
          "laya-multilingual": "Laya-multilingual (mmBERT-base)", "nanojev": "NanoJev (Qwen3-0.6B)",
          "anyjev-q0.8b": "AnyJev, Qwen3.5-0.8B-Base", "anyjev-q4b": "AnyJev, Qwen3.5-4B-Base",
          "jevmlx-q0.8b": "jevmlx, Qwen3.5-0.8B-Base", "jevmlx-q4b": "jevmlx, Qwen3.5-4B-Base"}
VARIANT = {"fp32": "fp32", "fp32-cpu": "fp32, CPU", "bf16-cpu": "bf16, CPU", "bf16": "bf16", "full-bf16-merged": "bf16, merged checkpoint",
           "8bit": "8-bit", "4bit": "4-bit", "adapter-bf16": "bf16, adapter path", "fp16": "fp16 autocast",
           "fp16-laya-mlx": "fp16, laya-mlx port", "bf16-upstream": "bf16, upstream code"}
BACKEND = {"mlx": "MLX", "torch-mps": "PyTorch MPS", "torch-cpu": "PyTorch CPU", "torch-cuda-only": "PyTorch (CUDA only)",
           "none": "--"}
n_total = len(fz)
n_run = int((fz.runs == "yes").sum())
reg("fz.total", str(n_total), rel(F_FZ), "row count")
reg("fz.runnable", str(n_run), rel(F_FZ), "count runs == yes")
reg("fz.nonrunnable", str(n_total - n_run), rel(F_FZ), "count runs == no")
n_cpu = int(((fz.runs == "yes") & (fz.backend == "torch-cpu")).sum())
reg("fz.cpu", str(n_cpu), rel(F_FZ), "count runs == yes & backend == torch-cpu")
reg("fz.accpool", str(n_run - n_cpu), rel(F_FZ), "runnable non-CPU combos (accuracy-eligible pool)")
reg("acc.ncombos", str(len(ma)), rel(F_MA), "row count")
reg("acc.notassessed", str(n_run - n_cpu - len(ma)), rel(F_FZ) + " ; " + rel(F_MA),
    "runnable non-CPU combos minus combos with accuracy runs")
for c in FZ_ORDER:
    r = fzi.loc[c]
    k = c.replace("/", ".")
    if pd.notna(r.peak_phys_footprint_gb):
        reg(f"fz.{k}.peak", f"{r.peak_phys_footprint_gb:.2f}", rel(F_FZ), f"{c}: peak_phys_footprint_gb")
    if pd.notna(r.load_s):
        reg(f"fz.{k}.load", f"{r.load_s:.1f}", rel(F_FZ), f"{c}: load_s")
    if pd.notna(r.latency_ms_single):
        reg(f"fz.{k}.first", f"{r.latency_ms_single:.1f}", rel(F_FZ), f"{c}: latency_ms_single")
    if pd.notna(r.latency_ms_warm):
        reg(f"fz.{k}.warm", f"{r.latency_ms_warm:.1f}", rel(F_FZ), f"{c}: latency_ms_warm")
    if pd.notna(r.swap_delta_gb):
        reg(f"fz.{k}.swap", f"{r.swap_delta_gb:.2f}", rel(F_FZ), f"{c}: swap_delta_gb")
    if pd.notna(r.min_sys_avail_gb):
        reg(f"fz.{k}.minavail", f"{r.min_sys_avail_gb:.2f}", rel(F_FZ), f"{c}: min_sys_avail_gb")
    if pd.notna(r.rss_peak_gb_parent):
        reg(f"fz.{k}.rss", f"{r.rss_peak_gb_parent:.2f}", rel(F_FZ), f"{c}: rss_peak_gb_parent")
    if pd.notna(r.mlx_peak_gb):
        reg(f"fz.{k}.mlxpeak", f"{r.mlx_peak_gb:.2f}", rel(F_FZ), f"{c}: mlx_peak_gb")
# guard limit / kill details parsed from the killed row
kill = fzi.loc["kev-4b/adapter-bf16"]
m = re.search(r"footprint ([\d.]+) GB > limit ([\d.]+) GB after ([\d.]+) s", kill.killed_reason)
GUARD_GB = float(m.group(2))
reg("guard.gb", m.group(2), rel(F_FZ), "kev-4b/adapter-bf16: killed_reason (limit)")
reg("kill.gb", m.group(1), rel(F_FZ), "kev-4b/adapter-bf16: killed_reason (footprint)")
reg("kill.s", m.group(3), rel(F_FZ), "kev-4b/adapter-bf16: killed_reason (time)")
# quantized backbone sizes (from notes)
for c in ["kev-0.8b/8bit", "kev-0.8b/4bit", "kev-4b/8bit", "kev-4b/4bit"]:
    mm = re.search(r"quantized (\d+) modules \((\d)-bit g(\d+)\), backbone ([\d.]+)->([\d.]+) GB", fzi.loc[c].notes)
    k = c.replace("/", ".")
    reg(f"fz.{k}.nmod", mm.group(1), rel(F_FZ), f"{c}: notes 'quantized N modules'")
    reg(f"fz.{k}.bbfrom", mm.group(4), rel(F_FZ), f"{c}: notes 'backbone X->Y GB' (X)")
    reg(f"fz.{k}.bbto", mm.group(5), rel(F_FZ), f"{c}: notes 'backbone X->Y GB' (Y)")
    reg(f"fz.{k}.group", mm.group(3), rel(F_FZ), f"{c}: notes 'gNN'")
# hardware / environment values
memsize = int(re.search(r"hw\.memsize: (\d+)", env_text).group(1))
RAM_GB = memsize / 1e9
reg("env.ramgb", f"{RAM_GB:.2f}", rel(F_ENV), "hw.memsize / 1e9")
reg("env.ramgib", f"{memsize / 2**30:.0f}", rel(F_ENV), "hw.memsize / 2^30")
mlx_ws = float(re.search(r"'max_recommended_working_set_size': (\d+)", env_text).group(1)) / 1e9
reg("env.mlxws", f"{mlx_ws:.2f}", rel(F_ENV), "MLX device_info max_recommended_working_set_size / 1e9")
reg("env.gpucores", re.search(r"Type: GPU.*?Total Number of Cores: (\d+)", env_text, re.S).group(1), rel(F_ENV),
    "SPDisplaysDataType Total Number of Cores")
cores = re.search(r"Total Number of Cores: (\d+) \((\d+) Performance and (\d+) Efficiency\)", env_text)
reg("env.cpucores", cores.group(1), rel(F_ENV), "SPHardware Total Number of Cores")
reg("env.pcores", cores.group(2), rel(F_ENV), "SPHardware Performance cores")
reg("env.ecores", cores.group(3), rel(F_ENV), "SPHardware Efficiency cores")
reg("env.macos", re.search(r"ProductVersion:\s+([\d.]+)", env_text).group(1), rel(F_ENV), "sw_vers ProductVersion")
for pkg in ["python", "mlx", "mlx-lm", "torch", "transformers", "anyjev", "laya", "jevmlx", "kev"]:
    v = re.search(rf"^- {re.escape(pkg)}: `([^`]+)`", env_text, re.M).group(1)
    reg(f"env.{pkg}", v, rel(F_ENV), f"Python and packages: {pkg}")
# merge parity
reg("mp.tensors", str(mp["bitwise_equal_tensors"]), rel(F_MP), "bitwise_equal_tensors")
reg("mp.compared", str(mp["compared"]), rel(F_MP), "compared")
reg("mp.elements", f"{mp['elements_total']:,}", rel(F_MP), "elements_total")
reg("mp.differing", str(mp["elements_differing"]), rel(F_MP), "elements_differing")

# ----------------------------------------------------------------------------------------------------------
# accuracy-run coverage
for _, r in ma.iterrows():
    k = CC[r.combo_id]
    reg(f"acc.{k}.req", str(int(r.n_requested)), rel(F_MA), f"{r.combo_id}: n_requested")
    reg(f"acc.{k}.ans", str(int(r.n_answered)), rel(F_MA), f"{r.combo_id}: n_answered")
    reg(f"acc.{k}.fail", str(int(r.n_failed)), rel(F_MA), f"{r.combo_id}: n_failed")
reg("acc.req", str(int(ma.n_requested.iloc[0])) if ma.n_requested.nunique() == 1 else "VAR", rel(F_MA),
    "n_requested (identical across combos)")
reg("acc.failtotal", str(int(ma.n_failed.sum())), rel(F_MA), "sum n_failed")
reg("pop.n", str(int(qm.n.iloc[0])) if qm.n.nunique() == 1 else "VAR", rel(F_QM), "n (identical across combos)")
reg("perm.n", str(int(qm.perm_n.iloc[0])), rel(F_QM), "perm_n")
_ref_rows = rl[rl.combo_id == "kev-0.8b/fp32"]
for t, n in _ref_rows.type.value_counts().items():
    reg(f"pop.type.{t}", str(int(n)), rel(F_RL), f"kev-0.8b/fp32 rows with type == {t}")

# point metrics
for c, k in CC.items():
    r = qm.loc[c]
    for mname, kk in [("acc", "acc"), ("nll", "nll"), ("brier", "brier"), ("ece10", "ece10"), ("ece15", "ece15"),
                      ("mean_conf", "conf"), ("coverage_at_5pct_error", "cov"), ("aurc", "aurc")]:
        reg(f"q.{k}.{kk}", f4(r[mname]), rel(F_QM), f"{c}: {mname}")
    reg(f"q.{k}.permflips", str(int(r.perm_flips)), rel(F_QM), f"{c}: perm_flips")
    # cross-check with Kev's own metric implementation
    assert abs(r.acc - r.kev_acc) < 1e-12 and abs(r.brier - r.kev_brier) < 1e-12 and abs(r.nll - r.kev_nll) < 1e-12

# paired differences
MK = {"acc": "acc", "nll": "nll", "brier": "brier", "ece10": "ece10", "ece15": "ece15", "mean_conf": "conf",
      "coverage_at_5pct_error": "cov", "aurc": "aurc"}
for _, r in pdif.iterrows():
    k = CC[r.combo_id]
    loc = f"{r.combo_id} vs {r.reference}: {r.metric}"
    if r.metric in MK:
        kk = MK[r.metric]
        reg(f"d.{k}.{kk}", sgn4(r["diff"]), rel(F_PD), loc + " diff")
        reg(f"lo.{k}.{kk}", sgn4(r.ci_low), rel(F_PD), loc + " ci_low")
        reg(f"hi.{k}.{kk}", sgn4(r.ci_high), rel(F_PD), loc + " ci_high")
        reg(f"pb.{k}.{kk}", pfmt(r.p_boot), rel(F_PD), loc + " p_boot")
        reg(f"ph.{k}.{kk}", pfmt(r.p_holm), rel(F_PD), loc + " p_holm")
    elif r.metric == "acc_mcnemar":
        reg(f"mc.{k}.b", str(int(r.ref_value)), rel(F_PD), loc + " ref_value (reference-only correct)")
        reg(f"mc.{k}.c", str(int(r.value)), rel(F_PD), loc + " value (variant-only correct)")
        reg(f"mc.{k}.disc", str(int(r.ref_value + r.value)), rel(F_PD), loc + " ref_value + value")
        reg(f"mc.{k}.p", pfmt(r.p_boot), rel(F_PD), loc + " p_boot (exact McNemar)")
        reg(f"mc.{k}.ph", pfmt(r.p_holm), rel(F_PD), loc + " p_holm")
    elif r.metric == "perm_flip_rate":
        nperm = int(qm.loc[r.combo_id].perm_n)
        reg(f"pf.{k}.ref", f"{round(r.ref_value * nperm)}/{nperm}", rel(F_PD), loc + " ref_value x perm_n")
        reg(f"pf.{k}.val", f"{round(r.value * nperm)}/{nperm}", rel(F_PD), loc + " value x perm_n")
        reg(f"pf.{k}.p", pfmt(r.p_boot), rel(F_PD), loc + " p_boot (exact McNemar)")
        reg(f"pf.{k}.ph", pfmt(r.p_holm), rel(F_PD), loc + " p_holm")
        reg(f"pf.{k}.disc", re.search(r"discordant (\d+/\d+)", r.note).group(1), rel(F_PD), loc + " note")
# number of tests per Holm family
fam_sizes = pdif.groupby("combo_id").size().unique()
assert len(fam_sizes) == 1
reg("holm.family", str(int(fam_sizes[0])), rel(F_PD), "rows per combo_id (Holm family size)")
# significant after Holm (alpha = .05)
sig = pdif[pdif.p_holm < 0.05]
reg("holm.nsig", str(len(sig)), rel(F_PD), "count p_holm < 0.05 over all rows")
reg("holm.ntests", str(len(pdif)), rel(F_PD), "row count")

# churn (existing CSV; exploratory per deviation log #13)
for c, r in ch.iterrows():
    k = CC[c]
    reg(f"ch.{k}.changed", str(int(r.changed)), rel(F_CH), f"{c}: changed")
    reg(f"ch.{k}.n", str(int(r.n)), rel(F_CH), f"{c}: n")
    reg(f"ch.{k}.pct", f"{100 * r.churn:.1f}", rel(F_CH), f"{c}: churn x 100")
    reg(f"ch.{k}.lo", f"{100 * r.ci_low:.1f}", rel(F_CH), f"{c}: ci_low x 100")
    reg(f"ch.{k}.hi", f"{100 * r.ci_high:.1f}", rel(F_CH), f"{c}: ci_high x 100")
    reg(f"ch.{k}.nhigh", str(int(r["n_ref_conf>=0.9"])), rel(F_CH), f"{c}: n_ref_conf>=0.9")
    reg(f"ch.{k}.chhigh", str(int(r["changed_among_ref_conf>=0.9"])), rel(F_CH),
        f"{c}: changed_among_ref_conf>=0.9")
bf_max = ch.loc[["kev-0.8b/bf16", "kev-0.8b/full-bf16-merged", "kev-0.8b/8bit", "kev-4b/8bit"], "churn"].max()
reg("ch.bf8max", f"{100 * bf_max:.1f}", rel(F_CH), "max churn x 100 over bf16 and 8-bit rows")

# recalibration
for c, k in CC.items():
    r = rc.loc[c]
    reg(f"rc.{k}.T", f"{r.extra_T_mean:.3f}", rel(F_RC), f"{c}: extra_T_mean")
    reg(f"rc.{k}.Tmin", f"{r.extra_T_min:.3f}", rel(F_RC), f"{c}: extra_T_min")
    reg(f"rc.{k}.Tmax", f"{r.extra_T_max:.3f}", rel(F_RC), f"{c}: extra_T_max")
    for col, kk in [("oof_nll", "nll"), ("oof_brier", "brier"), ("oof_ece15", "ece15"), ("oof_acc", "acc"),
                    ("oof_mean_conf", "conf")]:
        reg(f"rc.{k}.{kk}", f4(r[col]), rel(F_RC), f"{c}: {col}")

# ----------------------------------------------------------------------------------------------------------
# NEW (exploratory): churn by reference-confidence bin, recomputed from rows.json
BIN_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BIN_LABELS = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">=0.9"]
pops = {c: population(rows_json(c)) for c in CC}
cb_rows = []
for c in ch.index:
    ref = REFS[c.split("/")[0]]
    keys = sorted(pops[ref])
    assert set(keys) == set(pops[c])
    pref = np.array([max(pops[ref][q]["p"]) for q in keys])
    changed = np.array([int(np.argmax(pops[ref][q]["p"]) != np.argmax(pops[c][q]["p"])) for q in keys])
    # consistency with the existing churn CSV
    assert changed.sum() == int(ch.loc[c, "changed"]), (c, changed.sum())
    assert int((changed[pref >= 0.9]).sum()) == int(ch.loc[c, "changed_among_ref_conf>=0.9"])
    b = np.clip(np.digitize(pref, BIN_EDGES[1:-1], right=False), 0, len(BIN_LABELS) - 1)
    for i, lab in enumerate(BIN_LABELS):
        sel = b == i
        n, x = int(sel.sum()), int(changed[sel].sum())
        ci = binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact") if n else None
        cb_rows.append({"combo_id": c, "reference": ref, "ref_conf_bin": lab, "bin_low": BIN_EDGES[i],
                        "bin_high": BIN_EDGES[i + 1], "n": n, "changed": x, "churn": x / n if n else np.nan,
                        "ci_low_cp": ci.low if ci else np.nan, "ci_high_cp": ci.high if ci else np.nan,
                        "status": "EXPLORATORY (not pre-registered; deviation log #13)"})
cb = pd.DataFrame(cb_rows)
F_CB = AN / "EXPLORATORY_churn_by_refconf.csv"
cb.to_csv(F_CB, index=False)
BK = {lab: f"b{i + 1}" for i, lab in enumerate(BIN_LABELS)}
for _, r in cb.iterrows():
    k = CC[r.combo_id]
    loc = f"{r.combo_id}, bin {r.ref_conf_bin}"
    reg(f"cb.{k}.{BK[r.ref_conf_bin]}.n", str(r.n), rel(F_CB), loc + ": n")
    reg(f"cb.{k}.{BK[r.ref_conf_bin]}.x", str(r.changed), rel(F_CB), loc + ": changed")
    reg(f"cb.{k}.{BK[r.ref_conf_bin]}.pct", f"{100 * r.churn:.1f}", rel(F_CB), loc + ": churn x 100")
    reg(f"cb.{k}.{BK[r.ref_conf_bin]}.lo", f"{100 * r.ci_low_cp:.1f}", rel(F_CB), loc + ": ci_low_cp x 100")
    reg(f"cb.{k}.{BK[r.ref_conf_bin]}.hi", f"{100 * r.ci_high_cp:.1f}", rel(F_CB), loc + ": ci_high_cp x 100")
# share of 4-bit changes that fall below reference confidence 0.7 (exploratory summary)
for c in ["kev-0.8b/4bit", "kev-4b/4bit"]:
    s = cb[cb.combo_id == c]
    low = s[s.bin_high <= 0.7].changed.sum()
    reg(f"cb.{CC[c]}.below07", str(int(low)), rel(F_CB), f"{c}: sum changed over bins with bin_high <= 0.7")
    reg(f"cb.{CC[c]}.below07pct", f"{100 * low / s.changed.sum():.0f}", rel(F_CB),
        f"{c}: sum changed (bin_high<=0.7) / total changed x 100")

# ----------------------------------------------------------------------------------------------------------
# latency (partial protocol run, one configuration) -- reported in the appendix only
lat_worker = json.loads((RES / "bench" / "latency" / "kev-0.8b__fp32" / "worker.json").read_text())
lat_cpu = json.loads((RES / "bench" / "latency" / "kev-0.8b__fp32-cpu" / "worker.json").read_text())
F_LW = RES / "bench" / "latency" / "kev-0.8b__fp32" / "worker.json"
F_LC = RES / "bench" / "latency" / "kev-0.8b__fp32-cpu" / "worker.json"
mg = re.search(r"mps_driver ([\d.]+) GB > limit ([\d.]+) GB", lat_worker["error"])
reg("lat.stopgb", mg.group(1), rel(F_LW), "error: mps_driver X GB")
reg("lat.cells", str(lat_worker["cells_done"]), rel(F_LW), "cells_done")
reg("lat.cpuerr", lat_cpu["error"].split(":")[0], rel(F_LC), "error")
lsk = ls[ls.combo_id == "kev-0.8b/fp32"]
for _, r in lsk.iterrows():
    reg(f"lat.{r.cell}.{r.phase}.med", f"{r.median_ms:.1f}", rel(F_LS), f"{r.combo_id} {r.cell} {r.phase}: median_ms")
    reg(f"lat.{r.cell}.{r.phase}.n", str(int(r.n)), rel(F_LS), f"{r.combo_id} {r.cell} {r.phase}: n")
    reg(f"lat.{r.cell}.peak", f"{r.peak_phys_footprint_gb:.2f}", rel(F_LS), f"{r.combo_id} {r.cell}: peak_phys_footprint_gb")
reg("lat.maxctx", str(int(lsk.target_tokens.max())), rel(F_LS), "max target_tokens with status ok")
reg("lat.reps", str(int(lsk.n.iloc[0])) if lsk.n.nunique() == 1 else "VAR", rel(F_LS), "n per cell")
reg("lat.maxdrift", f"{lsk.drift_ratio.max():.2f}", rel(F_LS), "max drift_ratio")


# ==========================================================================================================
# FIGURES
def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    plt.close(fig)


def letter(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


# ---- Fig. 1: feasibility / peak physical footprint ------------------------------------------------------
def fig1():
    rows = list(reversed(FZ_ORDER))  # top-to-bottom in reading order
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    col = {"mlx": C_MLX, "torch-mps": C_MPS, "torch-cpu": C_CPU}
    labels = []
    for yi, c in zip(y, rows):
        r = fzi.loc[c]
        fam, var = c.split("/")
        labels.append(f"{FAMILY[fam]}: {VARIANT.get(var, var)}")
        if r.runs == "yes":
            ax.plot([0, r.peak_phys_footprint_gb], [yi, yi], color="#d9d9d9", lw=0.8, zorder=1)
            ax.plot(r.peak_phys_footprint_gb, yi, "o", ms=4.2, color=col[r.backend], zorder=3)
            if pd.notna(r.swap_delta_gb) and r.swap_delta_gb > 5:
                ax.annotate(f"swap +{r.swap_delta_gb:.1f} GB", (r.peak_phys_footprint_gb, yi), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=6, color="#555555")
        elif c == "kev-4b/adapter-bf16":
            ax.plot(r.peak_phys_footprint_gb, yi, "x", ms=5, mew=1.4, color=C_ALARM, zorder=4)
            ax.annotate("killed by memory guard", (r.peak_phys_footprint_gb, yi), xytext=(-6, 0),
                        textcoords="offset points", va="center", ha="right", fontsize=6, color=C_ALARM)
        elif c == "nanojev/bf16-upstream":
            ax.plot(r.peak_phys_footprint_gb, yi, "x", ms=5, mew=1.4, color="#222222", zorder=4)
            ax.annotate("upstream code refuses non-CUDA device", (r.peak_phys_footprint_gb, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=6, color="#222222")
        else:  # not attempted
            ax.annotate("not attempted: no Apple-Silicon 4-bit path in AnyJev 0.3.0", (0.15, yi), va="center",
                        fontsize=6, color="#555555", style="italic")
    # family separators
    fams = [c.split("/")[0] for c in rows]
    for i in range(1, len(rows)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color="#e6e6e6", lw=0.6, zorder=0)
    ax.axvline(GUARD_GB, color=C_ALARM, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.axvline(RAM_GB, color="#222222", lw=0.9, zorder=2)
    ax.text(GUARD_GB, len(rows) - 0.3, f"{GUARD_GB:g} GB guard", ha="right", va="bottom", fontsize=6,
            color=C_ALARM, rotation=0)
    ax.text(RAM_GB, len(rows) - 0.3, f"installed RAM\n({memsize / 2**30:.0f} GiB = {RAM_GB:.2f} GB)", ha="right",
            va="bottom", fontsize=6, color="#222222")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, len(rows) + 0.9)
    ax.set_xlim(0, RAM_GB + 0.6)
    ax.set_xlabel("Peak process physical footprint during smoke test (GB, 10$^9$ bytes)")
    ax.set_title(f"{n_run} of {n_total} model/precision/backend combinations ran on a 16 GB M2 Pro", loc="left")
    handles = [Line2D([], [], marker="o", ls="", color=C_MLX, ms=4.2, label="MLX"),
               Line2D([], [], marker="o", ls="", color=C_MPS, ms=4.2, label="PyTorch MPS"),
               Line2D([], [], marker="o", ls="", color=C_CPU, ms=4.2, label="PyTorch CPU"),
               Line2D([], [], marker="x", ls="", color="#222222", ms=5, mew=1.4, label="did not run")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.44, 0.985), ncol=1, handletextpad=0.3)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig1_feasibility_memory")


# ---- Fig. 2: reliability diagrams -------------------------------------------------------------------------
REL_BINS = np.linspace(0, 1, 11)
MIN_BIN_N = 5


def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.9), sharex="col", gridspec_kw={"height_ratios": [2.6, 1],
                                                                                  "hspace": 0.08, "wspace": 0.18})
    out = []
    for j, (fam, ref) in enumerate(REFS.items()):
        combos = [ref, f"{fam}/8bit", f"{fam}/4bit"]
        colors = [C_REF, C_8, C_4]
        names = ["fp32 reference" if fam == "kev-0.8b" else "bf16 reference", "8-bit", "4-bit"]
        ax, axh = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="#bdbdbd", lw=0.8, ls=(0, (3, 2)), zorder=0)
        for k, (c, colr, nm) in enumerate(zip(combos, colors, names)):
            d = rl[rl.combo_id == c]
            b = np.clip(np.digitize(d.p_max.values, REL_BINS[1:-1], right=False), 0, 9)
            xs, ys = [], []
            for i in range(10):
                sel = b == i
                n = int(sel.sum())
                if n:
                    out.append({"combo_id": c, "bin_low": REL_BINS[i], "bin_high": REL_BINS[i + 1], "n": n,
                                "mean_conf": d.p_max.values[sel].mean(), "accuracy": d.correct.values[sel].mean(),
                                "plotted": n >= MIN_BIN_N, "status": "EXPLORATORY figure support"})
                if n >= MIN_BIN_N:
                    xs.append(d.p_max.values[sel].mean())
                    ys.append(d.correct.values[sel].mean())
            ece = qm.loc[c, "ece15"]
            ax.plot(xs, ys, "-o", color=colr, ms=3, lw=1.3 if k == 2 else 1.0, zorder=3 + k,
                    label=f"{nm} (ECE$_{{15}}$ {ece:.4f})")
            axh.hist(d.p_max.values, bins=np.arange(0.2, 1.0001, 0.05), histtype="step", color=colr,
                     lw=1.2 if k == 2 else 0.9)
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.0, 1.02)
        letter(ax, "ab"[j], dx=-0.12 if j == 0 else -0.06)
        ax.set_title(f"{'Kev-0.8B' if fam == 'kev-0.8b' else 'Kev-4B'}: calibration after quantization")
        ax.legend(loc="upper left", handlelength=1.4)
        axh.set_xlabel("Confidence (top-1 probability)")
        if j == 0:
            ax.set_ylabel("Observed accuracy")
            axh.set_ylabel("Questions")
    for a in axes[0]:
        a.margins(0.04)
    pd.DataFrame(out).to_csv(AN / "EXPLORATORY_reliability_bins.csv", index=False)
    save(fig, "fig2_reliability")


# ---- Fig. 3: decision churn ----------------------------------------------------------------------------
def fig3():
    fig = plt.figure(figsize=(6.5, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    order = ["kev-0.8b/bf16", "kev-0.8b/full-bf16-merged", "kev-0.8b/8bit", "kev-0.8b/4bit", "kev-4b/8bit",
             "kev-4b/4bit"]
    lab = {"kev-0.8b/bf16": "0.8B bf16", "kev-0.8b/full-bf16-merged": "0.8B bf16 merged", "kev-0.8b/8bit": "0.8B 8-bit",
           "kev-0.8b/4bit": "0.8B 4-bit", "kev-4b/8bit": "4B 8-bit", "kev-4b/4bit": "4B 4-bit"}
    colr = {"kev-0.8b/bf16": C_BF, "kev-0.8b/full-bf16-merged": C_BF, "kev-0.8b/8bit": C_8, "kev-0.8b/4bit": C_4,
            "kev-4b/8bit": C_8, "kev-4b/4bit": C_4}
    ys = np.arange(len(order))[::-1]
    for yv, c in zip(ys, order):
        r = ch.loc[c]
        ax.plot([100 * r.ci_low, 100 * r.ci_high], [yv, yv], color=colr[c], lw=1.4)
        ax.plot(100 * r.churn, yv, "o", color=colr[c], ms=4)
        if c.endswith("4bit"):
            ax.annotate(f"{int(r.changed)}/{int(r.n)}", (100 * r.ci_high, yv), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6)
    ax.set_yticks(ys, [lab[c] for c in order])
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Decisions changed vs. reference (%)")
    ax.set_title("All questions (n = 656)")
    letter(ax, "a", dx=-0.42)
    ax.tick_params(axis="y", length=0)
    axes = [fig.add_subplot(gs[1]), None]
    axes[1] = fig.add_subplot(gs[2], sharey=axes[0])
    xs = np.arange(len(BIN_LABELS))
    pretty = ["<0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "≥0.9"]
    for a, fam in zip(axes, ["kev-0.8b", "kev-4b"]):
        for c, colr_, off in [(f"{fam}/8bit", C_8, -0.12), (f"{fam}/4bit", C_4, 0.12)]:
            s = cb[cb.combo_id == c].set_index("ref_conf_bin").loc[BIN_LABELS]
            yv = 100 * s.churn.values
            lo, hi = 100 * s.ci_low_cp.values, 100 * s.ci_high_cp.values
            a.errorbar(xs + off, yv, yerr=[yv - lo, hi - yv], fmt="o-", color=colr_, ms=3, lw=1.0, elinewidth=0.8,
                       capsize=0, label="8-bit" if "8bit" in c else "4-bit")
        a.set_xticks(xs, pretty, rotation=45, ha="right")
        a.set_title(f"{'Kev-0.8B' if fam == 'kev-0.8b' else 'Kev-4B'} by confidence bin")
        a.set_ylim(-1, None)
    axes[0].set_ylabel("Decisions changed (%)")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    letter(axes[0], "b", dx=-0.3); letter(axes[1], "c", dx=-0.12)
    bb0, bb1 = axes[0].get_position(), axes[1].get_position()
    fig.text((bb0.x0 + bb1.x1) / 2, -0.08, "Reference top-1 probability (bins)", ha="center", va="top", fontsize=8)
    axes[0].legend(loc="upper right")
    save(fig, "fig3_decision_churn")


# ---- Fig. 4: risk-coverage ----------------------------------------------------------------------------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    pts = []
    for a, (fam, ref) in zip(axes, REFS.items()):
        for c, colr_, nm in [(ref, C_REF, "fp32 reference" if fam == "kev-0.8b" else "bf16 reference"),
                             (f"{fam}/8bit", C_8, "8-bit"), (f"{fam}/4bit", C_4, "4-bit")]:
            d = rl[rl.combo_id == c].sort_values("p_max", ascending=False, kind="mergesort")
            err = 1 - d.correct.values
            k = np.arange(1, len(err) + 1)
            cov, risk = k / len(err), np.cumsum(err) / k
            a.plot(cov, risk, color=colr_, lw=1.3 if nm == "4-bit" else 1.0,
                   label=f"{nm} (AURC {qm.loc[c, 'aurc']:.4f})")
            for cv in (0.1, 0.25, 0.5, 0.75, 1.0):
                i = int(np.ceil(cv * len(err))) - 1
                pts.append({"combo_id": c, "coverage": cov[i], "risk": risk[i], "status": "EXPLORATORY figure support"})
        a.axhline(0.05, color="#999999", lw=0.7, ls=(0, (3, 2)))
        a.text(1.0, 0.05, "5% error budget", ha="right", va="bottom", fontsize=6, color="#666666")
        a.set_title(f"{'Kev-0.8B' if fam == 'kev-0.8b' else 'Kev-4B'}: selective-prediction risk")
        a.legend(loc="upper left")
        a.set_xlim(0, 1.0)
        a.margins(0.02)
        letter(a, "a" if fam == "kev-0.8b" else "b", dx=-0.12 if fam == "kev-0.8b" else -0.05)
    axes[0].set_ylabel("Error rate among answered")
    fig.text(0.5, -0.06, "Coverage (share of questions answered, most confident first)", ha="center", va="top",
             fontsize=8)
    pd.DataFrame(pts).to_csv(AN / "EXPLORATORY_risk_coverage_points.csv", index=False)
    save(fig, "fig4_risk_coverage")


# ==========================================================================================================
# TABLES
def tex_escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_tex(path, header, rows, colspec, caption=None, label=None, size="\\small", notes=None, midrules=()):
    lines = []
    lines.append(f"% generated by code/make_figures.py -- do not edit by hand")
    lines.append(size)
    lines.append("\\begin{adjustbox}{max width=\\linewidth}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    if notes:
        lines.append("\\par\\vspace{2pt}{\\footnotesize " + notes + "}")
    path.write_text("\n".join(lines) + "\n")


def t1():
    model_id = re.search(r"Model Identifier: (\S+)", env_text).group(1)
    build = re.search(r"BuildVersion:\s+(\S+)", env_text).group(1)
    hw = {
        "Machine": f"MacBook Pro ({model_id})",
        "Chip": re.search(r"Chip: (.+)", env_text).group(1).strip(),
        "CPU cores": f"{REG['env.cpucores']['value']} ({REG['env.pcores']['value']} performance + {REG['env.ecores']['value']} efficiency)",
        "GPU cores": REG["env.gpucores"]["value"],
        "Unified memory": f"{REG['env.ramgib']['value']} GiB ({REG['env.ramgb']['value']} GB)",
        "MLX recommended working set": f"{REG['env.mlxws']['value']} GB",
        "macOS": f"{REG['env.macos']['value']} (build {build})",
    }
    sw = {k: REG[f"env.{k}"]["value"] for k in ["python", "mlx", "mlx-lm", "torch", "transformers", "kev", "anyjev",
                                                 "laya", "jevmlx"]}
    repo_rows = re.findall(r"^\| (\S+) \| (https://github\.com/\S+?)(?:\.git)? \| `([0-9a-f]{40})` \| `([0-9a-f]{40})` \|$",
                           env_text, re.M)
    hf_rows = re.findall(r"^\| (\S+) \| (\S+/\S+) \| `([0-9a-f]{40})` \| ([^|]+) \|$", env_text, re.M)
    rows, csvrows = [], []
    for k, v in hw.items():
        rows.append(["Hardware", tex_escape(k), tex_escape(v)]); csvrows.append(["Hardware", k, v])
    for k, v in sw.items():
        rows.append(["Software", tex_escape(k), tex_escape(v)]); csvrows.append(["Software", k, v])
    used_hf = {"kev-0.8b", "kev-4b", "qwen3.5-0.8b-base", "qwen3.5-4b-base", "laya", "laya-multilingual", "nanojev",
               "laya-mlx", "laya-multilingual-mlx"}
    for name, url, pin, co in repo_rows:
        assert pin == co, f"checked-out SHA differs from pin for {name}"
        short = url.replace("https://github.com/", "")
        rows.append(["Code (GitHub)", tex_escape(short), f"\\texttt{{{pin[:12]}}}"]); csvrows.append(["Code", short, pin])
    for key, repo, sha, note in hf_rows:
        if key in used_hf and not repo.startswith("http"):
            rows.append(["Weights (HF)", tex_escape(repo), f"\\texttt{{{sha[:12]}}}"]); csvrows.append(["Weights", repo, sha])
    pd.DataFrame(csvrows, columns=["group", "item", "value"]).to_csv(TAB / "t1_environment.csv", index=False)
    write_tex(TAB / "t1_environment.tex", ["", "Item", "Value"], rows, "lll", size="\\footnotesize",
              notes="Commit hashes truncated to 12 characters here; full 40-character pins in "
                    "\\texttt{t1\\_environment.csv} and \\texttt{results/environment.md}. "
                    "Checked-out SHA equals the pinned SHA for every repository.")


def t2():
    rows, csvrows, mid = [], [], []
    prev = None
    footn = {"kev-4b/adapter-bf16": "a", "nanojev/bf16-upstream": "b", "anyjev-q0.8b/4bit": "c", "anyjev-q4b/4bit": "c"}
    for i, c in enumerate(FZ_ORDER):
        r = fzi.loc[c]
        fam, var = c.split("/")
        if prev is not None and fam != prev:
            mid.append(i)
        prev = fam
        def g(v, fmt):
            return "--" if pd.isna(v) else format(v, fmt)
        ran = "yes" if r.runs == "yes" else f"no$^{footn[c]}$"
        peak = g(r.peak_phys_footprint_gb, ".2f") if r.runs == "yes" or c == "kev-4b/adapter-bf16" else "--"
        if c == "kev-4b/adapter-bf16":
            peak = f"$>${peak}"
        rows.append([tex_escape(FAMILY[fam]), tex_escape(VARIANT.get(var, var)), BACKEND[r.backend], ran,
                     g(r.load_s, ".1f"), peak, g(r.latency_ms_single, ".1f"), g(r.latency_ms_warm, ".1f")])
        csvrows.append({"combo_id": c, "family": FAMILY[fam], "variant": var, "backend": r.backend, "runs": r.runs,
                        "load_s": r.load_s, "peak_phys_footprint_gb": r.peak_phys_footprint_gb,
                        "first_call_ms_single_shot_indicative": r.latency_ms_single,
                        "warm_call_ms_single_shot_indicative": r.latency_ms_warm, "swap_delta_gb": r.swap_delta_gb,
                        "error_or_killed": r.killed_reason if pd.notna(r.killed_reason) else r.error})
    pd.DataFrame(csvrows).to_csv(TAB / "t2_feasibility.csv", index=False)
    write_tex(TAB / "t2_feasibility.tex",
              ["Model", "Variant", "Backend", "Ran", "Load (s)", "Peak (GB)", "First call (ms)", "Warm call (ms)"],
              rows, "lllrrrrr", size="\\scriptsize", midrules=mid,
              notes=("Peak = peak process physical footprint (GB = $10^9$ bytes). First/warm call: ONE call each "
                     "in the smoke test (single-shot, indicative only; the first call includes graph compilation "
                     "and warm-up). "
                     f"$^a$Stopped by the memory guard ({REG['kill.gb']['value']} GB $>$ {REG['guard.gb']['value']} GB "
                     f"after {REG['kill.s']['value']} s). "
                     "$^b$Upstream inference entry point requires a CUDA device. "
                     "$^c$Not attempted: AnyJev 0.3.0 has no Apple-Silicon 4-bit path without a GGUF/llama.cpp "
                     "build (4-bit training-free readouts were run with jevmlx instead). "
                     "Source: \\texttt{results/feasibility\\_matrix\\_detail.csv}."))


def _diff_table(combos, path_stem, caption_note):
    metrics = ["acc", "nll", "brier", "ece15", "ece10", "mean_conf", "coverage_at_5pct_error", "aurc"]
    header = ["Metric"] + [tex_escape(NICE[c].replace(" (reference)", "")) for c in combos]
    rows, csvrows = [], []
    sub = pdif.set_index(["combo_id", "metric"])
    for m in metrics:
        r1, r2 = [METRIC_NICE[m]], [""]
        for c in combos:
            r = sub.loc[(c, m)]
            star = "$^{*}$" if r.p_holm < 0.05 else ""
            r1.append(f"{sgn4(r['diff'])}{star} (p$_H${peq(r.p_holm)})")
            r2.append(f"[{sgn4(r.ci_low)}, {sgn4(r.ci_high)}]")
            csvrows.append({"comparison": f"{c} - {r.reference}", "metric": m, "reference_value": r.ref_value,
                            "variant_value": r.value, "diff": r["diff"], "ci95_low": r.ci_low, "ci95_high": r.ci_high,
                            "p_boot": r.p_boot, "p_holm": r.p_holm})
        rows.append(r1); rows.append(r2)
    r1 = ["McNemar $b$ / $c$"]
    r2 = ["Order flips (of 36)"]
    for c in combos:
        mc = sub.loc[(c, "acc_mcnemar")]
        pf = sub.loc[(c, "perm_flip_rate")]
        r1.append(f"{int(mc.ref_value)} / {int(mc.value)} (p$_H${peq(mc.p_holm)})")
        nperm = int(qm.loc[c].perm_n)
        r2.append(f"{round(pf.ref_value * nperm)} $\\rightarrow$ {round(pf.value * nperm)} (p$_H${peq(pf.p_holm)})")
        for nm, rr in [("acc_mcnemar", mc), ("perm_flip_rate", pf)]:
            csvrows.append({"comparison": f"{c} - {rr.reference}", "metric": nm, "reference_value": rr.ref_value,
                            "variant_value": rr.value, "diff": rr["diff"], "ci95_low": np.nan, "ci95_high": np.nan,
                            "p_boot": rr.p_boot, "p_holm": rr.p_holm})
    rows.append(r1); rows.append(r2)
    pd.DataFrame(csvrows).to_csv(TAB / f"{path_stem}.csv", index=False)
    write_tex(TAB / f"{path_stem}.tex", header, rows, "l" + "r" * len(combos), size="\\scriptsize",
              midrules=[len(metrics) * 2],
              notes=("Each cell: variant minus reference on the same 656 questions; second line = paired record "
                     "bootstrap 95\\% percentile CI (10,000 resamples). p$_H$ = Holm-adjusted p within the "
                     "comparison's family of 10 tests; $^{*}$p$_H<0.05$. McNemar $b$ / $c$: questions correct only under the "
                     "reference / only under the variant (exact test). Order flips: permuted choice pairs whose top-1 "
                     "answer flips, reference $\\rightarrow$ variant (exact McNemar). Unadjusted p-values are in the CSV. "
                     + caption_note +
                     " Source: \\texttt{results/analysis/quant\\_paired\\_diffs.csv}."))


def t3():
    _diff_table(["kev-0.8b/8bit", "kev-0.8b/4bit", "kev-4b/8bit", "kev-4b/4bit"], "t3_h2_paired_diffs",
                "References: Kev-0.8B fp32 (PyTorch MPS) and Kev-4B merged bf16 (MLX).")
    _diff_table(["kev-0.8b/bf16", "kev-0.8b/full-bf16-merged"], "tA2_bf16_paired_diffs",
                "Reference: Kev-0.8B fp32 (PyTorch MPS).")
    # absolute values (appendix)
    cols = ["acc", "nll", "brier", "ece15", "ece10", "mean_conf", "coverage_at_5pct_error", "aurc"]
    rows = []
    for c in CC:
        r = qm.loc[c]
        rows.append([tex_escape(NICE[c])] + [f4(r[m]) for m in cols])
    qm.loc[list(CC), cols + ["n"]].to_csv(TAB / "tA1_point_metrics.csv")
    write_tex(TAB / "tA1_point_metrics.tex", ["Combination", "Acc.", "NLL", "Brier", "ECE$_{15}$", "ECE$_{10}$",
                                              "Mean conf.", "Cov.@5\\%", "AURC"],
              rows, "l" + "r" * len(cols), size="\\scriptsize", midrules=[5],
              notes=("Kev ``clean'' population of transfer-v4 development, n = 656 questions per row. "
                     "Accuracy, NLL and Brier equal Kev's own \\texttt{kev.metrics} implementation to machine precision. "
                     "Source: \\texttt{results/analysis/quant\\_metrics.csv}."))


def t4():
    rows, out = [], []
    for c in CC:
        r, q = rc.loc[c], qm.loc[c]
        rows.append([tex_escape(NICE[c]), f"{r.extra_T_mean:.3f} [{r.extra_T_min:.3f}, {r.extra_T_max:.3f}]",
                     f4(q.nll), f4(r.oof_nll), f4(q.brier), f4(r.oof_brier), f4(q.ece15), f4(r.oof_ece15),
                     f4(q.acc), f4(r.oof_acc)])
        out.append({"combo_id": c, "extra_T_mean": r.extra_T_mean, "extra_T_min": r.extra_T_min,
                    "extra_T_max": r.extra_T_max, "nll_shipped_T": q.nll, "oof_nll": r.oof_nll,
                    "brier_shipped_T": q.brier, "oof_brier": r.oof_brier, "ece15_shipped_T": q.ece15,
                    "oof_ece15": r.oof_ece15, "acc_shipped_T": q.acc, "oof_acc": r.oof_acc,
                    "status": "EXPLORATORY (deviation log #13)"})
    pd.DataFrame(out).to_csv(TAB / "t4_recalibration_oof.csv", index=False)
    write_tex(TAB / "t4_recalibration_oof.tex",
              ["Combination", "Extra $T$ mean [min, max]", "NLL", "NLL$_{\\mathrm{OOF}}$", "Brier",
               "Brier$_{\\mathrm{OOF}}$", "ECE$_{15}$", "ECE$_{15,\\mathrm{OOF}}$", "Acc.", "Acc.$_{\\mathrm{OOF}}$"],
              rows, "lrrrrrrrrr", size="\\scriptsize", midrules=[5],
              notes=("Exploratory. An extra temperature on top of Kev's shipped temperature was fitted by NLL on four "
                     "folds and applied to the fifth (5-fold out-of-fold, seed 0); columns without OOF use the shipped "
                     "temperature. Source: \\texttt{results/analysis/quant\\_recalibration.csv}, "
                     "\\texttt{quant\\_metrics.csv}."))


def t5():
    rows, out = [], []
    for _, r in lsk.iterrows():
        rows.append([tex_escape(r.cell), r.phase, "--" if pd.isna(r.target_tokens) else str(int(r.target_tokens)),
                     str(int(r.n_questions)), str(int(r.n)), f"{r.median_ms:.1f}", f"{r.q1_ms:.1f}--{r.q3_ms:.1f}",
                     f"{r.p95_ms:.1f}", f"{r.peak_phys_footprint_gb:.2f}"])
        out.append(r.to_dict())
    pd.DataFrame(out).to_csv(TAB / "tA3_latency_partial.csv", index=False)
    write_tex(TAB / "tA3_latency_partial.tex",
              ["Cell", "Phase", "Context (tok.)", "Questions", "n", "Median (ms)", "IQR (ms)", "p95 (ms)", "Peak (GB)"],
              rows, "llrrrrrrr", size="\\scriptsize",
              notes=("One configuration only: Kev-0.8B fp32 on PyTorch MPS, 3 warm-up + 10 timed calls per cell. "
                     f"The run stopped at the next context cell (8,192 tokens) when the in-process guard saw MPS driver "
                     f"memory of {REG['lat.stopgb']['value']} GB; the CPU run of the same model was interrupted "
                     "(\\texttt{KeyboardInterrupt}). No other configuration was timed. Source: "
                     "\\texttt{results/latency\\_summary.csv}, \\texttt{bench/latency/*/worker.json}."))


# ==========================================================================================================
def write_registry():
    lines = ["% generated by code/make_figures.py -- do not edit by hand",
             "\\makeatletter",
             "\\newcommand{\\V}[1]{\\ifcsname jv@#1\\endcsname\\csname jv@#1\\endcsname\\else"
             "\\textbf{??#1??}\\fi}"]
    for k, v in REG.items():
        lines.append(f"\\expandafter\\def\\csname jv@{k}\\endcsname{{{v['value']}}}")
    lines.append("\\makeatother")
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(REG.values()).to_csv(TAB / "numbers.csv", index=False)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    t1(); t2(); t3(); t4(); t5()
    write_registry()
    print(f"figures -> {FIG}\ntables  -> {TAB}\nregistry: {len(REG)} values")
