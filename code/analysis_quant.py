"""H2 analysis: does post-training quantization change accuracy / calibration of Kev decision models?

    python analysis_quant.py --kev-src <path to kev repo @ 2074cc44> [--results <results>] [--boot 10000]

Inputs : results/bench/accuracy/<combo>/rows.json (kev.benchmark rows; written by bench_accuracy.py)
Population: Kev's "clean" population = variant == "clean" and source != "unknowable" (656 questions on transfer-v4 dev),
            identical to kev.benchmark.summarize()["clean"]. Permutation flips use the 36 "permuted" rows.
Outputs: results/analysis/quant_metrics.csv        point estimates per combo (kev.metrics.metrics + ECE-15)
         results/analysis/quant_paired_diffs.csv   variant - reference, paired record bootstrap 95% CI, Holm p
         results/analysis/quant_recalibration.csv  5-fold out-of-fold temperature refit per combo
         results/analysis/quant_rows_long.csv      per-question p_max / correct / p_label for figures
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFS = {"kev-0.8b": "kev-0.8b/fp32", "kev-4b": "kev-4b/full-bf16-merged"}
COMBOS = ["kev-0.8b/fp32", "kev-0.8b/bf16", "kev-0.8b/full-bf16-merged", "kev-0.8b/8bit", "kev-0.8b/4bit",
          "kev-4b/full-bf16-merged", "kev-4b/8bit", "kev-4b/4bit"]
EPS = None  # set from kev.metrics.EPSILON


def ece(conf, correct, bins):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def load(results: Path, combo: str):
    rows = json.loads((results / "bench" / "accuracy" / combo.replace("/", "__") / "rows.json").read_text())
    return rows


def population(rows):
    clean = [r for r in rows if r["variant"] == "clean" and r["source"] != "unknowable"]
    return {(r["id"], r["question"]): r for r in clean}


def arrays(pop, keys, T=1.0):
    P = []
    for k in keys:
        p = np.asarray(pop[k]["p"], dtype=float)
        if T != 1.0:
            lp = np.log(np.clip(p, EPS, 1.0)) / T
            p = np.exp(lp - lp.max()); p /= p.sum()
        P.append(p)
    y = np.array([pop[k]["label"] for k in keys])
    conf = np.array([p.max() for p in P]); pred = np.array([int(p.argmax()) for p in P])
    correct = pred == y
    p_y = np.array([p[t] for p, t in zip(P, y)])
    nll = -np.log(np.clip(p_y, EPS, 1.0))
    brier = np.array([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(P, y)])
    return conf, correct, nll, brier, p_y


def scalar_metrics(conf, correct, nll, brier, cov_fn, aurc_fn):
    return {"acc": float(correct.mean()), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "ece10": ece(conf, correct, 10), "ece15": ece(conf, correct, 15), "mean_conf": float(conf.mean()),
            "overconfidence": float(conf.mean() - correct.mean()),
            "coverage_at_5pct_error": cov_fn(conf, correct, 0.05), "aurc": aurc_fn(conf, correct),
            "confident_error_rate": float(np.mean((conf >= 0.9) & ~correct))}


def perm_flips(rows):
    clean = {(r["id"], r["question"]): r for r in rows if r["variant"] == "clean"}
    out = {}
    for r in rows:
        if r["variant"] == "permuted" and r["type"] == "choice":
            o = clean[(r["parent"], r["question"])]
            aligned = [r["p"][r["keys"].index(k)] for k in o["keys"]]
            out[(r["id"], r["question"])] = int(np.argmax(aligned) != np.argmax(o["p"]))
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def mcnemar_exact(a_correct, b_correct):
    from scipy.stats import binomtest
    b = int(np.sum(a_correct & ~b_correct)); c = int(np.sum(~a_correct & b_correct))
    return (b, c, float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)


def fit_temperature(p_list, y):
    from scipy.optimize import minimize_scalar
    L = [np.log(np.clip(p, EPS, 1.0)) for p in p_list]

    def nll(logT):
        T = math.exp(logT); s = 0.0
        for lp, t in zip(L, y):
            z = lp / T; z = z - z.max(); s -= z[t] - math.log(np.exp(z).sum())
        return s / len(L)
    return math.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)


def main():
    global EPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--kev-src", required=True)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.kev_src)
    from kev.metrics import EPSILON, area_under_risk_coverage, coverage_at_error, metrics as kev_metrics
    EPS = EPSILON
    results = Path(a.results); out = results / "analysis"; out.mkdir(parents=True, exist_ok=True)
    rows = {c: load(results, c) for c in COMBOS}
    pops = {c: population(r) for c, r in rows.items()}
    keys = sorted(set.intersection(*[set(p) for p in pops.values()]))
    assert all(len(p) == len(keys) for p in pops.values()), "populations differ across combos"
    rng = np.random.default_rng(a.seed)
    n = len(keys)
    # record-level resampling (one question per record here, so records == questions)
    idx = rng.integers(0, n, size=(a.boot, n))

    point, long_rows, arr = [], [], {}
    for c in COMBOS:
        conf, correct, nll, brier, p_y = arrays(pops[c], keys)
        arr[c] = (conf, correct, nll, brier)
        m = scalar_metrics(conf, correct, nll, brier, coverage_at_error, area_under_risk_coverage)
        km = kev_metrics([pops[c][k] for k in keys])               # cross-check against kev's own implementation
        m.update(n=n, kev_acc=km["acc"], kev_brier=km["brier"], kev_ece10=km["ece"], kev_nll=km["nll"])
        fl = perm_flips(rows[c]); m.update(perm_n=len(fl), perm_flips=int(sum(fl.values())))
        point.append({"combo_id": c, **m})
        for k, cf, ok, py in zip(keys, conf, correct, p_y):
            long_rows.append({"combo_id": c, "id": k[0], "question": k[1], "source": pops[c][k]["source"],
                              "type": pops[c][k]["type"], "p_max": cf, "correct": int(ok), "p_label": py})
    pd.DataFrame(point).to_csv(out / "quant_metrics.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out / "quant_rows_long.csv", index=False)

    diffs = []
    metric_names = ["acc", "nll", "brier", "ece10", "ece15", "mean_conf", "coverage_at_5pct_error", "aurc"]
    for c in COMBOS:
        fam = c.split("/")[0]; ref = REFS[fam]
        if c == ref:
            continue
        A, Bv = arr[ref], arr[c]
        boots = {m: np.empty(a.boot) for m in metric_names}
        for b in range(a.boot):
            ii = idx[b]
            ma = scalar_metrics(A[0][ii], A[1][ii], A[2][ii], A[3][ii], coverage_at_error, area_under_risk_coverage)
            mb = scalar_metrics(Bv[0][ii], Bv[1][ii], Bv[2][ii], Bv[3][ii], coverage_at_error, area_under_risk_coverage)
            for m in metric_names:
                boots[m][b] = mb[m] - ma[m]
        pa = scalar_metrics(*A, coverage_at_error, area_under_risk_coverage)
        pb = scalar_metrics(*Bv, coverage_at_error, area_under_risk_coverage)
        for m in metric_names:
            d = boots[m]
            p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))     # percentile-bootstrap sign test
            diffs.append({"combo_id": c, "reference": ref, "metric": m, "ref_value": pa[m], "value": pb[m],
                          "diff": pb[m] - pa[m], "ci_low": float(np.quantile(d, .025)), "ci_high": float(np.quantile(d, .975)),
                          "p_boot": p_two})
        b_, c_, p_mc = mcnemar_exact(A[1], Bv[1])
        diffs.append({"combo_id": c, "reference": ref, "metric": "acc_mcnemar", "ref_value": b_, "value": c_,
                      "diff": (c_ - b_) / n, "p_boot": p_mc})
        fa, fb = perm_flips(rows[ref]), perm_flips(rows[c])
        common = sorted(set(fa) & set(fb))
        x = np.array([fa[k] for k in common]); z = np.array([fb[k] for k in common])
        b2, c2, p_fl = mcnemar_exact(x == 0, z == 0)   # discordant pairs: flipped in one and not the other
        diffs.append({"combo_id": c, "reference": ref, "metric": "perm_flip_rate", "ref_value": float(x.mean()),
                      "value": float(z.mean()), "diff": float(z.mean() - x.mean()), "p_boot": p_fl,
                      "note": f"exact McNemar on {len(common)} permuted pairs; discordant {b2}/{c2}"})
    D = pd.DataFrame(diffs)
    D["p_holm"] = np.nan
    for c in D.combo_id.unique():                       # Holm within each variant-vs-reference comparison family
        m = D.combo_id == c
        D.loc[m, "p_holm"] = holm(D.loc[m, "p_boot"].values)
    D.to_csv(out / "quant_paired_diffs.csv", index=False)

    rec = []
    fold = rng.permutation(n) % a.folds
    for c in COMBOS:
        P = [np.asarray(pops[c][k]["p"], dtype=float) for k in keys]; y = np.array([pops[c][k]["label"] for k in keys])
        Ts, oof = [], [None] * n
        for f in range(a.folds):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            T = fit_temperature([P[i] for i in tr], y[tr]); Ts.append(T)
            for i in te:
                lp = np.log(np.clip(P[i], EPS, 1.0)) / T; q = np.exp(lp - lp.max()); oof[i] = q / q.sum()
        conf = np.array([q.max() for q in oof]); correct = np.array([q.argmax() for q in oof]) == y
        p_y = np.array([q[t] for q, t in zip(oof, y)])
        nll = -np.log(np.clip(p_y, EPS, 1.0)); brier = np.array([((q - np.eye(len(q))[t]) ** 2).sum() for q, t in zip(oof, y)])
        m = scalar_metrics(conf, correct, nll, brier, coverage_at_error, area_under_risk_coverage)
        rec.append({"combo_id": c, "extra_T_mean": float(np.mean(Ts)), "extra_T_min": float(np.min(Ts)),
                    "extra_T_max": float(np.max(Ts)), **{f"oof_{k}": v for k, v in m.items()}})
    pd.DataFrame(rec).to_csv(out / "quant_recalibration.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
