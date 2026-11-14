"""Accuracy / calibration benchmark (protocol sections 5-7): every runnable model x precision on Kev's frozen
transfer-v4 DEVELOPMENT partition, scored with Kev's own row builder and metrics (kev.benchmark.prediction_rows /
summarize), so numbers are directly comparable with Kev's published reports.

    python bench_accuracy.py --dry-run                        # plan only
    python bench_accuracy.py                                  # all default combos, sequential
    python bench_accuracy.py --only kev-0.8b/bf16 --limit 20  # quick check on 20 records
    python bench_accuracy.py --only anyjev-q4b/bf16           # resumes where an interrupted run stopped

Design
  * Two processes per combo. The WORKER (guarded subprocess, one model, nothing else imported) answers every request
    in-process through the same adapters.py runner the smoke test validated, and appends one JSON line per record
    (raw response, probabilities in kev's key space, model latency) to answers.jsonl. Re-running skips records already
    answered, so a long run (AnyJev-4B) can be resumed. The PARENT builds kev rows, scores, and writes the reports.
  * A record whose answer is missing or invalid (error, NaN, wrong keys) is NOT silently dropped: it is listed in
    failures.json and counted in coverage; metrics are over the answered records (report states n). Cross-model
    comparisons in the analysis use the intersection of answered records.
  * The locked test partition is never read. CPU-only variants are excluded by default (same numerics as their MPS
    rows; they are latency references only) -- override with --only.
Outputs: results/bench/accuracy/<combo>/{answers.jsonl, rows.json, report.json, failures.json, worker.json}
         results/metrics_accuracy.csv (one row per combo), logs/bench_accuracy_<ts>.log
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pins  # noqa: E402
from combos import smoke_combos  # noqa: E402
from guard import GB, InProcessSampler, run_guarded, system_snapshot, write_json_atomic  # noqa: E402

os.environ.setdefault("HF_HOME", str(pins.cache_dir() / "hf"))
SUITE_REL = Path("evals") / "v4" / "transfer-v4"
EXCLUDE_DEFAULT = {"torch-cpu", "torch-cuda-only", "none"}
SUMMARY_COLS = ["combo_id", "model", "variant", "backend", "n_requested", "n_answered", "n_failed", "acc", "nll",
                "brier", "ece10_kev", "mean_conf", "acc_calibrated", "perm_n", "perm_flip_rate", "perm_mean_max_delta",
                "coverage_at_5pct_error", "aurc", "confident_error_rate", "latency_ms_median", "latency_ms_p95", "peak_mem_gb",
                "wall_s", "killed_reason", "notes"]


def iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slug(cid: str) -> str:
    return cid.replace("/", "__").replace(" ", "_")


def _sync(mx):
    if mx is not None:
        mx.synchronize()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass


def to_kev_probabilities(body: dict, req: dict) -> dict:
    """Same mapping as kev.predictors.RemotePredictor: noul -> {true, false}; choice/score -> probabilities as returned."""
    probs = {}
    for qid, q in req["questions"].items():
        a = body["answers"][qid]
        if q["type"] == "noul":
            p = float(a["noul"])
            probs[qid] = {"true": p, "false": 1 - p}
        else:
            probs[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return probs


# =============================================================================================== worker
def worker(combo: dict, requests_path: Path, out_dir: Path, limit_gb: float):
    answers_path = out_dir / "answers.jsonl"
    done = set()
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["id"])
            except json.JSONDecodeError:
                pass
    reqs = [json.loads(l) for l in requests_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [r for r in reqs if r["id"] not in done]
    res = {"combo_id": combo["id"], "started": iso(), "n_total": len(reqs), "n_already_done": len(done), "ok": False}

    def on_breach(reason):
        res.update(error=reason, guard="in-process", ok=False)
        write_json_atomic(out_dir / "worker.json", res)

    sampler = InProcessSampler(limit_gb=limit_gb, sample_s=0.1, on_breach=on_breach).start()
    mx = None
    try:
        if combo["backend"] == "mlx":
            import mlx.core as mx  # noqa: F811
            mx.reset_peak_memory()
        from adapters import build_runner
        t = time.perf_counter()
        runner = build_runner(combo["kind"], combo["params"])
        _sync(mx)
        res["load_s"] = round(time.perf_counter() - t, 3)
        res["info"] = runner.info()
        with open(answers_path, "a", encoding="utf-8") as out:
            for i, item in enumerate(todo):
                req = item["request"]
                line = {"id": item["id"]}
                try:
                    t = time.perf_counter()
                    body = runner.answer(req)
                    _sync(mx)
                    line["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
                    line["model_latency_ms"] = body.get("latency_ms")
                    line["probabilities"] = to_kev_probabilities(body, req)
                    bad = [qid for qid, d in line["probabilities"].items()
                           if not all(isinstance(v, float) and math.isfinite(v) for v in d.values())]
                    line["ok"] = not bad
                    if bad:   # keep the evidence, but as JSON-safe text (NaN would make json.dumps(allow_nan=False) raise)
                        line["error"] = f"non-finite probabilities for {bad}"
                        line["probabilities_raw"] = json.dumps(line.pop("probabilities"), default=str)
                        line["response_raw"] = json.dumps(body, default=str)[:4000]
                except Exception as e:  # noqa: BLE001 - every failure is recorded, the run continues
                    line.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
                out.write(json.dumps(line, allow_nan=False, default=str) + "\n")
                out.flush()
                if (i + 1) % 50 == 0:
                    print(f"[worker] {combo['id']}: {i + 1}/{len(todo)} answered this session", flush=True)
        runner.close()
        res["ok"] = True
    except BaseException as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        res.update(sampler.stop())
        if mx is not None:
            try:
                res["mlx_peak_gb"] = round(mx.get_peak_memory() / GB, 3)
            except Exception as e:  # noqa: BLE001
                res["mlx_peak_gb_error"] = str(e)
        res["finished"] = iso()
        write_json_atomic(out_dir / "worker.json", res)


# =============================================================================================== parent
def build_requests(suite: Path, path: Path, limit: int | None):
    from kev.data import api_request
    from kev.suite import load_split
    records = load_split(str(suite), "development", allow_test=False)
    if limit:
        records = records[:limit]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"id": r["_meta"]["id"], "request": {**api_request(r), "model": "kev-latest"}}) + "\n")
    return records


def score(combo, records, suite: Path, out_dir: Path, guard_res: dict | None):
    import numpy as np
    from kev.benchmark import prediction_rows, summarize
    from kev.suite import digest, read_manifest
    manifest = read_manifest(str(suite))
    answers = {}
    p = out_dir / "answers.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok") or rec["id"] not in answers:
                answers[rec["id"]] = rec            # a later successful retry replaces an earlier failure
    rows, failures, lat, mlat = [], [], [], []
    for r in records:
        rid = r["_meta"]["id"]
        a = answers.get(rid)
        if a is None:
            failures.append({"id": rid, "error": "not answered (worker stopped before this record)"}); continue
        if not a.get("ok"):
            failures.append({"id": rid, "error": a.get("error")}); continue
        try:
            rows.extend(prediction_rows(r, {"probabilities": a["probabilities"]}))
            lat.append(a["latency_ms"])
            if a.get("model_latency_ms") is not None:
                mlat.append(float(a["model_latency_ms"]))
        except Exception as e:  # noqa: BLE001 - kev's validator rejected the distribution
            failures.append({"id": rid, "error": f"kev validation: {type(e).__name__}: {e}"})
    write_json_atomic(out_dir / "rows.json", rows)
    write_json_atomic(out_dir / "failures.json", failures)
    report = summarize(rows, heldout_sources=tuple(manifest["holdout_sources"])) if rows else {}
    w = json.loads((out_dir / "worker.json").read_text()) if (out_dir / "worker.json").exists() else {}
    report.update(
        combo=combo, split="development", suite=str(suite), suite_manifest_sha256=digest(suite / "manifest.json"),
        coverage={"requested_records": len(records), "answered_records": len({x["id"] for x in rows}),
                  "failed_records": len(failures)},
        latency_ms={"median": float(np.median(lat)) if lat else None, "p95": float(np.quantile(lat, .95)) if lat else None,
                    "definition": "wall time of runner.answer() incl. device sync, in-process (no HTTP)"},
        worker={k: w.get(k) for k in ("load_s", "info", "error", "mlx_peak_gb", "ok", "n_already_done")},
        worker_memory={k: v for k, v in w.items() if "gb" in k or "footprint" in k or "rss" in k},
        guard=guard_res, ece_note="kev.metrics.ece uses 10 equal-width bins; protocol 15-bin ECE is computed in analysis from rows.json")
    write_json_atomic(out_dir / "report.json", report)
    return report, w


def summary_row(combo, report, w, guard_res):
    clean = report.get("clean") or {}
    cal = report.get("calibrated_clean") or {}
    perm = report.get("permutation") or {}
    cov5 = clean.get("coverage_at_5pct_error")
    peaks = [v for k, v in (w or {}).items() if isinstance(v, (int, float)) and k.endswith("_gb") and "peak" in k]
    return {"combo_id": combo["id"], "model": combo["model"], "variant": combo["variant"], "backend": combo["backend"],
            "n_requested": report.get("coverage", {}).get("requested_records"),
            "n_answered": report.get("coverage", {}).get("answered_records"),
            "n_failed": report.get("coverage", {}).get("failed_records"),
            "acc": clean.get("acc"), "nll": clean.get("nll"), "brier": clean.get("brier"), "ece10_kev": clean.get("ece"),
            "mean_conf": clean.get("mean_conf"), "acc_calibrated": cal.get("acc"), "perm_n": perm.get("n"),
            "perm_flip_rate": perm.get("flip_rate"), "perm_mean_max_delta": perm.get("mean_max_delta"),
            "coverage_at_5pct_error": cov5, "aurc": clean.get("aurc"),
            "confident_error_rate": clean.get("confident_error_rate"),
            "latency_ms_median": report.get("latency_ms", {}).get("median"),
            "latency_ms_p95": report.get("latency_ms", {}).get("p95"),
            "peak_mem_gb": max(peaks) if peaks else None, "wall_s": (guard_res or {}).get("wall_s"),
            "killed_reason": (guard_res or {}).get("killed_reason"), "notes": combo.get("notes", "")}


def write_summary(results: Path, rows: dict):
    path = results / "metrics_accuracy.csv"
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = {r["combo_id"]: r for r in csv.DictReader(f)}
    existing.update(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        wr.writeheader()
        for r in existing.values():
            wr.writerow({k: r.get(k) for k in SUMMARY_COLS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated combo ids (default: every runnable non-CPU combo)")
    ap.add_argument("--skip", default="")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N development records (smoke check)")
    ap.add_argument("--tf-base", choices=("base", "instruct"), default="base")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rss-limit-gb", type=float, default=14.0)
    ap.add_argument("--min-avail-gb", type=float, default=0.75)
    ap.add_argument("--swap-growth-gb", type=float, default=12.0,
                    help="larger than the smoke default: AnyJev-4B is known to run with ~10 GB swap growth (deviation log #8)")
    ap.add_argument("--timeout-s", type=float, default=6 * 3600)
    ap.add_argument("--cooldown-s", type=float, default=20)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--requests", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)
    a = ap.parse_args()

    combos = {c["id"]: c for c in smoke_combos(a.tf_base)}
    if a.worker:
        worker(combos[a.worker], Path(a.requests), Path(a.out_dir), a.rss_limit_gb)
        return

    results = pins.results_dir()
    suite = pins.repo_dir("kev") / SUITE_REL
    smoke_csv = results / "feasibility_matrix.csv"
    runnable = set()
    if smoke_csv.exists():
        with open(smoke_csv, encoding="utf-8") as f:
            runnable = {f"{r['model']}|{r['variant']}" for r in csv.DictReader(f) if r["runs"] == "yes"}
    if a.only:
        ids = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = [x for x in ids if x not in combos]
        if unknown:
            raise SystemExit(f"unknown combo(s): {unknown}; known: {sorted(combos)}")
    else:
        ids = [cid for cid, c in combos.items() if not c.get("unsupported") and c["backend"] not in EXCLUDE_DEFAULT
               and f"{c['model']}|{c['variant']}" in runnable]
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    ids = [i for i in ids if i not in skip]
    bench = results / "bench" / "accuracy"
    print(f"# accuracy benchmark {iso()}  suite={suite}  limit={a.limit}  combos={len(ids)}")
    for cid in ids:
        print(f"  {cid:32s} {combos[cid]['backend']}")
    if a.dry_run:
        print(f"suite exists: {suite.exists()}  smoke CSV runnable rows: {len(runnable)}")
        return
    if not suite.exists():
        raise SystemExit(f"suite not found: {suite} (run setup_mac.sh first)")

    import psutil
    bench.mkdir(parents=True, exist_ok=True)
    log_path = results / "logs" / f"bench_accuracy_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"limit{a.limit}" if a.limit else "full"
    req_path = bench / f"requests_development_{tag}.jsonl"
    records = build_requests(suite, req_path, a.limit)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    summary = {}
    for cid in ids:
        c = combos[cid]
        out_dir = bench / (slug(cid) + ("" if not a.limit else f"__{tag}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        time.sleep(a.cooldown_s)
        t_wait = time.time()
        while psutil.virtual_memory().available < 4 * GB and time.time() - t_wait < 120:
            time.sleep(2)
        before = system_snapshot(f"before {cid}")
        print(f"[run ] {cid}  available {psutil.virtual_memory().available / GB:.2f} GB  therm={before['pmset_therm'][:80]!r}", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", cid, "--requests", str(req_path),
               "--out-dir", str(out_dir), "--tf-base", a.tf_base, "--rss-limit-gb", str(a.rss_limit_gb)]
        g = run_guarded(cmd, log_path=log_path, env=env, cwd=str(HERE), rss_limit_gb=a.rss_limit_gb,
                        min_avail_gb=a.min_avail_gb, swap_growth_gb=a.swap_growth_gb, timeout_s=a.timeout_s)
        after = system_snapshot(f"after {cid}")
        write_json_atomic(out_dir / "system.json", {"before": before, "after": after, "guard": g})
        report, w = score(c, records, suite, out_dir, g)
        summary[cid] = summary_row(c, report, w, g)
        s = summary[cid]
        print(f"[done] {cid}: answered {s['n_answered']}/{s['n_requested']} acc={s['acc']} brier={s['brier']} "
              f"ece10={s['ece10_kev']} median_ms={s['latency_ms_median']} killed={s['killed_reason']}", flush=True)
        if not a.limit:
            write_summary(results, summary)
    if a.limit:
        print("(--limit run: metrics_accuracy.csv not updated)")


if __name__ == "__main__":
    main()
