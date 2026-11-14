"""Smoke test of every protocol section 4 model x precision combination on this Mac (RQ1 feasibility).

For each combination, sequentially, in its OWN subprocess (OOM isolation; never two GPU jobs at once):
  load the model -> answer the fixed 3-question request (code/smoke_request.json) once on a fresh state (cold)
  and once more (warm) -> validate the answer shape. Measured: import time, load time, single-request latency
  (wall clock around the call, device synchronized), and memory: the macOS process PHYSICAL FOOTPRINT
  (proc_pid_rusage ri_phys_footprint; includes Metal/GPU memory that RSS misses on Apple Silicon) sampled every
  100 ms by the parent (whole process tree) plus the worker's own ri_lifetime_max_phys_footprint, process RSS (psutil,
  100 ms, reported separately), MLX `mx.get_peak_memory()` and torch `torch.mps.driver_allocated_memory()` (sampled
  peak). The memory guard acts on the physical footprint (RSS only if the footprint cannot be read).
  pmset/memory state is logged before and after every combination.

Outputs (results):
  feasibility_matrix.csv        model,variant,backend,runs,load_s,peak_mem_gb,peak_phys_footprint_gb,peak_rss_gb,
                                latency_ms_single,error,notes,repo_sha,hf_revision
  feasibility_matrix_detail.csv  the same plus every measurement column
  raw/smoke/<combo>.json        everything the worker and the parent saw, including the full response and traceback
  logs/smoke_<timestamp>.log    full log (parent + every worker's stdout/stderr)
A failure is a runs=no row with the exact error text. Load time and latency are never filled in for something that
did not run; the only exception is peak_mem_gb (and the footprint/RSS peak columns), which records the peak memory
actually observed up to the failure (e.g. the guard-killed Kev-4B adapter load) and is flagged in the notes column.
peak_mem_gb = max(peak_phys_footprint_gb, MLX peak, MPS driver peak); peak_phys_footprint_gb = max(parent-sampled tree
footprint, worker-sampled footprint, worker lifetime max); peak_rss_gb = max(parent tree RSS, worker RSS). GB = 1e9 bytes.
latency_ms_single = first request after load (cold state), wall clock in the worker, ms.

    python smoke_test.py --dry-run             # plan, paths, prerequisites; imports no ML framework
    python smoke_test.py                       # everything (run_smoke.sh calls this after prepare_models.py)
    python smoke_test.py --only kev-0.8b/bf16,laya/fp32
    python smoke_test.py --selftest --results-dir <scratch>   # harness test with fake workers, no model
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pins  # noqa: E402
from combos import selftest_combos, smoke_combos  # noqa: E402
from guard import GB, InProcessSampler, describe_exit, run_guarded, system_snapshot, write_json_atomic  # noqa: E402

os.environ.setdefault("HF_HOME", str(pins.cache_dir() / "hf"))

CSV_COLS = ["model", "variant", "backend", "runs", "load_s", "peak_mem_gb", "peak_phys_footprint_gb", "peak_rss_gb",
            "latency_ms_single", "error", "notes", "repo_sha", "hf_revision"]
DETAIL_COLS = ["combo_id"] + CSV_COLS + ["import_s", "latency_ms_warm", "model_latency_ms_cold", "model_latency_ms_warm",
                                         "phys_footprint_peak_gb_parent", "phys_footprint_peak_gb_worker",
                                         "phys_footprint_lifetime_max_gb", "rss_peak_gb_parent", "rss_peak_gb_worker",
                                         "mlx_peak_gb", "mps_driver_peak_gb", "guard_metric", "min_sys_avail_gb",
                                         "swap_delta_gb", "wall_s", "exit", "killed_reason", "raw_json"]


def iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slug(combo_id: str) -> str:
    return combo_id.replace("/", "__").replace(" ", "_")


def done_path(task_id: str) -> Path:
    return pins.models_dir() / ".done" / f"{task_id}.json"


def git_sha(path: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


# =============================================================================================== validation
def validate(resp: dict, req: dict) -> dict:
    """Shape check of a /v1/systemone body against the request. Values are not judged (that is the benchmark)."""
    problems = []
    answers = resp.get("answers") if isinstance(resp, dict) else None
    if not isinstance(answers, dict):
        return {"valid": False, "problems": ["no 'answers' object"]}
    for qid, q in req["questions"].items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            problems.append(f"{qid}: missing")
            continue
        if q["type"] == "noul":
            v = a.get("noul")
            if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                problems.append(f"{qid}: noul={v!r} not a probability")
            continue
        probs = a.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            problems.append(f"{qid}: no probabilities")
            continue
        s = sum(float(v) for v in probs.values())
        if not 0.98 <= s <= 1.02:
            problems.append(f"{qid}: probabilities sum to {s:.4f}")
        if q["type"] == "choice":
            if a.get("choice") not in q["criteria"]:
                problems.append(f"{qid}: choice={a.get('choice')!r} not an option")
        else:
            n = len(q["criteria"])
            sc = a.get("score")
            if not isinstance(sc, (int, float)) or not -1e-6 <= sc <= n - 1 + 1e-6:
                problems.append(f"{qid}: score={sc!r} outside [0, {n - 1}]")
    return {"valid": not problems, "problems": problems}


def prob_vector(answer: dict, q: dict):
    """Comparable probability vector + top-1 of one answer (for the parity notes)."""
    if q["type"] == "noul":
        p = float(answer["noul"])
        return [1 - p, p], int(p >= 0.5)
    probs = answer["probabilities"]
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    if q["type"] == "score" and not all(k in probs for k in keys):
        keys = list(q["criteria"])           # some runners key score levels by name
    v = [float(probs.get(k, 0.0)) for k in keys]
    return v, max(range(len(v)), key=v.__getitem__)


def parity(resp_a: dict, resp_b: dict, req: dict) -> str:
    dmax, agree = 0.0, 0
    for qid, q in req["questions"].items():
        va, ta = prob_vector(resp_a["answers"][qid], q)
        vb, tb = prob_vector(resp_b["answers"][qid], q)
        dmax = max(dmax, max(abs(x - y) for x, y in zip(va, vb)))
        agree += ta == tb
    return f"max|dp|={dmax:.4f}, top-1 agree {agree}/{len(req['questions'])}"


# =============================================================================================== worker
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


def _selftest_runner(params):
    class R:
        def __init__(self, mode, gb=0.5, s=10):
            self.mode, self.hold = mode, []
            if mode == "crash":
                os.abort()
            if mode == "alloc":
                while True:                       # touch real pages until a guard stops us
                    self.hold.append(bytearray(b"\x01") * int(100e6))
                    time.sleep(0.05)
            if mode == "sleep":
                time.sleep(s)
            if mode == "raise":
                raise RuntimeError("self-test: deliberate failure during load")

        def answer(self, req):
            ans = {}
            for qid, q in req["questions"].items():
                if q["type"] == "noul":
                    ans[qid] = {"type": "noul", "noul": 0.5}
                elif q["type"] == "choice":
                    k = list(q["criteria"])
                    ans[qid] = {"type": "choice", "choice": k[0], "probabilities": {x: 1 / len(k) for x in k}}
                else:
                    n = len(q["criteria"])
                    ans[qid] = {"type": "score", "score": (n - 1) / 2, "probabilities": {str(i): 1 / n for i in range(n)}}
            return {"model": "selftest", "answers": ans, "latency_ms": 0.0}

        def info(self):
            return {"runner": "selftest", "mode": self.mode}

        def close(self):
            pass
    return R(**params)


def worker(combo: dict, req: dict, result_path: Path, limit_gb: float):
    res = {"combo_id": combo["id"], "started": iso(), "stage": "start", "ok": False}

    def on_breach(reason):
        res.update(error=reason, guard="in-process", ok=False)
        write_json_atomic(result_path, res)

    sampler = InProcessSampler(limit_gb=limit_gb, sample_s=0.1, on_breach=on_breach).start()
    mx, runner = None, None
    try:
        res["stage"] = "import"
        t = time.perf_counter()
        if combo["backend"] == "mlx":
            import mlx.core as mx  # noqa: F811
        if combo["kind"] in ("kev", "laya", "nanojev", "anyjev"):
            import torch  # noqa: F401
        res["import_s"] = round(time.perf_counter() - t, 3)
        if mx is not None:
            mx.reset_peak_memory()
        res["stage"] = "load"
        t = time.perf_counter()
        if combo["kind"] == "selftest":
            runner = _selftest_runner(combo["params"])
        else:
            from adapters import build_runner
            runner = build_runner(combo["kind"], combo["params"])
        _sync(mx)
        res["load_s"] = round(time.perf_counter() - t, 3)
        res["info"] = runner.info()
        res["mem_after_load"] = sampler.read()
        res["stage"] = "request"
        t = time.perf_counter()
        resp = runner.answer(req)
        _sync(mx)
        res["latency_ms_single"] = round((time.perf_counter() - t) * 1000, 1)
        res["response"] = resp
        res["stage"] = "request_warm"
        t = time.perf_counter()
        resp2 = runner.answer(req)
        _sync(mx)
        res["latency_ms_warm"] = round((time.perf_counter() - t) * 1000, 1)
        res["response_warm"] = resp2
        res["validation"] = validate(resp, req)
        res["ok"] = res["validation"]["valid"]
        if not res["ok"]:
            res["error"] = "invalid response: " + "; ".join(res["validation"]["problems"])
        res["stage"] = "close"
        runner.close()
        res["stage"] = "done"
    except BaseException as e:  # noqa: BLE001 - every failure is a finding
        res.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        res.update(sampler.stop())
        if mx is not None:
            try:
                res["mlx_peak_gb"] = round(mx.get_peak_memory() / GB, 3)
            except Exception as e:  # noqa: BLE001
                res["mlx_peak_gb_error"] = str(e)
        res["finished"] = iso()
        write_json_atomic(result_path, res)


# =============================================================================================== parent
class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def row_from(combo, worker_res, guard_res, repo_sha, raw_path, results_root):
    pin_sha = pins.REPOS.get(combo.get("repo"), (None, None))[1]
    notes = [combo.get("notes", "")]
    if repo_sha and pin_sha and repo_sha != pin_sha:
        notes.append(f"REPO SHA {repo_sha} differs from pin {pin_sha}")
    d = {"combo_id": combo["id"], "model": combo["model"], "variant": combo["variant"], "backend": combo["backend"],
         "repo_sha": repo_sha or "", "hf_revision": combo.get("hf", ""),
         "raw_json": os.path.relpath(raw_path, results_root) if raw_path else ""}
    if combo.get("unsupported"):
        d.update(runs="no", error=combo["unsupported"], notes="; ".join(n for n in notes if n))
        return d
    w, g = worker_res or {}, guard_res or {}
    ok = bool(w.get("ok")) and g.get("returncode") == 0 and not g.get("killed_reason")
    def mx_(*vals):          # max of the measured values; 0/None = never sampled (e.g. a crash before the first sample)
        vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        return max(vals) if vals else None
    peak_fp = mx_(g.get("peak_phys_footprint_gb"), w.get("phys_footprint_peak_gb"), w.get("phys_footprint_lifetime_max_gb"))
    peak_rss = mx_(g.get("peak_rss_gb"), w.get("rss_peak_gb"))
    # peak_mem_gb: physical footprint (includes Metal memory) and the frameworks' own peaks; RSS only as a fallback,
    # because on Apple Silicon RSS misses GPU memory and counts memory-mapped weight files
    peak = mx_(peak_fp, w.get("mlx_peak_gb"), w.get("mps_driver_peak_gb"))
    if peak_fp is None:
        peak = mx_(peak, peak_rss)
        if peak is not None:
            notes.append("physical footprint unavailable: peak_mem_gb includes RSS")
    err = ""
    if not ok:
        parts = [g.get("killed_reason"), w.get("error")]
        if not parts[0] and not parts[1]:
            parts.append(describe_exit(g.get("returncode")))
            tail = [l[6:] for l in g.get("tail", []) if l.startswith("[err]")][-5:]
            if tail:
                parts.append("stderr: " + " / ".join(tail))
        if w.get("stage") and w.get("stage") != "done":
            parts.append(f"(stage: {w['stage']})")
        err = " ".join(p for p in parts if p)
        if peak is not None:
            notes.append("peak_mem_gb = peak observed until the failure")
    mem = (f"mem: footprint_parent={fmt(g.get('peak_phys_footprint_gb'))} footprint_worker_max={fmt(w.get('phys_footprint_lifetime_max_gb'))} "
           f"rss_parent={fmt(g.get('peak_rss_gb'))} rss_worker={fmt(w.get('rss_peak_gb'))} "
           f"mlx_peak={fmt(w.get('mlx_peak_gb'))} mps_driver={fmt(w.get('mps_driver_peak_gb'))} "
           f"min_sys_avail={fmt(g.get('min_avail_gb'))} swap {fmt(g.get('swap_start_gb'))}->{fmt(g.get('swap_peak_gb'))} GB")
    notes.append(mem)
    if g.get("swap_guard", "active") != "active":
        notes.append(f"swap guard {g['swap_guard']}")
    q = (w.get("info") or {}).get("quantization")
    if q:
        notes.append(f"quantized {q['quantized_modules']} modules ({q['bits']}-bit g{q['group_size']}), "
                     f"backbone {q['backbone_bytes_before'] / GB:.2f}->{q['backbone_bytes_after'] / GB:.2f} GB, "
                     f"skipped {len(q['skipped_modules'])}, pointer head fp32")
    swap_delta = (g["swap_peak_gb"] - g["swap_start_gb"]) if g.get("swap_peak_gb") is not None else None
    resp = w.get("response") or {}
    d.update(runs="yes" if ok else "no", load_s=w.get("load_s") if ok else None, peak_mem_gb=peak,
             peak_phys_footprint_gb=peak_fp, peak_rss_gb=peak_rss,
             phys_footprint_peak_gb_parent=g.get("peak_phys_footprint_gb"),
             phys_footprint_peak_gb_worker=w.get("phys_footprint_peak_gb"),
             phys_footprint_lifetime_max_gb=w.get("phys_footprint_lifetime_max_gb"), guard_metric=g.get("guard_metric"),
             latency_ms_single=w.get("latency_ms_single") if ok else None, error=err,
             notes="; ".join(n for n in notes if n), import_s=w.get("import_s"), latency_ms_warm=w.get("latency_ms_warm"),
             model_latency_ms_cold=resp.get("latency_ms"), model_latency_ms_warm=(w.get("response_warm") or {}).get("latency_ms"),
             rss_peak_gb_parent=g.get("peak_rss_gb"), rss_peak_gb_worker=w.get("rss_peak_gb"), mlx_peak_gb=w.get("mlx_peak_gb"),
             mps_driver_peak_gb=w.get("mps_driver_peak_gb"), min_sys_avail_gb=g.get("min_avail_gb"),
             swap_delta_gb=round(swap_delta, 3) if swap_delta is not None else None, wall_s=g.get("wall_s"),
             exit=describe_exit(g.get("returncode")), killed_reason=g.get("killed_reason") or "")
    return d


def write_csvs(rows_by_id: dict, order: list, results: Path):
    def cell(v):
        return fmt(v) if isinstance(v, float) else ("" if v is None else str(v).replace("\n", " | "))
    rows = [rows_by_id[i] for i in order if i in rows_by_id] + [r for i, r in rows_by_id.items() if i not in order]
    for name, cols in (("feasibility_matrix.csv", CSV_COLS), ("feasibility_matrix_detail.csv", DETAIL_COLS)):
        tmp = results / (name + ".partial")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            wr.writeheader()
            for r in rows:
                wr.writerow({c: cell(r.get(c)) for c in cols})
        os.replace(tmp, results / name)


def load_existing(results: Path) -> dict:
    p = results / "feasibility_matrix_detail.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["combo_id"]: r for r in csv.DictReader(f)}


def add_parity_notes(rows: dict, raw: dict, req: dict):
    """Kev variants against their reference on the same 3 questions (indicative only; the benchmark does this properly)."""
    refs = {"kev-0.8b": ["kev-0.8b/fp32", "kev-0.8b/fp32-cpu"], "kev-4b": ["kev-4b/full-bf16-merged"]}
    extra = {"kev-0.8b/full-bf16-merged": "kev-0.8b/bf16"}
    for cid, row in rows.items():
        if row.get("runs") != "yes" or cid not in raw:
            continue
        row["notes"] = (row.get("notes") or "").split("; vs ")[0]      # drop parity notes of an earlier run
        model = row["model"]
        cmp_to = [r for r in refs.get(model, []) if r != cid and r in raw and rows.get(r, {}).get("runs") == "yes"][:1]
        if cid in extra and extra[cid] in raw and rows.get(extra[cid], {}).get("runs") == "yes":
            cmp_to.append(extra[cid])
        for ref in cmp_to:
            try:
                row["notes"] += f"; vs {ref}: {parity(raw[cid]['response'], raw[ref]['response'], req)} (3 questions)"
            except Exception as e:  # noqa: BLE001
                row["notes"] += f"; vs {ref}: parity not computed ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--tf-base", choices=("base", "instruct"), default="base")
    ap.add_argument("--request", default=str(HERE / "smoke_request.json"))
    ap.add_argument("--results-dir", default=str(pins.results_dir()))
    ap.add_argument("--rss-limit-gb", type=float, default=14.0)
    ap.add_argument("--min-avail-gb", type=float, default=0.75)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--cooldown-s", type=float, default=10.0)
    ap.add_argument("--settle-avail-gb", type=float, default=4.0,
                    help="before each combination wait (max 120 s) until this much memory is available")
    ap.add_argument("--online", action="store_true", help="let workers reach the Hub (default: HF_HUB_OFFLINE=1)")
    ap.add_argument("--force-fallback", action="store_true",
                    help="run fallback combinations (jevmlx default model) even if their primary combinations ran")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    a = ap.parse_args()

    req_path = Path(a.request)
    req = json.loads(req_path.read_text(encoding="utf-8"))
    combos = selftest_combos() if a.selftest else smoke_combos(a.tf_base)
    by_id = {c["id"]: c for c in combos}

    if a.worker:                                                       # ---- worker mode
        worker(by_id[a.worker], req, Path(a.result), a.rss_limit_gb)
        return

    results = Path(a.results_dir)
    if a.selftest and results.resolve() == pins.results_dir().resolve():
        raise SystemExit("--selftest needs --results-dir pointing to a scratch folder (never results)")
    raw_dir = results / "raw" / "smoke"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = Tee(results / "logs" / f"smoke_{stamp}.log")
    only = {s for s in a.only.split(",") if s}
    skip = {s for s in a.skip.split(",") if s}
    unknown = (only | skip) - set(by_id)
    if unknown:
        raise SystemExit(f"unknown combination ids: {sorted(unknown)}; known: {sorted(by_id)}")
    selected = [c for c in combos if (not only or c["id"] in only) and c["id"] not in skip]
    req_sha = hashlib.sha256(req_path.read_bytes()).hexdigest()
    cache = pins.cache_dir()
    if not a.selftest:
        pins.assert_not_on_drive(cache)

    log(f"# smoke test {iso()}  python={sys.executable}")
    log(f"cache={cache}  HF_HOME={os.environ.get('HF_HOME')}  results={results}")
    log(f"request={req_path} sha256={req_sha}  questions={list(req['questions'])}")
    log(f"guards: rss>{a.rss_limit_gb:g} GB | avail<{a.min_avail_gb:g} GB & swap+>{a.swap_growth_gb:g} GB | timeout {a.timeout_s:g} s")
    repo_shas = {name: git_sha(pins.repo_dir(name)) for name in pins.REPOS}
    for name, (url, sha) in pins.REPOS.items():
        got = repo_shas[name]
        log(f"repo {name:8} pin {sha[:12]} have {got[:12] if got else 'MISSING'}"
            f"{'' if (got == sha or got is None) else '  <-- MISMATCH'}")
    plan = []
    for c in selected:
        missing = [n for n in c.get("needs", []) if not done_path(n).exists()]
        status = "unsupported" if c.get("unsupported") else ("missing prerequisites " + ",".join(missing) if missing else "ready")
        plan.append((c, missing, status))
        log(f"  {c['id']:28} {c['backend']:16} {status}")
    if a.dry_run:
        log("dry run: no model loaded, nothing written except this log")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(results)
    rows, raw = dict(existing), {}
    for p in raw_dir.glob("*.json"):            # earlier responses, for parity notes across partial reruns
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(j, dict) and "worker" in j and (j["worker"] or {}).get("response"):
                raw[j["combo"]["id"]] = j["worker"]
        except Exception:  # noqa: BLE001
            pass
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if not a.online:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import psutil

    for c, missing, status in plan:
        cid = c["id"]
        raw_path = raw_dir / f"{slug(cid)}.json"
        repo_sha = repo_shas.get(c.get("repo"))
        record = {"combo": c, "request_path": str(req_path), "request_sha256": req_sha, "repo_sha_actual": repo_sha,
                  "started": iso()}
        if c.get("unsupported"):
            log(f"[unsupported] {cid}: {c['unsupported']}")
            rows[cid] = row_from(c, None, None, repo_sha, raw_path, results)
            record["result"] = "not attempted (unsupported)"
            write_json_atomic(raw_path, record)
            write_csvs(rows, [x["id"] for x in combos], results)
            continue
        if c.get("fallback_for") and not a.force_fallback:
            ran = [x for x in c["fallback_for"] if rows.get(x, {}).get("runs") == "yes"]
            if ran:
                log(f"[not needed] {cid}: fallback for {c['fallback_for']}; these ran: {ran} -> no row written")
                rows.pop(cid, None)
                for x in ran:
                    if "default-model fallback not run" not in rows[x].get("notes", ""):
                        rows[x]["notes"] = (rows[x].get("notes") or "") + "; jevmlx ran this Qwen3.5 checkpoint, so its default-model fallback was not run"
                write_csvs(rows, [x["id"] for x in combos], results)
                continue
            log(f"[fallback] {cid}: none of {c['fallback_for']} ran -> running the fallback")
        if missing and c.get("on_demand_prepare") and not a.selftest:
            from combos import prepare_tasks
            tasks = {t["id"]: t for t in prepare_tasks(a.tf_base)}
            need, stack = [], list(missing)
            while stack:                                  # the missing tasks and their own unfinished prerequisites
                tid = stack.pop()
                if tid in need or done_path(tid).exists():
                    continue
                need.append(tid)
                stack += tasks.get(tid, {}).get("needs", [])
            log(f"[prepare on demand] {cid}: {need}")
            penv = {k: v for k, v in env.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
            subprocess.run([sys.executable, str(HERE / "prepare_models.py"), "--only", ",".join(reversed(need)),
                            "--tf-base", a.tf_base], env=penv, cwd=str(HERE))
            missing = [n for n in c.get("needs", []) if not done_path(n).exists()]
        if missing:
            err = f"prerequisite not available: {', '.join(missing)} (see raw/smoke/prepare_report.json)"
            log(f"[skip] {cid}: {err}")
            rows[cid] = row_from(dict(c, unsupported=err), None, None, repo_sha, raw_path, results)
            record["result"] = err
            write_json_atomic(raw_path, record)
            write_csvs(rows, [x["id"] for x in combos], results)
            continue
        # let memory settle after the previous worker
        t_wait = time.time()
        time.sleep(a.cooldown_s)
        while psutil.virtual_memory().available < a.settle_avail_gb * GB and time.time() - t_wait < 120:
            time.sleep(2)
        log(f"[run ] {cid}  (available {psutil.virtual_memory().available / GB:.2f} GB after {time.time() - t_wait:.0f} s settle)")
        record["before"] = system_snapshot(f"before {cid}")
        log(f"       pmset therm: {record['before']['pmset_therm'][:200]!r}")
        result_path = raw_dir / f".{slug(cid)}.worker.json"
        if result_path.exists():
            result_path.unlink()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", cid, "--result", str(result_path),
               "--request", str(req_path), "--tf-base", a.tf_base, "--rss-limit-gb", str(a.rss_limit_gb)]
        if a.selftest:
            cmd.append("--selftest")
        g = run_guarded(cmd, log_path=log.path, env=env, cwd=str(HERE), rss_limit_gb=a.rss_limit_gb,
                        min_avail_gb=a.min_avail_gb, swap_growth_gb=a.swap_growth_gb, timeout_s=a.timeout_s)
        w = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        if result_path.exists():
            result_path.unlink()
        record.update(guard=g, worker=w, after=system_snapshot(f"after {cid}"), finished=iso())
        write_json_atomic(raw_path, record)
        rows[cid] = row_from(c, w, g, repo_sha, raw_path, results)
        if w and w.get("ok"):
            raw[cid] = w
        r = rows[cid]
        log(f"[{'ok' if r['runs'] == 'yes' else 'FAIL'}  ] {cid}: load_s={fmt(r.get('load_s'))} peak_mem_gb={fmt(r.get('peak_mem_gb'))} "
            f"latency_ms={fmt(r.get('latency_ms_single'), 1)} {('error=' + r['error']) if r.get('error') else ''}")
        write_csvs(rows, [x["id"] for x in combos], results)
    add_parity_notes(rows, raw, req)
    write_csvs(rows, [x["id"] for x in combos], results)
    n_yes = sum(1 for c in selected if rows.get(c["id"], {}).get("runs") == "yes")
    log(f"# finished {iso()}: {n_yes}/{len(selected)} combinations ran; CSV: {results / 'feasibility_matrix.csv'}")


if __name__ == "__main__":
    main()
