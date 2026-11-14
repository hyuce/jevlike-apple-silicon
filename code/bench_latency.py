"""Latency / memory / context-scaling benchmark (protocol sections 6-8, RQ4).

    python bench_latency.py --dry-run                 # plan + build the synthetic workloads (needs the Qwen3.5 tokenizer)
    python bench_latency.py                           # every runnable combo (incl. CPU references), sequential
    python bench_latency.py --only kev-0.8b/bf16 --reps 3 --contexts 512,2048   # quick check

Per combo, in ONE guarded worker process (load once, then):
  A  question-count scaling, short ticket state, n_questions in {1, 5, 20}:
       3 warm-up calls, `reps` COLD calls (every call a new state: a unique ticket number in front, so no state cache
       can be reused), then 1 priming call + `reps` WARM calls (the identical request repeated).
  B  context scaling, 5 questions, state = unrelated filler text + the ticket, padded to {512, ..., 16384} tokens
     (counted with the Qwen3.5 tokenizer; other tokenizers differ -- the actual count per model is not re-measured):
       1 warm-up, `reps` COLD (unique prefix), 1 priming + `reps` WARM.
     If one call at a length takes longer than --max-call-s, the remaining reps at that length and all longer lengths
     are skipped and recorded as "not assessed (budget)"; --combo-budget-min caps a whole combo the same way.
Every call is timed as wall time of runner.answer() incl. device sync; answers are checked for finite probabilities.
Memory per cell: sampler peaks (phys_footprint / RSS / MLX active / MPS driver, 100 ms) reset at each cell start, and
mx.get_peak_memory() reset per cell for MLX backends.
Outputs: results/bench/latency/<combo>/{calls.jsonl, cells.json, worker.json, system.json}
         results/latency_summary.csv (one row per combo x cell), results/bench/latency/workload.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
EXCLUDE_DEFAULT = {"torch-cuda-only", "none"}
TICKET = ("My blender arrived three days late and the box was crushed. It does not turn on at all. "
          "I do not want a refund, please send a replacement unit.")
TOKENIZER = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
FILLER_WORDS = ("the of and to in a is that for it as was with be by on not he this are or his from at which but have an "
                "they you were her she there been one all we their has would when if so what no out up into more other "
                "about some time could them only new these may first then do any like my now over such our man me even "
                "most made after also did many before must through back years where much your way well down should "
                "because each just those people how too little state good very make world still own see men work long "
                "get here between both life being under never day same another know while last might us great old year "
                "off come since against go came right used take three river garden window market letter mountain report "
                "village engine summer winter station paper music island harbor bridge forest kitchen story signal").split()
SUMMARY_COLS = ["combo_id", "model", "variant", "backend", "cell", "phase", "n_questions", "target_tokens", "state_tokens_qwen",
                "n", "n_ok", "median_ms", "q1_ms", "q3_ms", "mean_ms", "std_ms", "p95_ms", "min_ms", "max_ms",
                "drift_ratio", "peak_phys_footprint_gb", "peak_rss_gb", "peak_mlx_gb", "peak_mps_driver_gb", "status", "note"]


def iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slug(cid):
    return cid.replace("/", "__").replace(" ", "_")


# ----------------------------------------------------------------------------------------------- workload
def question_pool():
    """20 deterministic ticket questions cycling noul / choice / score (synthetic workload, not evaluation data)."""
    nouls = ["Does the customer report a defective or damaged product?", "Is the customer asking for a refund?",
             "Does the message mention a delivery delay?", "Is the customer angry?", "Does the customer ask for a replacement?",
             "Is a product name mentioned?", "Does the ticket require a call back?"]
    choices = [("Which team should handle this ticket?", {"returns": "Exchanges, replacements, refunds, wrong or damaged items",
                                                          "shipping": "Delivery status, delays, lost packages",
                                                          "billing": "Charges, invoices, payment problems"}),
               ("What is the main topic?", {"product": None, "delivery": None, "payment": None, "account": None}),
               ("Which channel fits the reply best?", {"email": None, "phone": None, "chat": None}),
               ("What product category is involved?", {"kitchen": None, "electronics": None, "clothing": None, "garden": None}),
               ("What should happen next?", {"ship replacement": None, "issue refund": None, "ask for photos": None}),
               ("What tone should the reply use?", {"apologetic": None, "neutral": None, "formal": None}),
               ("Which priority queue?", {"p1": None, "p2": None, "p3": None, "p4": None})]
    scores = [("How urgent is this ticket?", ["Low", "Medium", "High"]),
              ("How frustrated is the customer?", ["Calm", "Annoyed", "Very upset"]),
              ("How clear is the request?", ["Unclear", "Somewhat clear", "Very clear"]),
              ("How costly is resolving this likely to be?", ["Cheap", "Moderate", "Expensive"]),
              ("How likely is churn?", ["Unlikely", "Possible", "Likely"]),
              ("How much detail does the ticket give?", ["Little", "Some", "A lot"])]
    pool, i = [], 0
    while len(pool) < 20:
        k = len(pool) % 3
        if k == 0 and nouls:
            pool.append(("noul", {"type": "noul", "instructions": nouls.pop(0)}))
        elif k == 1 and choices:
            t, c = choices.pop(0)    # every option gets a text description (no nulls: not every runner handles them)
            c = {k: (v if v else k.capitalize()) for k, v in c.items()}
            pool.append(("choice", {"type": "choice", "instructions": t, "criteria": c}))
        elif scores:
            t, c = scores.pop(0); pool.append(("score", {"type": "score", "instructions": t, "criteria": c}))
        i += 1
    return {f"q{j:02d}_{kind}": q for j, (kind, q) in enumerate(pool)}


def build_contexts(targets, out_path: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    rng = random.Random(0)
    words = [rng.choice(FILLER_WORDS) for _ in range(40000)]
    tail = "\n\nTicket #{N}: " + TICKET
    tail_tokens = ntok(tail.replace("{N}", "10000"))
    contexts = {}
    for T in targets:
        lo, hi = 0, len(words)
        while lo < hi:                       # largest word count whose filler + tail stays <= T tokens
            mid = (lo + hi + 1) // 2
            if ntok(" ".join(words[:mid])) + tail_tokens <= T:
                lo = mid
            else:
                hi = mid - 1
        filler = " ".join(words[:lo])
        # measured as served (cold calls prepend "Ref <n>. ", ~4 tokens, not included in the binary search)
        contexts[str(T)] = {"filler": filler, "tokens_qwen": ntok("Ref 100001. " + filler + tail.replace("{N}", "100001"))}
    write_json_atomic(out_path, {"tokenizer": TOKENIZER, "ticket": TICKET, "tail_template": tail, "contexts": contexts,
                                 "questions": question_pool()})
    return contexts


# ----------------------------------------------------------------------------------------------- worker
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


def _finite(body):
    try:
        for a in body["answers"].values():
            vals = [a["noul"]] if a.get("type") == "noul" else list(a["probabilities"].values())
            if not all(math.isfinite(float(v)) for v in vals):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def worker(combo, workload_path: Path, out_dir: Path, a):
    wl = json.loads(workload_path.read_text(encoding="utf-8"))
    qs = wl["questions"]; qids = list(qs)
    res = {"combo_id": combo["id"], "started": iso(), "ok": False}
    cells = []

    def on_breach(reason):
        res.update(error=reason, guard="in-process", ok=False, cells_done=len(cells))
        write_json_atomic(out_dir / "worker.json", res)
        write_json_atomic(out_dir / "cells.json", cells)

    sampler = InProcessSampler(limit_gb=a.rss_limit_gb, sample_s=0.1, on_breach=on_breach).start()
    mx = None
    nonce = [100000]
    t_combo = time.perf_counter()

    def req(state, nq):
        return {"state": state, "model": "kev-latest", "questions": {q: qs[q] for q in qids[:nq]}}

    def ticket(filler=None):
        # the unique number is the FIRST thing in the state, so no prefix/KV cache can reuse an earlier call's state
        nonce[0] += 1
        if filler is None:
            return wl["tail_template"].replace("{N}", str(nonce[0])).lstrip("\n")
        return f"Ref {nonce[0]}. " + filler + wl["tail_template"].replace("{N}", str(nonce[0]))

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
        res["after_load"] = sampler.read()
        calls = open(out_dir / "calls.jsonl", "a", encoding="utf-8")

        def call(cell, phase, rep, r):
            t0 = time.perf_counter()
            try:
                body = runner.answer(r)
                _sync(mx)
                ms = (time.perf_counter() - t0) * 1000
                ok = _finite(body)
                err = None if ok else "non-finite probabilities"
            except Exception as e:  # noqa: BLE001
                ms, ok, err = (time.perf_counter() - t0) * 1000, False, f"{type(e).__name__}: {e}"
            calls.write(json.dumps({"cell": cell, "phase": phase, "rep": rep, "latency_ms": round(ms, 3), "ok": ok,
                                    "error": err, "t": iso()}) + "\n")
            calls.flush()
            return ms, ok, err

        def run_cell(cell, nq, make_state, target=None, tokens=None):
            if (time.perf_counter() - t_combo) / 60 > a.combo_budget_min:
                cells.append({"cell": cell, "status": "not assessed (combo budget)", "n_questions": nq, "target_tokens": target})
                return False
            for k in sampler.peak:
                sampler.peak[k] = 0
            if mx is not None:
                mx.reset_peak_memory()
            out = {"cell": cell, "n_questions": nq, "target_tokens": target, "state_tokens_qwen": tokens,
                   "cold": [], "warm": [], "errors": [], "status": "ok"}
            ms, ok, err = call(cell, "warmup", 0, req(make_state(), nq))
            if err:
                out["errors"].append(err)
            if ms / 1000 > a.max_call_s:
                out["status"] = f"not assessed (budget: warm-up call {ms / 1000:.1f} s > {a.max_call_s:g} s)"
            else:
                for i in range(a.warmups - 1 if target is None else 0):
                    call(cell, "warmup", i + 1, req(make_state(), nq))
                for i in range(a.reps):
                    ms, ok, err = call(cell, "cold", i, req(make_state(), nq))
                    out["cold"].append(round(ms, 3) if ok else None)
                    if err:
                        out["errors"].append(err)
                same = req(make_state(), nq)
                call(cell, "prime", 0, same)
                for i in range(a.reps):
                    ms, ok, err = call(cell, "warm", i, same)
                    out["warm"].append(round(ms, 3) if ok else None)
                    if err:
                        out["errors"].append(err)
            sampler.read()
            out["peaks"] = sampler.peaks_gb()
            if mx is not None:
                out["peaks"]["mlx_peak_gb"] = round(mx.get_peak_memory() / GB, 3)
            out["errors"] = sorted(set(out["errors"]))[:5]
            cells.append(out)
            write_json_atomic(out_dir / "cells.json", cells)
            return out["status"] == "ok"

        for nq in a.questions:
            run_cell(f"A_q{nq}", nq, lambda: ticket())
        ctx = wl["contexts"]
        for T in a.contexts:
            c = ctx[str(T)]
            okc = run_cell(f"B_ctx{T}", 5, lambda c=c: ticket(c["filler"]), target=T, tokens=c["tokens_qwen"])
            if not okc:
                for T2 in [x for x in a.contexts if x > T]:
                    cells.append({"cell": f"B_ctx{T2}", "n_questions": 5, "target_tokens": T2,
                                  "status": f"not assessed (skipped after B_ctx{T} exceeded budget)"})
                break
        calls.close()
        runner.close()
        res["ok"] = True
    except BaseException as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        res.update(sampler.stop())
        res["finished"] = iso()
        write_json_atomic(out_dir / "cells.json", cells)
        write_json_atomic(out_dir / "worker.json", res)


# ----------------------------------------------------------------------------------------------- parent
def _stats(xs):
    import numpy as np
    v = np.array([x for x in xs if x is not None], dtype=float)
    if not len(v):
        return {}
    k = max(1, min(3, len(v) // 3))
    return {"n_ok": int(len(v)), "median_ms": float(np.median(v)), "q1_ms": float(np.quantile(v, .25)),
            "q3_ms": float(np.quantile(v, .75)), "mean_ms": float(v.mean()), "std_ms": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "p95_ms": float(np.quantile(v, .95)), "min_ms": float(v.min()), "max_ms": float(v.max()),
            # protocol section 8: drift flag when the last reps run > 15% slower than the first ones
            "drift_ratio": float(np.median(v[-k:]) / np.median(v[:k])) if len(v) >= 2 * k else None}


def summarize(combo, out_dir: Path, workload):
    cells = json.loads((out_dir / "cells.json").read_text()) if (out_dir / "cells.json").exists() else []
    rows = []
    for c in cells:
        for phase in ("cold", "warm"):
            vals = c.get(phase) or []
            s = _stats(vals)
            p = c.get("peaks") or {}
            note = "; ".join(c.get("errors") or [])
            if s.get("drift_ratio") and s["drift_ratio"] > 1.15:
                note = (note + "; " if note else "") + "drift > 15% (possible throttling)"
            rows.append({"combo_id": combo["id"], "model": combo["model"], "variant": combo["variant"],
                         "backend": combo["backend"], "cell": c["cell"], "phase": phase, "n_questions": c.get("n_questions"),
                         "target_tokens": c.get("target_tokens"), "state_tokens_qwen": c.get("state_tokens_qwen"),
                         "n": len(vals), **{k: s.get(k) for k in ("n_ok", "median_ms", "q1_ms", "q3_ms", "mean_ms", "std_ms",
                                                                  "p95_ms", "min_ms", "max_ms", "drift_ratio")},
                         "peak_phys_footprint_gb": p.get("phys_footprint_peak_gb"), "peak_rss_gb": p.get("rss_peak_gb"),
                         "peak_mlx_gb": p.get("mlx_peak_gb"), "peak_mps_driver_gb": p.get("mps_driver_peak_gb"),
                         "status": c.get("status"), "note": note})
    return rows


def write_summary(results: Path, new_rows):
    path = results / "latency_summary.csv"
    keep = []
    ids = {r["combo_id"] for r in new_rows}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f) if r["combo_id"] not in ids]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in keep + new_rows:
            w.writerow({k: r.get(k) for k in SUMMARY_COLS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--tf-base", choices=("base", "instruct"), default="base")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--questions", default="1,5,20")
    ap.add_argument("--contexts", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--max-call-s", type=float, default=60.0)
    ap.add_argument("--combo-budget-min", type=float, default=45.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rss-limit-gb", type=float, default=14.0)
    ap.add_argument("--min-avail-gb", type=float, default=0.75)
    ap.add_argument("--swap-growth-gb", type=float, default=12.0)
    ap.add_argument("--cooldown-s", type=float, default=30)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--workload", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.questions = [int(x) for x in str(a.questions).split(",")]
    a.contexts = [int(x) for x in str(a.contexts).split(",")]
    combos = {c["id"]: c for c in smoke_combos(a.tf_base)}
    if a.worker:
        worker(combos[a.worker], Path(a.workload), Path(a.out_dir), a)
        return

    results = pins.results_dir()
    runnable = set()
    if (results / "feasibility_matrix.csv").exists():
        with open(results / "feasibility_matrix.csv", encoding="utf-8") as f:
            runnable = {f"{r['model']}|{r['variant']}" for r in csv.DictReader(f) if r["runs"] == "yes"}
    if a.only:
        ids = [x.strip() for x in a.only.split(",") if x.strip()]
        bad = [x for x in ids if x not in combos]
        if bad:
            raise SystemExit(f"unknown combo(s): {bad}")
    else:
        ids = [cid for cid, c in combos.items() if not c.get("unsupported") and c["backend"] not in EXCLUDE_DEFAULT
               and f"{c['model']}|{c['variant']}" in runnable]
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    ids = [i for i in ids if i not in skip]
    bench = results / "bench" / "latency"
    bench.mkdir(parents=True, exist_ok=True)
    wl_path = bench / "workload.json"
    ctx = build_contexts(sorted(set(a.contexts)), wl_path)
    print(f"# latency benchmark {iso()}  combos={len(ids)}  reps={a.reps}  questions={a.questions}  contexts={a.contexts}")
    for T in a.contexts:
        print(f"  context {T:>6}: {ctx[str(T)]['tokens_qwen']} Qwen3.5 tokens")
    for cid in ids:
        print(f"  {cid:32s} {combos[cid]['backend']}")
    if a.dry_run:
        return
    import psutil
    log_path = results / "logs" / f"bench_latency_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    for cid in ids:
        c = combos[cid]
        out_dir = bench / slug(cid)
        if out_dir.exists():                         # a latency run is not resumable: start the combo afresh
            for f in ("calls.jsonl", "cells.json", "worker.json", "system.json"):
                (out_dir / f).unlink(missing_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        time.sleep(a.cooldown_s)
        t_wait = time.time()
        while psutil.virtual_memory().available < 4 * GB and time.time() - t_wait < 120:
            time.sleep(2)
        before = system_snapshot(f"before {cid}")
        print(f"[run ] {cid}  available {psutil.virtual_memory().available / GB:.2f} GB  therm={before['pmset_therm'][:80]!r}", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", cid, "--workload", str(wl_path),
               "--out-dir", str(out_dir), "--tf-base", a.tf_base, "--reps", str(a.reps), "--warmups", str(a.warmups),
               "--questions", ",".join(map(str, a.questions)), "--contexts", ",".join(map(str, a.contexts)),
               "--max-call-s", str(a.max_call_s), "--combo-budget-min", str(a.combo_budget_min),
               "--rss-limit-gb", str(a.rss_limit_gb)]
        g = run_guarded(cmd, log_path=log_path, env=env, cwd=str(HERE), rss_limit_gb=a.rss_limit_gb,
                        min_avail_gb=a.min_avail_gb, swap_growth_gb=a.swap_growth_gb,
                        timeout_s=a.combo_budget_min * 60 * 1.5 + 600)
        write_json_atomic(out_dir / "system.json", {"before": before, "after": system_snapshot(f"after {cid}"), "guard": g})
        rows = summarize(c, out_dir, None)
        if g.get("killed_reason"):
            rows.append({"combo_id": cid, "model": c["model"], "variant": c["variant"], "backend": c["backend"],
                         "cell": "ALL", "phase": "-", "status": f"killed: {g['killed_reason']}"})
        write_summary(results, rows)
        a_q5 = [r for r in rows if r.get("cell") == "A_q5" and r.get("phase") == "warm"]
        print(f"[done] {cid}: cells={len({r['cell'] for r in rows})}  A_q5 warm median="
              f"{a_q5[0]['median_ms'] if a_q5 else None} ms  killed={g.get('killed_reason')}", flush=True)


if __name__ == "__main__":
    main()
