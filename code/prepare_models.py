"""Download the pinned weights and build the derived checkpoints the smoke test needs, all under ~/jevlike_cache.

Idempotent: a finished task leaves models/.done/<task>.json and is skipped next time (--force to redo).
Each task runs in its own process under guard.run_guarded (RSS/swap/time watchdog), sequentially.
Nothing is written to Google Drive except the log and prepare_report.json in results.

    python prepare_models.py --dry-run            # plan + disk estimate, touches nothing
    python prepare_models.py                       # run every task not yet done
    python prepare_models.py --only dl-laya,merge-kev-4b-stream
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pins  # noqa: E402
from combos import prepare_tasks  # noqa: E402
from guard import describe_exit, run_guarded, system_snapshot, write_json_atomic  # noqa: E402

os.environ.setdefault("HF_HOME", str(pins.cache_dir() / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def done_path(task_id: str) -> Path:
    return pins.models_dir() / ".done" / f"{task_id}.json"


def du_gb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path, followlinks=True):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return round(total / 1e9, 3)


# ------------------------------------------------------------------------------------------------ task bodies
def run_task(task: dict) -> dict:
    kind = task["kind"]
    if kind == "download":
        from huggingface_hub import snapshot_download
        repo, sha, note = pins.HF[task["key"]]
        kw = dict(revision=sha, allow_patterns=task["patterns"])
        if task.get("local_dir"):
            kw["local_dir"] = task["local_dir"]
        path = snapshot_download(repo, **kw)
        out = {"repo": repo, "revision": sha, "note": note, "path": path, "size_gb": du_gb(Path(path))}
        if task.get("kev_meta"):
            import torch
            meta = torch.load(Path(path) / "head.pt", map_location="cpu")
            base_key = {"kev-0.8b": "qwen3.5-0.8b-base", "kev-4b": "qwen3.5-4b-base"}[task["key"]]
            out["head_meta"] = {k: meta.get(k) for k in ("base", "base_revision", "lora", "head_dim", "weights",
                                                         "weights_dtype", "temperature", "option_isolation")}
            want = pins.HF[base_key]
            out["base_matches_pin"] = (meta.get("base"), meta.get("base_revision")) == (want[0], want[1])
            if not out["base_matches_pin"]:
                raise RuntimeError(f"head.pt names base {meta.get('base')}@{meta.get('base_revision')}, pins.py has "
                                   f"{want[0]}@{want[1]}: update pins.py (the loader downloads the head.pt revision)")
        return out
    if kind == "kev_official_merge":
        out_dir = Path(task["out"])
        if (out_dir / "checkpoint").exists():
            return {"skipped": "already merged", "checkpoint": str(out_dir / "checkpoint")}
        kev = pins.repo_dir("kev")
        # Launched via run_kev_script.py, which binds `scripts` to Kev's scripts/ dir: an installed distribution in the
        # shared venv ships its own top-level `scripts` package that shadows Kev's namespace package, so both the
        # file-path and the `-m` invocation failed on 2026-11-13 (see the deviation log #2).
        launcher = Path(__file__).resolve().parent / "run_kev_script.py"
        cmd = [sys.executable, str(launcher), str(kev), "merge_lora_checkpoint.py", "--lora", task["lora"],
               "--out", str(out_dir), "--weights_dtype", "bf16"]
        r = subprocess.run(cmd, cwd=str(kev))
        if r.returncode != 0:
            raise RuntimeError(f"merge_lora_checkpoint.py exited {r.returncode}")
        return {"checkpoint": str(out_dir / "checkpoint"), "report": json.loads((out_dir / "merge.json").read_text())}
    if kind == "kev_stream_merge":
        from kev_merge_stream import merge
        out_dir = Path(task["out"])
        if (out_dir / "checkpoint").exists():
            return {"skipped": "already merged", "checkpoint": str(out_dir / "checkpoint")}
        return merge(task["lora"], out_dir)
    if kind == "merge_compare":
        from kev_merge_stream import compare
        rep = compare(Path(task["a"]), Path(task["b"]))
        dst = pins.results_dir() / "raw" / "smoke" / "merge_parity_kev-0.8b.json"
        write_json_atomic(dst, rep)
        return {"report": str(dst), "verdict": rep["verdict"], "max_abs_diff_overall": rep["max_abs_diff_overall"],
                "elements_differing": rep["elements_differing"], "config_diff": rep["config_diff"]}
    if kind == "jevmlx_link":
        from huggingface_hub import snapshot_download
        repo, sha, _ = pins.HF[task["key"]]
        path = Path(snapshot_download(repo, revision=sha, allow_patterns=pins.KEV_PATTERNS, local_files_only=True))
        link = Path(task["link"])
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and Path(os.readlink(link)) != path:
            link.unlink()
        if not link.exists():
            link.symlink_to(path, target_is_directory=True)
        return {"link": str(link), "target": str(path)}
    if kind == "mlx_convert":
        from mlx_lm import convert
        dst = Path(task["dst"])
        if dst.exists():
            return {"skipped": "already converted", "path": str(dst)}
        tmp = dst.with_name(dst.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)          # our own half-written conversion from an interrupted run
        convert(hf_path=str(task["src"]), mlx_path=str(tmp), quantize=True, q_bits=task["bits"], q_group_size=task["group_size"])
        tmp.rename(dst)
        cfg = json.loads((dst / "config.json").read_text())
        return {"path": str(dst), "size_gb": du_gb(dst), "quantization": cfg.get("quantization") or cfg.get("quantization_config"),
                "method": f"mlx_lm.convert(quantize=True, q_bits={task['bits']}, q_group_size={task['group_size']})"}
    raise ValueError(f"unknown task kind {kind}")


# ------------------------------------------------------------------------------------------------ orchestration
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tf-base", choices=("base", "instruct"), default="base")
    ap.add_argument("--rss-limit-gb", type=float, default=14.0)
    ap.add_argument("--min-avail-gb", type=float, default=0.75)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--disk-margin-gb", type=float, default=5.0)
    ap.add_argument("--task", help=argparse.SUPPRESS)          # worker mode
    ap.add_argument("--task-json", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.task:                                                    # ---- worker
        task = json.loads(a.task_json)
        t0 = time.time()
        result = run_task(task)
        result["seconds"] = round(time.time() - t0, 1)
        write_json_atomic(done_path(task["id"]), {"task": task, "result": result, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(json.dumps({"task": task["id"], "ok": True}, default=str))
        return

    cache = pins.cache_dir()
    pins.assert_not_on_drive(cache)
    tasks = prepare_tasks(a.tf_base)
    only = {s for s in a.only.split(",") if s}
    if only:
        unknown = only - {t["id"] for t in tasks}
        if unknown:
            raise SystemExit(f"unknown task ids: {sorted(unknown)}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = pins.results_dir() / "logs" / f"prepare_{stamp}.log"
    report_path = pins.results_dir() / "raw" / "smoke" / "prepare_report.json"
    # on-demand tasks (the jevmlx default-model fallback) run only when asked for by id (smoke_test.py does that)
    todo = [t for t in tasks if (t["id"] in only) or (not only and not t.get("on_demand"))]
    need_gb = sum(t["gb"] for t in todo if a.force or not done_path(t["id"]).exists())
    free_gb = shutil.disk_usage(cache if cache.exists() else Path.home()).free / 1e9
    print(f"cache: {cache}\nHF_HOME: {os.environ['HF_HOME']}\nlog: {log}\nreport: {report_path}")
    print(f"disk: {free_gb:.1f} GB free, about {need_gb:.1f} GB still to write (+{a.disk_margin_gb:g} GB margin)")
    for t in todo:
        state = "done" if done_path(t["id"]).exists() and not a.force else "todo"
        print(f"  [{state:4}] {t['id']:32} {t['kind']:20} ~{t['gb']:.1f} GB  needs={t.get('needs', [])}")
    if a.dry_run:
        print("dry run: nothing executed")
        return
    if free_gb < need_gb + a.disk_margin_gb:
        raise SystemExit(f"not enough disk: {free_gb:.1f} GB free < {need_gb:.1f} + {a.disk_margin_gb:g} GB needed")

    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    failed = {}
    for t in todo:
        tid = t["id"]
        if done_path(tid).exists() and not a.force:
            report[tid] = {"status": "done", **json.loads(done_path(tid).read_text())}
            continue
        missing = [n for n in t.get("needs", []) if not done_path(n).exists()]
        if missing:
            failed[tid] = f"prerequisite not available: {missing}"
            report[tid] = {"status": "skipped", "error": failed[tid]}
            print(f"[skip] {tid}: {failed[tid]}")
            continue
        free_now = shutil.disk_usage(cache).free / 1e9
        if free_now < t["gb"] + a.disk_margin_gb:
            failed[tid] = f"insufficient disk: {free_now:.1f} GB free < {t['gb'] + a.disk_margin_gb:.1f} GB"
            report[tid] = {"status": "failed", "error": failed[tid]}
            print(f"[fail] {tid}: {failed[tid]}")
            continue
        print(f"[run ] {tid}")
        pre = system_snapshot(f"before {tid}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--task", tid, "--task-json", json.dumps(t)]
        g = run_guarded(cmd, log_path=log, cwd=str(pins.code_dir()), rss_limit_gb=a.rss_limit_gb, min_avail_gb=a.min_avail_gb,
                        swap_growth_gb=a.swap_growth_gb, timeout_s=a.timeout_s)
        post = system_snapshot(f"after {tid}")
        ok = g["returncode"] == 0 and not g["killed_reason"] and done_path(tid).exists()
        entry = {"status": "done" if ok else "failed", "guard": g, "before": pre, "after": post}
        if ok:
            entry.update(json.loads(done_path(tid).read_text()))
        else:
            err = g["killed_reason"] or describe_exit(g["returncode"])
            last = [l for l in g["tail"] if l.startswith("[err]")][-6:]
            entry["error"] = err + ((" | " + " / ".join(x[6:] for x in last)) if last else "")
            failed[tid] = entry["error"]
            print(f"[fail] {tid}: {entry['error']}")
        report[tid] = entry
        write_json_atomic(report_path, report)
    write_json_atomic(report_path, report)
    print(f"prepare finished: {sum(1 for t in todo if report.get(t['id'], {}).get('status') == 'done')}/{len(todo)} done; "
          f"failed: {sorted(failed) or 'none'}")


if __name__ == "__main__":
    main()
