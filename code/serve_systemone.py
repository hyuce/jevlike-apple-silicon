"""Serve ANY smoke-test combination behind a TypeSafe-compatible POST /v1/systemone (for the benchmark phase).

    python serve_systemone.py --combo kev-0.8b/4bit --port 8010
    python -m kev.benchmark --remote http://127.0.0.1:8010 --suite evals/v4/transfer-v4 --out runs/kev08-q4   # (from the kev repo)

One model per process, requests answered one at a time (protocol section 8: one GPU job at a time). The runner is
the same adapters.py object the smoke test validated. Plain Kev variants can also use upstream `python -m kev.serve`
(see results/environment_notes.md); this server is what gives quantized Kev, Laya, NanoJev, AnyJev and jevmlx the
same endpoint shape.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pins  # noqa: E402
from combos import smoke_combos  # noqa: E402

os.environ.setdefault("HF_HOME", str(pins.cache_dir() / "hf"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--combo", required=True)
    ap.add_argument("--tf-base", choices=("base", "instruct"), default="base")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    combos = {c["id"]: c for c in smoke_combos(a.tf_base)}
    c = combos.get(a.combo)
    if c is None:
        raise SystemExit(f"unknown combo {a.combo}; known: {sorted(combos)}")
    if c.get("unsupported"):
        raise SystemExit(f"{a.combo} is not runnable: {c['unsupported']}")

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from adapters import build_runner

    t = time.perf_counter()
    runner = build_runner(c["kind"], c["params"])
    load_s = time.perf_counter() - t
    lock = threading.Lock()
    app = FastAPI(title=f"jevlike {a.combo}")

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.json()
        try:
            with lock:
                t0 = time.perf_counter()
                out = runner.answer(body)
                out.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
            out.setdefault("model", body.get("model", a.combo))
            return JSONResponse(out, headers={"x-typesafe-request-id": request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex})
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, status_code=422)

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": n, "description": f"jevlike {a.combo}", "combo": a.combo, "load_s": round(load_s, 2),
                            "info": runner.info()} for n in ("kev-latest", "jev-latest")]}

    print(f"serving {a.combo} on http://{a.host}:{a.port} (loaded in {load_s:.1f} s)")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
