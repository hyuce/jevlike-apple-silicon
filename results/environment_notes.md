# Environment notes: launching each candidate for the benchmark phase

Status (2026-11-13): written BEFORE the smoke test ran. Which combinations are runnable on this 16 GB M2 Pro is
decided only by `feasibility_matrix.csv` (`runs=yes`) after `code/run_smoke.sh` has run on the Mac. The sandbox
that produced these scripts has no Metal device, so no model was loaded there. Everything below uses
`~/jevlike_cache/venv/bin/python` (from `code/setup_mac.sh`). Run it from `code/` with `HF_HOME=~/jevlike_cache/hf`.

## One endpoint shape for every candidate

`code/serve_systemone.py --combo <id> --port <p>` loads one smoke-test combination (ids as in
`feasibility_matrix_detail.csv`, column `combo_id`). It serves `POST /v1/systemone` (TypeSafe body) and `GET /v1/models`.
It answers one request at a time, through the same `adapters.py` runner the smoke test validated.
Any System One client can then score it, including Kev's own benchmark:

```bash
cd ~/jevlike_cache/repos/kev
~/jevlike_cache/venv/bin/python -m kev.benchmark --remote http://127.0.0.1:8010 --suite evals/v4/transfer-v4 --out runs/<name>
```

In-process alternative (no HTTP):
`from adapters import build_runner; from combos import smoke_combos`.
Take `c = {x["id"]: x for x in smoke_combos()}["<id>"]`, build `r = build_runner(c["kind"], c["params"])`
and call `r.answer(request_dict)`.

## Per family

| family | combo ids | launch | notes |
|---|---|---|---|
| Kev (MLX, bf16) | `kev-0.8b/bf16`, `kev-4b/adapter-bf16` | upstream: `python -m kev.serve --run jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e --port 8009` | kev.serve default on Apple Silicon: MLX, LoRA merged in memory |
| Kev (MLX, full-weight bf16) | `kev-0.8b/full-bf16-merged`, `kev-4b/full-bf16-merged` | upstream: `python -m kev.serve --run ~/jevlike_cache/models/merged/kev-4b-stream/checkpoint --port 8009` | checkpoint written by `code/kev_merge_stream.py`; loaded by `kev.mlx_model.load_full` |
| Kev (fp32 reference) | `kev-0.8b/fp32`, `kev-0.8b/fp32-cpu` | upstream: `KEV_DTYPE=fp32 python -m kev.serve --run ... [--device cpu]` | the exact path of Kev's published numbers (torch) |
| Kev 8/4-bit | `kev-*/8bit`, `kev-*/4bit` | `serve_systemone.py --combo kev-4b/4bit` | ours: `mlx.nn.quantize` (affine, group 64) on the merged backbone; pointer head fp32 |
| Laya (torch) | `laya/fp32`, `laya/fp16`, `laya/fp32-cpu`, `laya-multilingual/*` | `serve_systemone.py --combo laya/fp32` (upstream `laya-serve` also exists) | fp16 = Laya's MPS autocast forced from 1 row |
| laya-mlx | `laya/fp16-laya-mlx`, `laya-multilingual/fp16-laya-mlx` | `serve_systemone.py --combo laya/fp16-laya-mlx` (upstream `laya-mlx-serve` also exists) | independent FP16 MLX port |
| NanoJev | `nanojev/bf16`, `nanojev/bf16-cpu` | `serve_systemone.py --combo nanojev/bf16` | our device port; NanoJev documents CUDA only; game-control model (protocol §4: accuracy table only after a recorded decision) |
| AnyJev (training-free, torch MPS) | `anyjev-q0.8b/bf16`, `anyjev-q4b/bf16` | `serve_systemone.py --combo anyjev-q4b/bf16` | Decider L0; its batch prior needs >= 8 states, so for the benchmark consider `Decider.decide_batch` per question (open issue) |
| jevmlx (training-free, MLX) | `jevmlx-q*/bf16`, `jevmlx-q*/4bit` (fallback `jevmlx-default/4bit`) | `serve_systemone.py --combo jevmlx-q4b/4bit` (upstream `jevmlx serve` also exists) | its /v1/systemone mapping passes choice names only, not descriptions |

## Measurement conventions carried into the benchmark

- Memory: report the process **physical footprint** (`proc_pid_rusage` ri_phys_footprint / lifetime max), plus
  `mx.get_peak_memory()` or `torch.mps.driver_allocated_memory()`. RSS is reported separately: on Apple Silicon it
  misses Metal memory and counts memory-mapped weight files. GB = 1e9 bytes.
- One GPU job at a time; log `pmset -g therm` and `pmset -g batt` before and after each run (`code/guard.py: system_snapshot`).
- Pinned sources: `code/pins.py` (repo SHAs and Hugging Face commits). Actual checkouts are recorded in `environment.md`.
