"""Write results/environment.md: hardware, OS, package versions, repo SHAs, MLX/Metal and torch MPS status.

Run with the study venv's python (setup_mac.sh does). Serial number, hardware UUID and provisioning UDID lines of
system_profiler are dropped (not needed, and identifying). Every value is read from this machine at run time.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pins  # noqa: E402

PRIVATE = ("Serial Number", "Hardware UUID", "Provisioning UDID", "UDID")
PACKAGES = ["mlx", "mlx-metal", "mlx-lm", "torch", "transformers", "accelerate", "safetensors", "huggingface-hub", "tokenizers",
            "peft", "numpy", "pandas", "scipy", "psutil", "fastapi", "uvicorn", "pydantic", "typesafe-sdk", "kev", "laya",
            "vegaml", "anyjev", "jevmlx"]


def sh(args, timeout=60):
    if shutil.which(args[0]) is None:
        return f"[{args[0]} not found]"
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "")).strip() + ("" if r.returncode == 0 else f"\n[exit {r.returncode}]")
    except Exception as e:  # noqa: BLE001
        return f"[{type(e).__name__}: {e}]"


def scrub(text):
    return "\n".join(l for l in text.splitlines() if not any(p in l for p in PRIVATE))


def mlx_status():
    out = {}
    try:
        import mlx.core as mx
        out["metal_available"] = mx.metal.is_available()
        info = getattr(mx, "device_info", None) or mx.metal.device_info
        out["device_info"] = info()
        out["default_device"] = str(mx.default_device())
        a = mx.ones((512, 512))
        t = time.perf_counter()
        mx.eval(a @ a)
        out["matmul_512_ok_ms"] = round((time.perf_counter() - t) * 1000, 2)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def torch_status():
    out = {}
    try:
        import torch
        out["mps_built"] = torch.backends.mps.is_built()
        out["mps_available"] = torch.backends.mps.is_available()
        if out["mps_available"]:
            a = torch.ones(512, 512, device="mps")
            (a @ a).sum().item()
            out["mps_matmul_ok"] = True
            out["mps_recommended_max_memory_gb"] = round(torch.mps.recommended_max_memory() / 1e9, 3)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pins.results_dir() / "environment.md"))
    a = ap.parse_args()
    cache = pins.cache_dir()
    L = [f"# Environment record", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} by `code/env_report.py` on this machine. "
         "Serial/UUID/UDID lines removed.", ""]
    L += ["## Hardware", "", "```", scrub(sh(["system_profiler", "SPHardwareDataType"])), "```", "",
          "### GPU (system_profiler SPDisplaysDataType)", "", "```", scrub(sh(["system_profiler", "SPDisplaysDataType"])), "```", "",
          "### sysctl", "", "```"]
    for k in ("hw.memsize", "hw.ncpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu", "machdep.cpu.brand_string", "vm.swapusage"):
        L.append(f"{k}: {sh(['sysctl', '-n', k])}")
    L += ["```", "", "## Operating system", "", "```", sh(["sw_vers"]), platform.platform(), "```", "",
          "## Power / thermal at recording time", "", "```", sh(["pmset", "-g", "therm"]), sh(["pmset", "-g", "batt"]), "```", "",
          "## Python and packages", "", f"- python: `{sys.version.split()[0]}` at `{sys.executable}`",
          f"- uv: `{sh(['uv', '--version'])}`", f"- git: `{sh(['git', '--version'])}`"]
    for p in PACKAGES:
        try:
            L.append(f"- {p}: `{md.version(p)}`")
        except md.PackageNotFoundError:
            L.append(f"- {p}: not installed")
    L += ["", "## MLX / Metal", "", "```", repr(mlx_status()), "```", "", "## torch MPS", "", "```", repr(torch_status()), "```", "",
          "## Repositories (clones in the local cache, not in Google Drive)", "", "| repo | URL | pinned SHA | checked-out SHA |",
          "|---|---|---|---|"]
    for name, (url, sha) in pins.REPOS.items():
        got = sh(["git", "-C", str(pins.repo_dir(name)), "rev-parse", "HEAD"]).splitlines()[0]
        L.append(f"| {name} | {url} | `{sha}` | `{got}` |")
    L += ["", "## Hugging Face revisions pinned for the smoke test", "", "| key | repo | commit | note |", "|---|---|---|---|"]
    for k, (repo, sha, note) in pins.HF.items():
        L.append(f"| {k} | {repo} | `{sha}` | {note} |")
    L += ["", "## Paths", "", f"- cache (weights, clones, venv): `{cache}`", f"- HF_HOME: `{cache / 'hf'}`",
          f"- disk free at cache: {shutil.disk_usage(cache if cache.exists() else Path.home()).free / 1e9:.1f} GB", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
