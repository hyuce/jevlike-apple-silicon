"""Run one heavy step (model load + request, a merge, a conversion) in its own process under a memory/time watchdog.

Swap-death guard (protocol section 8; task brief: abort a load above ~14 GB):
  1. child-tree RSS (psutil, sampled every `sample_s`, default 100 ms) above `rss_limit_gb`  -> SIGKILL the tree
  2. system available memory below `min_avail_gb` AND swap grown by more than `swap_growth_gb` since the step began
     (macOS is paging hard on our behalf)                                                     -> SIGKILL the tree
  3. wall time above `timeout_s`                                                               -> SIGKILL the tree
The worker additionally runs `InProcessSampler`, which watches the framework's own counters (MLX active memory,
torch MPS driver memory) that RSS may not show for Metal buffers, and exits itself (code GUARD_EXIT) above the same
limit after writing what it knows.

GB in this module = 1e9 bytes.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

GB = 1e9
GUARD_EXIT = 86          # exit code of a worker that stopped itself on the memory guard


# ----------------------------------------------------------------------------------------------- physical footprint
# psutil's RSS misses most GPU (Metal) memory on Apple Silicon (warning in jeffclimbsbikes/system-one-playground @304c528f;
# results repo inventory). The process "physical footprint" (what Activity Monitor calls Memory) includes it.
# Read with proc_pid_rusage(pid, RUSAGE_INFO_V4) from libproc (<sys/resource.h>): after the 16-byte ri_uuid come uint64
# fields; index 6 = ri_resident_size, 7 = ri_phys_footprint, 28 = ri_lifetime_max_phys_footprint (V4).
# Layout checked on this project's Mac on 2026-11-13: resident equals psutil RSS exactly, phys_footprint grows by
# 0.50 GB for a 0.5 GB bytearray and drops back after `del`, lifetime_max keeps 0.51 GB.
_RUSAGE_INFO_V4 = 4
_libproc = None


def rusage(pid: int):
    """-> {"phys_footprint", "lifetime_max_phys_footprint", "resident"} in bytes, or None where unavailable."""
    global _libproc
    try:
        import ctypes
        import struct
        if _libproc is None:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            _libproc.proc_pid_rusage.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(1024)            # larger than any rusage_info_vN
        if _libproc.proc_pid_rusage(int(pid), _RUSAGE_INFO_V4, buf) != 0:
            return None
        v = struct.unpack_from("<35Q", buf.raw, 16)
        return {"resident": v[6], "phys_footprint": v[7], "lifetime_max_phys_footprint": v[28]}
    except Exception:  # noqa: BLE001 - not macOS, or libproc refused
        return None


# ----------------------------------------------------------------------------------------------- system snapshots
def _cmd(args, timeout=20):
    """stdout+stderr of a command, or the error text; never raises."""
    if shutil.which(args[0]) is None:
        return f"[{args[0]} not found]"
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip() if r.returncode == 0 else f"[exit {r.returncode}] {out.strip()}"
    except Exception as e:  # noqa: BLE001
        return f"[{type(e).__name__}: {e}]"


def system_snapshot(label: str = "") -> dict:
    """Thermal/power/memory state, logged before and after every step (protocol section 8)."""
    snap = {"label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pmset_therm": _cmd(["pmset", "-g", "therm"]), "pmset_batt": _cmd(["pmset", "-g", "batt"]),
            "memory_pressure": _cmd(["memory_pressure", "-Q"]), "swapusage": _cmd(["sysctl", "vm.swapusage"])}
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), _swap_used()
        snap.update(mem_total_gb=round(vm.total / GB, 3), mem_available_gb=round(vm.available / GB, 3),
                    swap_used_gb=None if sw is None else round(sw / GB, 3), cpu_percent=psutil.cpu_percent(interval=0.2))
    except Exception as e:  # noqa: BLE001
        snap["psutil_error"] = f"{type(e).__name__}: {e}"
    return snap


# ----------------------------------------------------------------------------------------------- parent-side guard
def _tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except Exception:  # noqa: BLE001
        return [proc]


def _tree_rss(proc) -> int:
    total = 0
    for p in _tree(proc):
        try:
            total += p.memory_info().rss
        except Exception:  # noqa: BLE001 - a child that just exited
            pass
    return total


def _tree_footprint(proc):
    """Sum of phys_footprint over the process tree, or None if any member cannot be read."""
    total = 0
    for p in _tree(proc):
        r = rusage(p.pid)
        if r is None:
            if p.is_running():
                return None
            continue
        total += r["phys_footprint"]
    return total


def _swap_used():
    """Swap in use (bytes), or None where psutil cannot read it (seen in sandboxed shells: OSError from sysctl)."""
    try:
        import psutil
        return psutil.swap_memory().used
    except Exception:  # noqa: BLE001
        return None


def _kill_tree(proc):
    for p in reversed(_tree(proc)):
        try:
            p.send_signal(signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def run_guarded(cmd, *, log_path: Path, env=None, cwd=None, rss_limit_gb=14.0, min_avail_gb=0.75,
                swap_growth_gb=2.0, timeout_s=1800.0, sample_s=0.1, echo=True) -> dict:
    """Run `cmd`; stream its stdout/stderr into `log_path` (appending) and, if `echo`, to our stdout.
    -> {returncode, killed_reason, wall_s, peak_rss_gb, min_avail_gb, swap_start_gb, swap_peak_gb, tail}."""
    import psutil

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail, lock = [], threading.Lock()
    logf = open(log_path, "a", encoding="utf-8", errors="replace")

    def pump(stream, tag):
        for line in iter(stream.readline, ""):
            with lock:
                logf.write(f"[{tag}] {line}")
                logf.flush()
                tail.append(f"[{tag}] {line.rstrip()}")
                del tail[:-80]
            if echo:
                sys.stdout.write(f"    {line}")
                sys.stdout.flush()
        stream.close()

    sw0 = _swap_used()          # None: swap unreadable -> the swap condition is disabled and the result says so
    t0 = time.time()
    proc = psutil.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1, errors="replace")
    threads = [threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    peak_rss, peak_fp, fp_ok, min_avail, swap_peak, reason = 0, 0, True, float("inf"), sw0, None
    while proc.poll() is None:
        rss, fp = _tree_rss(proc), _tree_footprint(proc)
        vm, sw = psutil.virtual_memory(), _swap_used()
        peak_rss, min_avail = max(peak_rss, rss), min(min_avail, vm.available)
        if fp is None:
            fp_ok = False
        else:
            peak_fp = max(peak_fp, fp)
        if sw is not None and swap_peak is not None:
            swap_peak = max(swap_peak, sw)
        grown = (sw - sw0) if (sw is not None and sw0 is not None) else None
        elapsed = time.time() - t0
        guard_val, guard_name = (fp, "physical footprint") if fp is not None else (rss, "RSS (footprint unreadable)")
        if guard_val > rss_limit_gb * GB:
            reason = f"memory guard: process {guard_name} {guard_val / GB:.2f} GB > limit {rss_limit_gb:g} GB after {elapsed:.1f} s"
        elif vm.available < min_avail_gb * GB and grown is not None and grown > swap_growth_gb * GB:
            reason = (f"swap guard: system available {vm.available / GB:.2f} GB < {min_avail_gb:g} GB and swap grew "
                      f"{grown / GB:.2f} GB (> {swap_growth_gb:g} GB) after {elapsed:.1f} s")
        elif elapsed > timeout_s:
            reason = f"timeout: {elapsed:.0f} s > {timeout_s:g} s"
        if reason:
            _kill_tree(proc)
            break
        time.sleep(sample_s)
    try:
        rc = proc.wait(timeout=60)
    except Exception:  # noqa: BLE001
        _kill_tree(proc)
        rc = proc.wait()
    for t in threads:
        t.join(timeout=10)
    wall = time.time() - t0
    with lock:
        logf.write(f"[guard] returncode={rc} killed_reason={reason} wall_s={wall:.1f} peak_rss_gb={peak_rss / GB:.3f} "
                   f"peak_phys_footprint_gb={peak_fp / GB:.3f}\n")
        logf.close()
    return {"returncode": rc, "killed_reason": reason, "wall_s": round(wall, 2), "peak_rss_gb": round(peak_rss / GB, 3),
            "peak_phys_footprint_gb": round(peak_fp / GB, 3) if peak_fp else None,
            "guard_metric": "physical footprint" if fp_ok else "RSS (footprint unreadable at least once)",
            "min_avail_gb": None if min_avail == float("inf") else round(min_avail / GB, 3),
            "swap_start_gb": None if sw0 is None else round(sw0 / GB, 3),
            "swap_peak_gb": None if swap_peak is None else round(swap_peak / GB, 3),
            "swap_guard": "active" if sw0 is not None else "disabled (swap usage unreadable)", "tail": list(tail)}


def describe_exit(rc: int) -> str:
    if rc is None:
        return "no exit code"
    if rc < 0:
        try:
            return f"terminated by signal {signal.Signals(-rc).name} ({rc})"
        except ValueError:
            return f"terminated by signal {-rc}"
    if rc == GUARD_EXIT:
        return f"worker stopped itself on the in-process memory guard (exit {GUARD_EXIT})"
    return f"exit code {rc}"


# ----------------------------------------------------------------------------------------------- worker-side sampler
class InProcessSampler:
    """Background thread in the worker: every `sample_s` it reads this process's RSS and, when the frameworks are
    loaded, MLX active memory and torch MPS driver memory. Keeps the peaks. Above `limit_gb` on any of them it calls
    `on_breach(reason)` (the worker writes its partial result) and exits the process with GUARD_EXIT."""

    def __init__(self, limit_gb=14.0, sample_s=0.1, on_breach=None):
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.limit, self.sample_s, self.on_breach = limit_gb * GB, sample_s, on_breach
        self.peak = {"rss": 0, "phys_footprint": 0, "mlx_active": 0, "mps_driver": 0}
        self.lifetime_max = None
        self.samples = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="jevlike-sampler")

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)
        self.read()
        return self.peaks_gb()

    @staticmethod
    def _mlx_active():
        mx = sys.modules.get("mlx.core")
        if mx is None:
            return 0
        f = getattr(mx, "get_active_memory", None) or getattr(getattr(mx, "metal", None), "get_active_memory", None)
        try:
            return int(f()) if f else 0
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _mps_driver():
        torch = sys.modules.get("torch")
        if torch is None:
            return 0
        try:
            if torch.backends.mps.is_available():
                return int(torch.mps.driver_allocated_memory())
        except Exception:  # noqa: BLE001
            pass
        return 0

    def read(self):
        r = rusage(os.getpid())
        vals = {"rss": self.proc.memory_info().rss, "phys_footprint": r["phys_footprint"] if r else 0,
                "mlx_active": self._mlx_active(), "mps_driver": self._mps_driver()}
        for k, v in vals.items():
            self.peak[k] = max(self.peak.get(k, 0), v)
        if r:
            self.lifetime_max = r["lifetime_max_phys_footprint"]
        self.samples += 1
        return vals

    def _run(self):
        while not self._stop.is_set():
            try:
                vals = self.read()
            except Exception:  # noqa: BLE001
                vals = {}
            # guard on the physical footprint (it includes Metal memory) and the framework counters; RSS only when the
            # footprint cannot be read (RSS also counts memory-mapped weight files that are not memory pressure)
            watched = {k: v for k, v in vals.items() if k != "rss" or not vals.get("phys_footprint")}
            over = {k: v for k, v in watched.items() if v > self.limit}
            if over:
                k, v = next(iter(over.items()))
                reason = f"in-process memory guard: {k} {v / GB:.2f} GB > limit {self.limit / GB:g} GB"
                try:
                    if self.on_breach:
                        self.on_breach(reason)
                finally:
                    os._exit(GUARD_EXIT)
            time.sleep(self.sample_s)

    def peaks_gb(self):
        out = {f"{k}_peak_gb": round(v / GB, 3) for k, v in self.peak.items()} | {"sampler_samples": self.samples}
        # the kernel's own running maximum: catches peaks between two 100 ms samples
        out["phys_footprint_lifetime_max_gb"] = round(self.lifetime_max / GB, 3) if self.lifetime_max else None
        return out


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
