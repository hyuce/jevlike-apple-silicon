"""Run a script from Kev's scripts/ directory with `scripts` bound to that directory.

Usage: python run_kev_script.py <kev_repo_dir> <script_name.py> [script args...]

Why: Kev's scripts/ has no __init__.py, so `from scripts.interpolate_checkpoint import ...` relies on a namespace
package. In the shared benchmark venv another installed distribution ships a regular top-level `scripts` package,
which wins over a namespace package regardless of sys.path order. Both `python scripts/merge_lora_checkpoint.py`
and `python -m scripts.merge_lora_checkpoint` then fail (2026-11-13 runs). Binding sys.modules['scripts'] to Kev's
directory before the script runs removes the ambiguity without modifying Kev's code.
"""
import importlib.util
import os
import runpy
import sys
import types


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    kev, script, args = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3:]
    try:
        spec = importlib.util.find_spec("scripts")
        where = (getattr(spec, "origin", None), list(getattr(spec, "submodule_search_locations", None) or []))
    except Exception as e:  # diagnostics only
        where = f"{type(e).__name__}: {e}"
    print(f"[run_kev_script] pre-existing 'scripts' resolution: {where}", file=sys.stderr, flush=True)
    sys.path.insert(0, kev)
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [os.path.join(kev, "scripts")]
    sys.modules["scripts"] = pkg
    path = os.path.join(kev, "scripts", script)
    sys.argv = [path] + args
    os.chdir(kev)
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
