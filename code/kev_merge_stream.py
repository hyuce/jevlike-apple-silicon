"""Memory-light merge of a Kev LoRA checkpoint into a full-weight bf16 checkpoint that kev.mlx_model.load_full loads.

Why this exists (OURS, not upstream): Kev's own scripts/merge_lora_checkpoint.py builds the base the way the loader
does, i.e. in the checkpoint's weights dtype -- fp32 for every Kev below 27B -- so merging Kev-4B needs the whole
~4B-parameter backbone in fp32 (~17 GB) and does not fit a 16 GiB Mac. The MLX adapter path merges in memory but
Kev's README reports a 15.9 GB loading peak for Kev-4B. This script streams tensor by tensor instead.

Arithmetic (same as `merge_lora_checkpoint.py --weights_dtype bf16` and kev.mlx_model.merge_lora):
    W_out = round_to_bf16( fp32(W_base) + fp32(B) @ fp32(A) * scaling ),   scaling = lora_alpha / r  (/ sqrt(r) for rsLoRA)
    every other backbone tensor: round_to_bf16(W_base)  (the Qwen3.5 bases keep a few fp32 tensors, e.g. A_log)
Layout (same as kev.full_ft.write_backbone = save_pretrained of transformers' Qwen3_5TextModel):
    tensor names = the base's `model.language_model.*` names without that prefix; vision tower and MTP dropped;
    config.json = the base's text config (model_type qwen3_5_text), architectures [Qwen3_5TextModel], dtype bfloat16,
    use_cache false; head.pt = the LoRA checkpoint's meta and pointer head with lora=0, weights="full",
    weights_dtype="bf16" and a `merged_lora` record; tokenizer files copied from the checkpoint.
Equivalence is not assumed: `compare` checks names, shapes, dtypes and values against the official script's output
(run for Kev-0.8B, where the official script fits in memory), plus config.json and head.pt.

    python kev_merge_stream.py merge   --lora jaredpalmer/kev-4b@<sha> --out <dir>         # -> <dir>/checkpoint
    python kev_merge_stream.py compare <dirA>/checkpoint <dirB>/checkpoint --out report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

BASE_PREFIX = "model.language_model."
SKIP_PREFIXES = ("model.visual.", "mtp.", "lm_head.")
# files of the LoRA run that are rewritten or are records of that run (as scripts/merge_lora_checkpoint.py NOT_COPIED)
NOT_COPIED = {"head.pt", "adapter_config.json", "adapter_model.safetensors", "training_config.json", "training_metrics.json",
              "provenance.json", "result.json", "train.log", "README.md", ".gitattributes"}
FORMULA = "round_to_bf16(fp32(W) + fp32(B) @ fp32(A) * scaling) for adapted weights; round_to_bf16(W) elsewhere"


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def base_tensor_index(base_dir: Path) -> dict:
    """name -> shard file of the base snapshot."""
    idx = base_dir / "model.safetensors.index.json"
    if idx.exists():
        return {k: base_dir / v for k, v in json.loads(idx.read_text())["weight_map"].items()}
    from safetensors import safe_open
    out = {}
    for shard in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            out.update({k: shard for k in f.keys()})
    return out


def merge(lora: str, out: Path, shard_bytes: float = 2e9, log=print) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from kev.checkpoint import Checkpoint, resolve_run, write_meta

    t0 = time.time()
    out = Path(out)
    target, partial = out / "checkpoint", out / "checkpoint.partial"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    ck = Checkpoint(lora)
    if ck.full:
        raise ValueError(f"{lora} is already a full-weight checkpoint")
    meta, acfg = ck.meta, ck.adapter_config()
    for key in ("use_dora", "modules_to_save", "trainable_token_indices", "alpha_pattern", "rank_pattern"):
        if acfg.get(key):
            raise ValueError(f"adapter_config.json sets {key}={acfg[key]!r}; this streaming merge handles plain LoRA only")
    if acfg.get("bias", "none") != "none":
        raise ValueError("LoRA biases are not supported")
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scaling = alpha / (math.sqrt(r) if acfg.get("use_rslora") else r)
    # a local base directory is used as is (kev.model.DecisionModel does the same through from_pretrained)
    base_dir = Path(meta.base) if os.path.isdir(meta.base) else Path(resolve_run(f"{meta.base}@{meta.base_revision or ''}"))
    log(f"base {meta.base}@{meta.base_revision} -> {base_dir}; adapter r={r} alpha={alpha} scaling={scaling}")

    index = base_tensor_index(base_dir)
    names = sorted(k for k in index if k.startswith(BASE_PREFIX))
    unknown = sorted(k for k in index if not k.startswith(BASE_PREFIX) and not k.startswith(SKIP_PREFIXES))
    if unknown:
        raise ValueError(f"base tensors outside {BASE_PREFIX!r} and the skipped prefixes: {unknown[:5]}")
    text_names = {k[len(BASE_PREFIX):] for k in names}

    adapter_path = Path(ck.path) / "adapter_model.safetensors"
    pairs = {}
    with safe_open(str(adapter_path), "pt") as fa:
        keys = list(fa.keys())
        for k in keys:
            if k.endswith(".lora_A.weight"):
                stem = k[: -len(".lora_A.weight")]
                if not stem.startswith("base_model.model."):
                    raise ValueError(f"unexpected adapter tensor name {k}")
                tgt = stem[len("base_model.model."):] + ".weight"
                if tgt not in text_names:
                    raise ValueError(f"adapter tensor {stem} has no base weight {BASE_PREFIX}{tgt}")
                pairs[tgt] = (fa.get_tensor(k).float(), fa.get_tensor(stem + ".lora_B.weight").float())
        stray = [k for k in keys if ".lora_A." not in k and ".lora_B." not in k]
        if stray:
            raise ValueError(f"adapter carries non-LoRA tensors {stray[:3]}; not supported")
    log(f"{len(names)} backbone tensors, {len(pairs)} adapted")

    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    shard, shard_size, shard_files, weight_map, total = {}, 0, [], {}, 0
    handles = {}

    def flush():
        nonlocal shard, shard_size
        if not shard:
            return
        tmp = partial / f"shard-{len(shard_files):05d}.tmp.safetensors"
        save_file(shard, str(tmp), metadata={"format": "pt"})
        shard_files.append((tmp, list(shard)))
        shard, shard_size = {}, 0

    adapted = 0
    with torch.no_grad():
        for name in names:
            path = index[name]
            if path not in handles:
                handles[path] = safe_open(str(path), "pt")
            w = handles[path].get_tensor(name)
            out_name = name[len(BASE_PREFIX):]
            if out_name in pairs:
                a, b = pairs.pop(out_name)
                w = (w.float() + (b @ a) * scaling).to(torch.bfloat16)
                adapted += 1
            else:
                w = w.to(torch.bfloat16)
            w = w.contiguous()
            shard[out_name] = w
            shard_size += w.numel() * w.element_size()
            total += w.numel() * w.element_size()
            if shard_size >= shard_bytes:
                flush()
        flush()
    handles.clear()
    if pairs:
        raise RuntimeError(f"{len(pairs)} adapter targets were never met: {list(pairs)[:3]}")

    n = len(shard_files)
    if n == 1:
        os.replace(shard_files[0][0], partial / "model.safetensors")
        weight_map = None
    else:
        weight_map = {}
        for i, (tmp, keys) in enumerate(shard_files, 1):
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp, partial / final)
            weight_map.update({k: final for k in keys})
        (partial / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8")

    cfg = AutoConfig.from_pretrained(str(base_dir)).get_text_config()
    cfg.architectures = ["Qwen3_5TextModel"]
    cfg.use_cache = False
    cfg.dtype = torch.bfloat16
    cfg.save_pretrained(str(partial))

    for p in Path(ck.path).iterdir():
        if p.is_file() and p.name not in NOT_COPIED:
            shutil.copy2(p, partial / p.name)
    source = {"path": lora, "resolved": ck.path, "adapter_sha256": sha256_file(adapter_path),
              "head_sha256": sha256_file(Path(ck.path) / "head.pt")}
    trained_dtype = meta.weights_dtype
    meta.lora, meta.weights, meta.weights_dtype = 0, "full", "bf16"
    meta.extra["merged_lora"] = {"source": source, "base": f"{meta.base}@{meta.base_revision}", "adapted_tensors": adapted,
                                 "formula": FORMULA, "scaling": scaling, "tool": "jevlike code/kev_merge_stream.py (streaming)",
                                 "weights_dtype": {"trained": trained_dtype, "written": "bf16"}}
    write_meta(str(partial), meta)
    check = Checkpoint(str(partial))
    if not check.full or check.saved_dtype() != "bfloat16":
        raise RuntimeError(f"{partial} does not load as a bf16 full-weight checkpoint")
    partial.rename(target)
    report = {"checkpoint": str(target), "tensors": len(names), "adapted_tensors": adapted, "bytes": total,
              "shards": {p.name: sha256_file(p) for p in sorted(target.glob("model*.safetensors"))},
              "head_sha256": sha256_file(target / "head.pt"), "source": source, "formula": FORMULA,
              "seconds": round(time.time() - t0, 1)}
    (out / "merge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"-> {target}: {len(names)} tensors ({adapted} adapted), {total / 1e9:.2f} GB, {report['seconds']} s")
    return report


def _tensor_files(d: Path) -> dict:
    from safetensors import safe_open
    out = {}
    for shard in sorted(Path(d).glob("model*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                if k in out:
                    raise ValueError(f"{k} in two shards of {d}")
                out[k] = shard
    return out


def compare(a: Path, b: Path) -> dict:
    """Tensor-by-tensor comparison of two full-weight checkpoints (+ config.json and head.pt)."""
    import torch
    from safetensors import safe_open

    a, b = Path(a), Path(b)
    fa, fb = _tensor_files(a), _tensor_files(b)
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    rows, handles = [], {}

    def get(files, k):
        p = files[k]
        if p not in handles:
            handles[p] = safe_open(str(p), "pt")
        return handles[p].get_tensor(k)

    for k in sorted(set(fa) & set(fb)):
        x, y = get(fa, k), get(fb, k)
        row = {"name": k, "shape_a": list(x.shape), "shape_b": list(y.shape), "dtype_a": str(x.dtype), "dtype_b": str(y.dtype)}
        if x.shape == y.shape:
            d = (x.float() - y.float()).abs()
            row.update(n_diff=int((d > 0).sum()), max_abs_diff=float(d.max()) if d.numel() else 0.0, numel=x.numel(),
                       bitwise_equal=bool(torch.equal(x, y)) if x.dtype == y.dtype else False)
        rows.append(row)
    handles.clear()
    ca = json.loads((a / "config.json").read_text())
    cb = json.loads((b / "config.json").read_text())
    ignore = {"transformers_version"}
    cfg_diff = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb)) if k not in ignore and ca.get(k) != cb.get(k)}
    ha = torch.load(a / "head.pt", map_location="cpu")
    hb = torch.load(b / "head.pt", map_location="cpu")
    head_equal = all(torch.equal(ha["head"][k], hb["head"][k]) for k in ha["head"]) and set(ha["head"]) == set(hb["head"])
    meta_keys = ("base", "base_revision", "lora", "head_dim", "weights", "weights_dtype", "temperature", "option_isolation")
    meta_diff = {k: [ha.get(k), hb.get(k)] for k in meta_keys if ha.get(k) != hb.get(k)}
    n_bitwise = sum(r.get("bitwise_equal", False) for r in rows)
    return {"a": str(a), "b": str(b), "tensors_a": len(fa), "tensors_b": len(fb), "only_in_a": only_a, "only_in_b": only_b,
            "compared": len(rows), "bitwise_equal_tensors": n_bitwise,
            "shape_or_dtype_mismatch": [r["name"] for r in rows if r["shape_a"] != r["shape_b"] or r["dtype_a"] != r["dtype_b"]],
            "max_abs_diff_overall": max((r.get("max_abs_diff", 0.0) for r in rows), default=None),
            "elements_differing": sum(r.get("n_diff", 0) for r in rows), "elements_total": sum(r.get("numel", 0) for r in rows),
            "differing_tensors": [{k: r[k] for k in ("name", "n_diff", "max_abs_diff")} for r in rows if r.get("n_diff")][:50],
            "config_diff": cfg_diff, "pointer_head_equal": head_equal, "meta_diff": meta_diff,
            "verdict": "identical" if (not only_a and not only_b and n_bitwise == len(rows) and not cfg_diff and head_equal)
                       else "differs (see fields)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--lora", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--shard_gb", type=float, default=2.0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.lora, Path(a.out), shard_bytes=a.shard_gb * 1e9)
    else:
        rep = compare(Path(a.a), Path(a.b))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("verdict", "compared", "bitwise_equal_tensors", "max_abs_diff_overall",
                                               "elements_differing", "config_diff", "pointer_head_equal", "meta_diff")}, default=str))


if __name__ == "__main__":
    main()
