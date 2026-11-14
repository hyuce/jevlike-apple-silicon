"""The protocol section 4 matrix: preparation tasks (downloads, merges, conversions) and smoke-test combinations.

Each combination names the runner (adapters.py), its parameters, the preparation tasks it needs, and the repo /
Hub revisions it rests on. Combinations that are unsupported by every available code path are listed too (with the
reason) so they appear in feasibility_matrix.csv as runs=no instead of being dropped (protocol section 4).

Training-free base (`tf_base`): "base" (default) = the same Qwen3.5-*-Base checkpoints Kev is built on (protocol RQ3:
"aynı Qwen tabanı"); "instruct" = the post-trained Qwen3.5-0.8B / 4B (extra ~11 GB download).

Special keys: `unsupported` (row written as runs=no, nothing run); `fallback_for` (run only if none of those combos
ran; otherwise no row); `on_demand_prepare` (smoke_test.py runs the missing prepare tasks itself, online, first).
Prepare tasks with `on_demand` are skipped by a plain `prepare_models.py` run.
"""
from __future__ import annotations

from pins import (HF, KEV_PATTERNS, LAYA_MLX_PATTERNS, LAYA_PATTERNS, NANOJEV_PATTERNS, QUANT_GROUP_SIZE, hub_id,
                  models_dir, repo_dir)

TF = {
    "base": {"0.8b": ("qwen3.5-0.8b-base", "Qwen3.5-0.8B-Base"), "4b": ("qwen3.5-4b-base", "Qwen3.5-4B-Base")},
    "instruct": {"0.8b": ("qwen3.5-0.8b", "Qwen3.5-0.8B"), "4b": ("qwen3.5-4b", "Qwen3.5-4B")},
}


def _rev(key):
    repo, sha, note = HF[key]
    return f"{repo}@{sha}" + (f" ({note})" if note and note != "optional" else "")


def prepare_tasks(tf_base="base"):
    """Ordered list of {id, kind, ...}. kinds: download | kev_official_merge | kev_stream_merge | merge_compare |
    jevmlx_link | mlx_convert. `gb` is the approximate disk the task adds (for the free-space check)."""
    m = models_dir()
    t = [
        {"id": "dl-kev-0.8b", "kind": "download", "key": "kev-0.8b", "patterns": KEV_PATTERNS, "gb": 0.1, "kev_meta": True},
        {"id": "dl-kev-4b", "kind": "download", "key": "kev-4b", "patterns": KEV_PATTERNS, "gb": 0.2, "kev_meta": True},
        {"id": "dl-qwen3.5-0.8b-base", "kind": "download", "key": "qwen3.5-0.8b-base", "patterns": KEV_PATTERNS, "gb": 1.8},
        {"id": "dl-qwen3.5-4b-base", "kind": "download", "key": "qwen3.5-4b-base", "patterns": KEV_PATTERNS, "gb": 9.4},
        {"id": "dl-laya", "kind": "download", "key": "laya", "patterns": LAYA_PATTERNS, "gb": 0.9},
        {"id": "dl-laya-multilingual", "kind": "download", "key": "laya-multilingual", "patterns": LAYA_PATTERNS, "gb": 0.7},
        {"id": "dl-nanojev", "kind": "download", "key": "nanojev", "patterns": NANOJEV_PATTERNS, "gb": 2.4,
         "local_dir": str(m / "nanojev" / "unified-games-v1")},
        {"id": "dl-laya-mlx", "kind": "download", "key": "laya-mlx", "patterns": LAYA_MLX_PATTERNS, "gb": 0.9},
        {"id": "dl-laya-multilingual-mlx", "kind": "download", "key": "laya-multilingual-mlx", "patterns": LAYA_MLX_PATTERNS,
         "gb": 0.7},
        # on demand only (smoke_test.py runs these if jevmlx cannot run any Qwen3.5 checkpoint)
        {"id": "dl-jevmlx-default", "kind": "download", "key": "jevmlx-default", "patterns": KEV_PATTERNS, "gb": 4.3,
         "on_demand": True},
        {"id": "link-jevmlx-default", "kind": "jevmlx_link", "key": "jevmlx-default", "link": str(m / "jevmlx" / "Qwen2.5-7B-Instruct-4bit"),
         "needs": ["dl-jevmlx-default"], "gb": 0.0, "on_demand": True},
    ]
    if tf_base == "instruct":
        t += [{"id": "dl-qwen3.5-0.8b", "kind": "download", "key": "qwen3.5-0.8b", "patterns": KEV_PATTERNS, "gb": 1.8},
              {"id": "dl-qwen3.5-4b", "kind": "download", "key": "qwen3.5-4b", "patterns": KEV_PATTERNS, "gb": 9.4}]
    t += [
        # Kev-0.8B both ways: the official script (fits for 0.8B) and our streaming merge, then a tensor-level comparison
        {"id": "merge-kev-0.8b-official", "kind": "kev_official_merge", "lora": hub_id("kev-0.8b"),
         "out": str(m / "merged" / "kev-0.8b-official"), "needs": ["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"], "gb": 1.8},
        {"id": "merge-kev-0.8b-stream", "kind": "kev_stream_merge", "lora": hub_id("kev-0.8b"),
         "out": str(m / "merged" / "kev-0.8b-stream"), "needs": ["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"], "gb": 1.8},
        {"id": "compare-kev-0.8b-merges", "kind": "merge_compare", "a": str(m / "merged" / "kev-0.8b-official" / "checkpoint"),
         "b": str(m / "merged" / "kev-0.8b-stream" / "checkpoint"),
         "needs": ["merge-kev-0.8b-official", "merge-kev-0.8b-stream"], "gb": 0.0},
        # Kev-4B: only the streaming merge fits a 16 GiB Mac (official script: fp32 backbone)
        {"id": "merge-kev-4b-stream", "kind": "kev_stream_merge", "lora": hub_id("kev-4b"),
         "out": str(m / "merged" / "kev-4b-stream"), "needs": ["dl-kev-4b", "dl-qwen3.5-4b-base"], "gb": 8.6},
    ]
    for size in ("0.8b", "4b"):
        key, name = TF[tf_base][size]
        t += [{"id": f"link-jevmlx-{name}", "kind": "jevmlx_link", "key": key, "link": str(m / "jevmlx" / name),
               "needs": [f"dl-{key}"], "gb": 0.0},
              {"id": f"convert-q4-{name}", "kind": "mlx_convert", "src": str(m / "jevmlx" / name),
               "dst": str(m / "jevmlx" / f"{name}-4bit-g{QUANT_GROUP_SIZE}"), "bits": 4, "group_size": QUANT_GROUP_SIZE,
               "needs": [f"link-jevmlx-{name}"], "gb": 0.7 if size == "0.8b" else 2.6}]
    return t


def smoke_combos(tf_base="base"):
    m = models_dir()
    kev08, kev4 = hub_id("kev-0.8b"), hub_id("kev-4b")
    kev08_rev = f"{_rev('kev-0.8b')}; base {_rev('qwen3.5-0.8b-base')}"
    kev4_rev = f"{_rev('kev-4b')}; base {_rev('qwen3.5-4b-base')}"
    merged08 = str(m / "merged" / "kev-0.8b-stream" / "checkpoint")
    merged4 = str(m / "merged" / "kev-4b-stream" / "checkpoint")
    gs = QUANT_GROUP_SIZE
    qnote = (f"OURS: mlx.nn.quantize(affine, group_size={gs}) on the merged bf16 Qwen3.5 backbone with mlx-lm's predicate; "
             "pointer head kept fp32 (torch); Kev has no native quantized path")
    C = [
        # ---------------- Kev-0.8B
        dict(id="kev-0.8b/fp32", model="kev-0.8b", variant="fp32", backend="torch-mps", kind="kev",
             params=dict(run=kev08, device="mps", backend="torch", dtype="fp32"), needs=["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"],
             repo="kev", hf=kev08_rev, notes="reference precision; KEV_DTYPE=fp32 equivalent (torch path; MPS has no DeltaNet kernels -> torch reference code)"),
        dict(id="kev-0.8b/fp32-cpu", model="kev-0.8b", variant="fp32 (CPU)", backend="torch-cpu", kind="kev",
             params=dict(run=kev08, device="cpu", backend="torch", dtype="fp32"), needs=["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"],
             repo="kev", hf=kev08_rev, notes="extra fp32 reference on CPU (the exact path Kev's published numbers use)"),
        dict(id="kev-0.8b/bf16", model="kev-0.8b", variant="bf16", backend="mlx", kind="kev",
             params=dict(run=kev08, device="mps", backend="mlx"), needs=["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"],
             repo="kev", hf=kev08_rev, notes="kev.serve default on Apple Silicon: LoRA merged in memory (kev.mlx_model.merge_lora), bf16"),
        dict(id="kev-0.8b/8bit", model="kev-0.8b", variant="8-bit", backend="mlx", kind="kev",
             params=dict(run=kev08, device="mps", backend="mlx", quant_bits=8, group_size=gs),
             needs=["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"], repo="kev", hf=kev08_rev, notes=qnote),
        dict(id="kev-0.8b/4bit", model="kev-0.8b", variant="4-bit", backend="mlx", kind="kev",
             params=dict(run=kev08, device="mps", backend="mlx", quant_bits=4, group_size=gs),
             needs=["dl-kev-0.8b", "dl-qwen3.5-0.8b-base"], repo="kev", hf=kev08_rev, notes=qnote),
        dict(id="kev-0.8b/full-bf16-merged", model="kev-0.8b", variant="full-bf16-merged", backend="mlx", kind="kev",
             params=dict(run=merged08, device="mps", backend="mlx"), needs=["merge-kev-0.8b-stream"],
             repo="kev", hf=kev08_rev + " -> merged locally (kev_merge_stream.py)",
             notes="parity check of our streaming merge: kev.mlx_model.load_full path; compare with kev-0.8b/bf16"),
        # ---------------- Kev-4B (adapter boundary test runs LAST, see order below)
        dict(id="kev-4b/full-bf16-merged", model="kev-4b", variant="full-bf16-merged", backend="mlx", kind="kev",
             params=dict(run=merged4, device="mps", backend="mlx"), needs=["merge-kev-4b-stream"],
             repo="kev", hf=kev4_rev + " -> merged locally (kev_merge_stream.py)",
             notes="full-weight bf16 path (kev.mlx_model.load_full); source reports 8.4 GB peak on a 32 GB M5"),
        dict(id="kev-4b/8bit", model="kev-4b", variant="8-bit", backend="mlx", kind="kev",
             params=dict(run=merged4, device="mps", backend="mlx", quant_bits=8, group_size=gs), needs=["merge-kev-4b-stream"],
             repo="kev", hf=kev4_rev + " -> merged locally", notes=qnote + "; loaded via load_full then quantized"),
        dict(id="kev-4b/4bit", model="kev-4b", variant="4-bit", backend="mlx", kind="kev",
             params=dict(run=merged4, device="mps", backend="mlx", quant_bits=4, group_size=gs), needs=["merge-kev-4b-stream"],
             repo="kev", hf=kev4_rev + " -> merged locally", notes=qnote + "; loaded via load_full then quantized"),
        # ---------------- Laya
        dict(id="laya/fp32", model="laya", variant="fp32", backend="torch-mps", kind="laya",
             params=dict(model_id=HF["laya"][0], revision=HF["laya"][1], device="mps", precision="fp32"), needs=["dl-laya"],
             repo="laya", hf=_rev("laya"), notes="ModernBERT-large checkpoint; Laya autocast off"),
        dict(id="laya/fp16", model="laya", variant="fp16 (autocast)", backend="torch-mps", kind="laya",
             params=dict(model_id=HF["laya"][0], revision=HF["laya"][1], device="mps", precision="fp16"), needs=["dl-laya"],
             repo="laya", hf=_rev("laya"), notes="fp32 weights, MPS fp16 autocast forced from 1 row (LAYA_MPS_AMP_MIN_ROWS=1)"),
        dict(id="laya/fp32-cpu", model="laya", variant="fp32 (CPU)", backend="torch-cpu", kind="laya",
             params=dict(model_id=HF["laya"][0], revision=HF["laya"][1], device="cpu", precision="fp32"), needs=["dl-laya"],
             repo="laya", hf=_rev("laya"), notes="CPU fallback reference"),
        dict(id="laya-multilingual/fp32", model="laya-multilingual", variant="fp32", backend="torch-mps", kind="laya",
             params=dict(model_id=HF["laya-multilingual"][0], revision=HF["laya-multilingual"][1], device="mps", precision="fp32"),
             needs=["dl-laya-multilingual"], repo="laya", hf=_rev("laya-multilingual"), notes="mmBERT-base checkpoint; autocast off"),
        dict(id="laya-multilingual/fp16", model="laya-multilingual", variant="fp16 (autocast)", backend="torch-mps", kind="laya",
             params=dict(model_id=HF["laya-multilingual"][0], revision=HF["laya-multilingual"][1], device="mps", precision="fp16"),
             needs=["dl-laya-multilingual"], repo="laya", hf=_rev("laya-multilingual"), notes="fp16 autocast forced from 1 row"),
        # ---------------- laya-mlx (independent FP16 MLX port), a separate variant of the same two checkpoints
        dict(id="laya/fp16-laya-mlx", model="laya", variant="fp16 (laya-mlx port)", backend="mlx", kind="layamlx",
             params=dict(model_id=HF["laya-mlx"][0], revision=HF["laya-mlx"][1]), needs=["dl-laya-mlx"],
             repo="laya-mlx", hf=_rev("laya-mlx"), notes="mizorewww/laya-mlx runtime; port of convaiinnovations/laya"),
        dict(id="laya-multilingual/fp16-laya-mlx", model="laya-multilingual", variant="fp16 (laya-mlx port)", backend="mlx",
             kind="layamlx", params=dict(model_id=HF["laya-multilingual-mlx"][0], revision=HF["laya-multilingual-mlx"][1]),
             needs=["dl-laya-multilingual-mlx"], repo="laya-mlx", hf=_rev("laya-multilingual-mlx"),
             notes="mizorewww/laya-mlx runtime; port of convaiinnovations/laya-multilingual"),
        # ---------------- NanoJev (documents CUDA only: torch 2.14 / transformers 5.17 on an A100)
        dict(id="nanojev/bf16-upstream", model="nanojev", variant="bf16 (upstream entry point)", backend="torch-cuda-only", kind="nanojev",
             params=dict(checkpoint_dir=str(m / "nanojev" / "unified-games-v1"), repo=None, mode="upstream", precision="bf16"),
             needs=["dl-nanojev"], repo="nanojev", hf=_rev("nanojev"),
             notes="upstream DecisionPredictor as shipped (device cuda:0); expected to refuse a Mac"),
        dict(id="nanojev/bf16", model="nanojev", variant="bf16", backend="torch-mps", kind="nanojev",
             params=dict(checkpoint_dir=str(m / "nanojev" / "unified-games-v1"), repo=None, mode="port", precision="bf16",
                         device="mps"),
             needs=["dl-nanojev"], repo="nanojev", hf=_rev("nanojev"),
             notes="OUR device=mps port of DecisionPredictor (fp32 storage, bf16 autocast as upstream); untested by the author; "
                   "temperature 1.0 (none shipped); trained for game control, not general classification (protocol section 4); "
                   "run under kev's torch 2.8 / transformers 5.17, not the author's torch 2.14"),
        dict(id="nanojev/bf16-cpu", model="nanojev", variant="bf16 (CPU)", backend="torch-cpu", kind="nanojev",
             params=dict(checkpoint_dir=str(m / "nanojev" / "unified-games-v1"), repo=None, mode="port", precision="bf16",
                         device="cpu"),
             needs=["dl-nanojev"], repo="nanojev", hf=_rev("nanojev"),
             notes="OUR device=cpu port (bf16 autocast on CPU); fallback if MPS fails"),
    ]
    # ---------------- training-free on the same Qwen3.5 checkpoints
    for size in ("0.8b", "4b"):
        key, name = TF[tf_base][size]
        repo_id, sha, _ = HF[key]
        C += [
            dict(id=f"anyjev-q{size}/bf16", model=f"anyjev-q{size}", variant="bf16", backend="torch-mps", kind="anyjev",
                 params=dict(model_name=repo_id, revision=sha, device="mps", dtype="bfloat16"), needs=[f"dl-{key}"],
                 repo="anyjev", hf=_rev(key), notes=f"AnyJev Decider L0 (defaults) over HFBackend on {name}; batch prior needs >= 8 states, so a single request runs without it"),
        ]
        if size == "0.8b":
            # Added 2026-11-13 after the first run: bf16 on MPS returned NaN for the noul/score questions.
            # fp32 isolates an MPS bf16 numerics issue from an adapter bug. 4B fp32 (~16 GB) cannot fit 16 GB, so 0.8B only.
            C += [dict(id=f"anyjev-q{size}/fp32", model=f"anyjev-q{size}", variant="fp32", backend="torch-mps", kind="anyjev",
                       params=dict(model_name=repo_id, revision=sha, device="mps", dtype="float32"), needs=[f"dl-{key}"],
                       repo="anyjev", hf=_rev(key),
                       notes=f"diagnostic: fp32 on MPS after bf16 NaN; AnyJev Decider L0 over HFBackend on {name}")]
        C += [
            dict(id=f"anyjev-q{size}/4bit", model=f"anyjev-q{size}", variant="4-bit", backend="none", kind=None, params={},
                 needs=[], repo="anyjev", hf=_rev(key),
                 unsupported=("not attempted: AnyJev 0.3.0 offers backends hf (transformers), vllm, sglang and llamacpp (GGUF); "
                              "its README lists an MLX backend as an open roadmap item, and none of the shipped paths provides a "
                              "4-bit model on Apple Silicon without a GGUF/llama.cpp setup that this study did not build"),
                 notes="4-bit training-free numbers come from jevmlx on MLX instead"),
            dict(id=f"jevmlx-q{size}/bf16", model=f"jevmlx-q{size}", variant="bf16", backend="mlx", kind="jevmlx",
                 params=dict(model_path=str(m / "jevmlx" / name)), needs=[f"link-jevmlx-{name}"], repo="jevmlx", hf=_rev(key),
                 notes="jevmlx engine on the unmodified checkpoint (symlinked so jevmlx picks its Qwen3 prompt profile); "
                       "jevmlx's /v1/systemone mapping passes choice names only"),
            dict(id=f"jevmlx-q{size}/4bit", model=f"jevmlx-q{size}", variant="4-bit", backend="mlx", kind="jevmlx",
                 params=dict(model_path=str(m / "jevmlx" / f"{name}-4bit-g{gs}")), needs=[f"convert-q4-{name}"], repo="jevmlx",
                 hf=_rev(key) + f" -> mlx_lm.convert q4 g{gs} locally",
                 notes=f"OURS: mlx_lm.convert(quantize=True, q_bits=4, q_group_size={gs}) of the same checkpoint"),
        ]
    # ---------------- jevmlx's own default model: ONLY if jevmlx ran none of the Qwen3.5 checkpoints above
    jv = [c["id"] for c in C if c["id"].startswith("jevmlx-q")]
    C.append(dict(id="jevmlx-default/4bit", model="jevmlx-default", variant="4-bit (Qwen2.5-7B-Instruct)", backend="mlx",
                  kind="jevmlx", params=dict(model_path=str(m / "jevmlx" / "Qwen2.5-7B-Instruct-4bit")),
                  needs=["link-jevmlx-default"], on_demand_prepare=True, fallback_for=jv, repo="jevmlx", hf=_rev("jevmlx-default"),
                  notes="fallback: jevmlx's documented default model (alias quality); run only because no Qwen3.5 checkpoint ran "
                        "under jevmlx; not the same base as Kev"))
    # ---------------- boundary test last: the source reports a 15.9 GB loading peak for this path
    C.append(dict(id="kev-4b/adapter-bf16", model="kev-4b", variant="adapter-bf16", backend="mlx", kind="kev",
                  params=dict(run=kev4, device="mps", backend="mlx"), needs=["dl-kev-4b", "dl-qwen3.5-4b-base"],
                  repo="kev", hf=kev4_rev,
                  notes="boundary test: kev.serve default (LoRA merged in memory); source reports a 15.9 GB loading peak"))
    for c in C:
        if c.get("kind") == "nanojev":
            c["params"]["repo"] = str(repo_dir("nanojev"))
    return C


def selftest_combos():
    """Fake combinations for testing the harness itself (no model, no GPU): used by `smoke_test.py --selftest`."""
    mk = lambda i, mode, **p: dict(id=f"selftest/{i}", model="selftest", variant=i, backend="cpu", kind="selftest",
                                   params=dict(mode=mode, **p), needs=[], repo="kev", hf="n/a", notes="harness self-test")
    return [mk("ok", "ok"), mk("raise", "raise"), mk("crash", "crash"), mk("alloc", "alloc", gb=0.6), mk("sleep", "sleep", s=30),
            dict(id="selftest/unsupported", model="selftest", variant="unsupported", backend="none", kind=None, params={},
                 needs=[], repo="kev", hf="n/a", unsupported="not attempted: self-test of the unsupported path", notes=""),
            dict(id="selftest/missing-prereq", model="selftest", variant="missing-prereq", backend="cpu", kind="selftest",
                 params=dict(mode="ok"), needs=["does-not-exist"], repo="kev", hf="n/a", notes="")]
