"""One interface over every candidate model: `runner.answer(request) -> /v1/systemone-shaped body`.

A request is the TypeSafe System One body ({"state", "model", "questions": {id: {type, instructions, criteria}}}).
Each runner loads once in __init__, answers with `answer(req)`, describes itself with `info()` and releases with
`close()`. Used by smoke_test.py (one runner per subprocess) and serve_systemone.py (an HTTP endpoint that
`kev.benchmark --remote` can score), so the benchmark phase reuses exactly the code the smoke test checked.

What is upstream and what is ours, per runner:
  KevRunner      upstream kev.checkpoint + kev.serve.Server (the code path of `python -m kev.serve`).
                 OURS: optional `quantize_mlx_backbone` (Kev has no quantized path; see that function).
  LayaRunner     upstream `laya.load(...).predict` (= Agent.system_one). OURS: precision switch through Laya's own
                 MPS autocast knobs (fp32: autocast off; fp16: autocast from 1 row, LAYA_MPS_AMP_MIN_ROWS=1).
  LayaMlxRunner  upstream laya_mlx.load(...).predict (FP16 MLX port of Laya); nothing of ours.
  NanoJevRunner  upstream helpers of scripts/predict_toy_decisions.py. Its DecisionPredictor refuses any non-CUDA
                 device (NanoJev documents CUDA only), so mode="port" is OUR port: the same construction and forward
                 with device=mps or cpu and torch.autocast(<device>, bf16) in place of autocast("cuda").
                 mode="upstream" runs the original class.
                 Request mapping noul->"boolean" (NanoJev's spelling). Temperature 1.0 (NanoJev ships none).
  AnyJevRunner   upstream anyjev.Decider over anyjev.backends.hf.HFBackend (torch, device=mps). Defaults kept
                 (level L0, prior "batch", min_prior_n 8). OURS: request mapping (choice option text "name: desc").
  JevmlxRunner   upstream jevmlx.engine.load_engine/run_parallel_generation and the /v1/systemone mapping of
                 jevmlx.serve (_systemone_to_schema/_format_systemone_response). Note: that mapping passes choice
                 option NAMES only, not their descriptions.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


# =============================================================================================== Kev
def quantize_mlx_backbone(model, bits: int, group_size: int = 64) -> dict:
    """OUR quantization of a loaded kev.mlx_model.MLXDecisionModel (Kev 2074cc44 has no quantized path: no
    'quantiz' anywhere in kev/, and MLX_DTYPES covers bfloat16/float32 only).

    What is done: `mlx.nn.quantize(model.text, group_size, bits, mode="affine")` on the Qwen3.5 text backbone the
    pointer head reads (embeddings, attention, Gated DeltaNet and MLP projections), with the predicate mlx-lm's own
    `quantize_model` uses for Qwen3.5 (module has `to_quantized` and weight.shape[-1] % group_size == 0; qwen3_5 has
    no extra predicate for dense models). It is applied to the bf16 backbone AFTER Kev merged the LoRA (adapter
    checkpoint) or loaded the merged full weights, so the adapter delta is inside the quantized weights.
    What is NOT quantized: the pointer head (a torch fp32 module, `model.head`, untouched, temperature unchanged),
    norms, DeltaNet conv1d/A_log/dt_bias, anything whose last dim is not a multiple of group_size (listed)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    text = model.text
    nbytes = lambda m: sum(v.nbytes for _, v in tree_flatten(m.parameters()))
    before = nbytes(text)
    quantized, skipped = [], []

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            skipped.append(f"{path} {tuple(module.weight.shape)}")
            return False
        quantized.append(path)
        return True

    t = time.perf_counter()
    nn.quantize(text, group_size=group_size, bits=bits, mode="affine", class_predicate=predicate)
    mx.eval(text.parameters())
    mx.clear_cache()
    head_dtypes = sorted({str(p.dtype) for p in model.head.parameters()})
    return {"method": "mlx.nn.quantize(model.text, mode='affine') with mlx-lm's quantize_model predicate",
            "bits": bits, "group_size": group_size, "quantized_modules": len(quantized),
            "skipped_modules": skipped, "backbone_bytes_before": int(before), "backbone_bytes_after": int(nbytes(text)),
            "pointer_head_dtypes": head_dtypes, "pointer_head_quantized": False, "seconds": round(time.perf_counter() - t, 3)}


class KevRunner:
    def __init__(self, run: str, device: str = "mps", backend: str = "mlx", dtype: str | None = None,
                 quant_bits: int | None = None, group_size: int = 64):
        import torch
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server

        dt = {None: None, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        attn = "sdpa" if (device == "mps" and backend == "torch") else None   # kev.serve's default on Apple GPUs
        self.device, self.backend_req = device, backend
        self.ck = Checkpoint(run)
        self.opts = LoadOptions(dtype=dt, backend=backend, attn=attn)
        self.tok, self.model = self.ck.load(device, self.opts)
        self.quant = None
        if quant_bits:
            if getattr(self.model, "backend", None) != "mlx":
                raise ValueError("quantization is implemented for the MLX backend only")
            self.quant = quantize_mlx_backbone(self.model, quant_bits, group_size)
        # release_date given so the Server does not query the Hub
        self.server = Server(self.ck, self.tok, self.model, device, release_date="n/a (local smoke run)")

    def answer(self, req: dict) -> dict:
        from kev.api import SystemOneRequest
        return self.server.answer(SystemOneRequest.model_validate(req))

    def info(self) -> dict:
        m = self.ck.meta
        return {"runner": "kev", "run": self.ck.requested, "resolved_path": self.ck.path, "full_weights": self.ck.full,
                "backend": getattr(self.model, "backend", None), "dtype": str(getattr(self.model, "dtype", None)),
                "device": self.device, "load_options": repr(self.opts), "temperature": float(self.model.head.temperature),
                "base": m.base, "base_revision": m.base_revision, "weights_dtype": m.weights_dtype, "quantization": self.quant}

    def close(self):
        self.server.close()


# =============================================================================================== Laya
class LayaRunner:
    def __init__(self, model_id: str, revision: str | None, device: str = "mps", precision: str = "fp32"):
        if precision == "fp16":
            if device != "mps":
                raise ValueError("fp16 is only offered through Laya's MPS autocast")
            os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "1"   # read in Agent.__init__: autocast every call
        import laya
        self.precision, self.device_req = precision, device
        self.agent = laya.load(model_id, device=device, revision=revision)
        if precision == "fp32":
            self.agent.amp_enabled = False                  # Laya keeps fp32 weights; this turns autocast off

    def answer(self, req: dict) -> dict:
        t = time.perf_counter()
        out = self.agent.predict(req["state"], req["questions"])
        body = {"model": req.get("model", "laya"), "answers": out.get("answers", out)}
        body["latency_ms"] = out.get("latency_ms", round((time.perf_counter() - t) * 1000, 1))
        for k in ("usage", "routing"):
            if k in out:
                body[k] = out[k]
        return body

    def info(self) -> dict:
        a = self.agent
        d = {"runner": "laya", "model_id": getattr(a, "model_id", None), "revision": getattr(a, "revision", None),
             "device": str(getattr(a, "device", None)), "autocast_dtype": str(getattr(a, "dtype", None)),
             "amp_enabled": getattr(a, "amp_enabled", None), "mps_amp_min_rows": getattr(a, "mps_amp_min_rows", None),
             "precision_requested": self.precision}
        try:
            d["dtype_for_3_rows"] = str(a.dtype_for(3))
        except Exception as e:  # noqa: BLE001
            d["dtype_for_3_rows"] = f"n/a ({type(e).__name__})"
        return d

    def close(self):
        pass


class LayaMlxRunner:
    """Upstream laya-mlx (mizorewww/laya-mlx @2159c6c9): `laya_mlx.load(id, revision=...).predict` (= system_one),
    the FP16 MLX port of the Laya checkpoints. Nothing of ours beyond the call."""

    def __init__(self, model_id: str, revision: str | None):
        import laya_mlx
        self.agent = laya_mlx.load(model_id, revision=revision)
        self.model_id = model_id

    def answer(self, req: dict) -> dict:
        t = time.perf_counter()
        out = self.agent.predict(req["state"], req["questions"])
        body = {"model": req.get("model", "laya-mlx"), "answers": out.get("answers", out)}
        body["latency_ms"] = out.get("latency_ms", round((time.perf_counter() - t) * 1000, 1))
        return body

    def info(self) -> dict:
        a = self.agent
        return {"runner": "laya-mlx", "model_id": self.model_id, "revision": getattr(a, "revision", None),
                "model_dir": str(getattr(a, "model_dir", "")), "dtype": str(getattr(a, "dtype", None))}

    def close(self):
        pass


# =============================================================================================== NanoJev
def systemone_to_nanojev(req: dict) -> dict:
    qs = {}
    for qid, q in req["questions"].items():
        typ = {"noul": "boolean"}.get(q["type"], q["type"])
        item = {"type": typ, "instructions": q.get("instructions") or ""}
        if q.get("criteria") is not None:
            if typ == "choice":
                item["criteria"] = {k: (v if isinstance(v, str) and v.strip() else k) for k, v in q["criteria"].items()}
            elif typ == "score":
                item["criteria"] = list(q["criteria"])
            elif typ == "boolean":
                item["criteria"] = {k: v for k, v in q["criteria"].items() if v}
        qs[qid] = item
    return {"states": [{"id": "s0", "state": req["state"], "questions": qs}]}


def nanojev_to_systemone(result: dict, req: dict) -> dict:
    ans = result["states"][0]["answers"]
    out = {}
    for qid, q in req["questions"].items():
        a = ans[qid]
        if q["type"] == "noul":
            out[qid] = {"type": "noul", "noul": a["p_true"]}
        elif q["type"] == "choice":
            out[qid] = {"type": "choice", "choice": a["choice"], "probabilities": a["probabilities"]}
        else:
            out[qid] = {"type": "score", "score": a["score"], "probabilities": a["probabilities"],
                        "legend": {str(i): lvl for i, lvl in enumerate(q["criteria"])}}
    return out


class NanoJevRunner:
    def __init__(self, checkpoint_dir: str, repo: str, mode: str = "port", precision: str = "bf16", temperature: float = 1.0,
                 device: str = "mps"):
        sys.path.insert(0, str(Path(repo) / "scripts"))
        import predict_toy_decisions as P
        self.P, self.mode, self.precision, self.temperature, self.device_name = P, mode, precision, temperature, device
        if mode == "upstream":
            # the original class, default device cuda:0: on a Mac it raises before any weight is read
            self.engine = P.DecisionPredictor(checkpoint_dir, precision=precision)
            return
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        # --- OUR port: DecisionPredictor.__init__ (predict_toy_decisions.py @45b62928) with the device swapped ---
        root, paths = P.local_checkpoint_files(checkpoint_dir)
        run_config = P.read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config lacks a valid set_head")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS is not available")
        device = torch.device(device)
        tok = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True, trust_remote_code=False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True, trust_remote_code=False)
        body_config.use_cache = False
        self.limit = run_config.get("max_length", 512)
        body = AutoModel.from_config(body_config, attn_implementation="sdpa", trust_remote_code=False).float()
        model = P.load_decision_model_class()(body, run_config["set_head"])
        weights = load_file(str(paths["weights"]), device="cpu")
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(device=device, dtype=torch.float32)   # upstream keeps fp32 storage and autocasts the forward
        model.eval()
        self.torch, self.model, self.tok, self.device, self.root, self.run_config = torch, model, tok, device, root, run_config

    def _predict_mps(self, payload):
        torch, P = self.torch, self.P
        examples = P.prepare_examples(payload, self.tok, self.limit)
        out = {s["id"]: {"id": s["id"], "answers": {}} for s in payload["states"]}
        with torch.inference_mode():
            for batch in P.complete_question_batches(examples, 0):
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                    logits, _ = self.model(batch, self.tok.pad_token_id)
                for ex, values in zip(batch, logits):
                    scores = values[: len(ex["candidate_ids"])].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("non-finite logits")
                    probs = (scores / self.temperature).softmax(-1).cpu().tolist()
                    out[ex["state_id"]]["answers"][ex["qid"]] = P.answer_from_probabilities(ex, probs)
        if self.device.type == "mps":
            torch.mps.synchronize()
        return {"states": list(out.values())}

    def answer(self, req: dict) -> dict:
        payload = systemone_to_nanojev(req)
        t = time.perf_counter()
        result = self.engine.predict(payload) if self.mode == "upstream" else self._predict_mps(payload)
        return {"model": req.get("model", "nanojev"), "answers": nanojev_to_systemone(result, req),
                "latency_ms": round((time.perf_counter() - t) * 1000, 1)}

    def info(self) -> dict:
        return {"runner": "nanojev", "mode": self.mode, "precision": self.precision, "temperature": self.temperature,
                "base_model": getattr(self, "run_config", {}).get("model"),
                "base_revision": getattr(self, "run_config", {}).get("resolved_model_revision"),
                "set_head": getattr(self, "run_config", {}).get("set_head"), "max_length": getattr(self, "limit", None),
                "device": self.device_name,
                "port_note": None if self.mode == "upstream" else f"device={self.device_name} port of DecisionPredictor (ours)"}

    def close(self):
        pass


# =============================================================================================== AnyJev
class AnyJevRunner:
    def __init__(self, model_name: str, revision: str | None, device: str = "mps", dtype: str = "bfloat16",
                 batch_size: int = 1):
        # batch_size=1 (AnyJev default 16) is a DIAGNOSTIC choice, not a verified fix. Observed 2026-11-13 on the Mac
        # (transformers 5.17 / torch 2.8 / MPS, batch 16): noul and score questions -- which go through
        # HFBackend.next_token_logprobs, where prompts of different lengths are left-padded into one batch -- returned
        # NaN in bf16 AND fp32, while the choice question (equal-length shared-prefix path) was finite. Hypothesis:
        # left padding on the Qwen3.5 hybrid (Gated DeltaNet layers ignore the attention mask) produces NaN. NOT
        # reproduced in the sandbox (tiny random Qwen3_5ForCausalLM, transformers 5.22 / torch 2.15, CPU: padded rows
        # finite, max|d| vs unbatched ~1e-7). Batch 1 removes padding and leaves each prompt's computation unchanged,
        # so the batch-16 vs batch-1 comparison isolates the cause. Recorded in the deviation log #6.
        from anyjev import Decider
        from anyjev.backends.hf import HFBackend
        self.backend = HFBackend(model_name, device=device, dtype=dtype, revision=revision, batch_size=batch_size)
        self.decider = Decider(self.backend)       # defaults: level "L0", prior "batch", min_prior_n 8
        self.model_name, self.revision, self.device, self.dtype = model_name, revision, device, dtype
        self.batch_size = batch_size

    def answer(self, req: dict) -> dict:
        from anyjev import Question
        qs, back = [], []
        for qid, q in req["questions"].items():
            text = q.get("instructions") or ""
            if q["type"] == "noul":
                qs.append(Question.noul(text, name=qid)); back.append((qid, q, None))
            elif q["type"] == "choice":
                names = list(q["criteria"])
                opts = [f"{k}: {v}" if isinstance(v, str) and v.strip() else k for k, v in q["criteria"].items()]
                qs.append(Question.choice(text, opts, name=qid)); back.append((qid, q, dict(zip(opts, names))))
            else:
                qs.append(Question.score(text, levels=list(q["criteria"]), name=qid)); back.append((qid, q, None))
        t = time.perf_counter()
        ds = self.decider.decide(_state_text(req["state"]), qs)
        if self.device == "mps":
            import torch
            torch.mps.synchronize()
        ms = round((time.perf_counter() - t) * 1000, 1)
        answers, levels = {}, {}
        for d, (qid, q, names) in zip(ds, back):
            levels[qid] = d.level
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": d.p_true}
            elif q["type"] == "choice":
                dist = {names[o]: p for o, p in d.distribution.items()}
                answers[qid] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
            else:
                answers[qid] = {"type": "score", "score": d.value,
                                "probabilities": {str(i): float(p) for i, p in enumerate(d.probs)},
                                "legend": {str(i): lvl for i, lvl in enumerate(q["criteria"])}}
        return {"model": req.get("model", "anyjev"), "answers": answers, "latency_ms": ms, "anyjev_levels": levels}

    def info(self) -> dict:
        return {"runner": "anyjev", "model": self.model_name, "revision": self.revision, "device": self.device,
                "dtype": self.dtype, "batch_size": self.batch_size,
                "level": self.decider.level, "prior": getattr(self.decider, "prior", None),
                "min_prior_n": getattr(self.decider, "min_prior_n", None)}

    def close(self):
        pass


# =============================================================================================== jevmlx
class JevmlxRunner:
    def __init__(self, model_path: str, temperature: float = 1.0):
        # jevmlx picks its prompt profile from the LAST path component (engine._profile_for: a name starting with
        # "qwen3" disables the Qwen3 thinking template), so model_path must end in a name like Qwen3.5-0.8B-Base.
        from jevmlx.engine import load_engine
        self.model_path, self.temperature = model_path, temperature
        self.engine = load_engine(model_path)

    def answer(self, req: dict) -> dict:
        from jevmlx.engine import run_parallel_generation
        from jevmlx.schema import StructuredSchema
        from jevmlx.serve import _format_systemone_response, _systemone_to_schema
        schema = StructuredSchema(_systemone_to_schema(req["questions"]))
        t = time.perf_counter()
        result = run_parallel_generation(self.engine, _state_text(req["state"]), schema, temperature=self.temperature)
        ms = round((time.perf_counter() - t) * 1000, 1)
        body = _format_systemone_response(result, req["questions"], self.engine.model_id)
        body["latency_ms"] = result.get("elapsed_ms", ms)
        return body

    def info(self) -> dict:
        e = self.engine
        return {"runner": "jevmlx", "model_path": self.model_path, "model_id": getattr(e, "model_id", None),
                "revision": getattr(e, "revision", None), "weight_bytes": getattr(e, "weight_bytes", None),
                "profile": repr(getattr(e, "profile", None)), "temperature": self.temperature}

    def close(self):
        try:
            from jevmlx.engine import clear_engine_cache
            clear_engine_cache()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================================== factory
RUNNERS = {"kev": KevRunner, "laya": LayaRunner, "layamlx": LayaMlxRunner, "nanojev": NanoJevRunner,
           "anyjev": AnyJevRunner, "jevmlx": JevmlxRunner}


def build_runner(kind: str, params: dict):
    return RUNNERS[kind](**params)
