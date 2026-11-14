"""Pinned sources for the Jev-like-on-Apple-Silicon study: the single source of truth for the Python side.

Every value below was resolved on 2026-11-13 from the public GitHub / Hugging Face APIs (not guessed):
  * GitHub repos: `git ls-remote <url> HEAD` and a shallow fetch of exactly this SHA.
  * Hugging Face: `GET /api/models/<repo>/revision/<rev>` -> `sha` (the commit; tags on these repos are
    annotated, so the tag object id differs from the commit id).
Candidate URLs come from the awesome lists named in the protocol
(AnotiaWang/awesome-decision-models @ 9fa2fb1b, andyrewlee/awesome-system-one @ 405c9d0d,
 cobanov/awesome-jev @ a3b09e35).

`setup_mac.sh` carries the same repo SHAs in bash; `TMP/test_harness_cpu.py` checks that both agree.
"""
import os
from pathlib import Path

# name -> (git url, commit sha)
REPOS = {
    "kev": ("https://github.com/jaredpalmer/kev.git", "2074cc44256200bb37540103fa4572901faa10f3"),
    "laya": ("https://github.com/NandhaKishorM/laya.git", "3a38cbe83f6ef556f3125fb0065fdc5a55d05792"),
    "nanojev": ("https://github.com/TianyuCodings/NanoJev.git", "45b629283ddeb3b3784806f352bbf5e97f71dd40"),
    "jevmlx": ("https://github.com/bnsd55/jevmlx.git", "5f94ae262c1cf4f226b436433946be281ec1ac99"),
    # nokia-applied-research/AnyJev and MorrisZJ/AnyJev resolve to the same commit (checked 2026-11-12)
    "anyjev": ("https://github.com/nokia-applied-research/AnyJev.git", "35311c58955716cb8955ce1368a665988bc63735"),
    # independent MLX port of Laya (awesome-decision-models; repo inventory 2026-11-13)
    "laya-mlx": ("https://github.com/mizorewww/laya-mlx.git", "2159c6c9988e4cbfd6281db1888cef45de09cea9"),
}

# key -> (hub repo id, commit sha, note)
HF = {
    # Kev 1.0 weight revisions named in the Kev README ("Hub revision of the weights": 9a45d25e / 139fdd94), full SHAs
    # from the Hub commit list (= results repo inventory). Their adapter_model.safetensors and head.pt have the same
    # LFS sha256 as main/tag v1.0 (0.8B: adapter 9b908623acb1..., head f400bd12802b...; 4B: adapter 90e817356246...,
    # head dd633435998e...), checked through the Hub API on 2026-11-13.
    "kev-0.8b": ("jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e", "README Kev 1.0 weights rev"),
    "kev-4b": ("jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101", "README Kev 1.0 weights rev"),
    # bases named in the Kev adapters' training_config.json (base_revision)
    "qwen3.5-0.8b-base": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", "Kev-0.8B base"),
    "qwen3.5-4b-base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", "Kev-4B base"),
    # post-trained (instruct) siblings: optional training-free bases (--tf-base instruct)
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "optional"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "optional"),
    "laya": ("convaiinnovations/laya", "85cc74ddae6109077ebe2853a9b3077fb6606a78", "ModernBERT-large encoder, ~421M"),
    "laya-multilingual": ("convaiinnovations/laya-multilingual", "1720e3e3357cfe1e281542e223f8273b0890ca34", "mmBERT-base encoder, ~322M"),
    # tag unified-games-v1 (tag object 456cea98...) -> this commit (best.safetensors lfs sha256 f68c47d66998...)
    "nanojev": ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a", "tag unified-games-v1"),
    # laya-mlx FP16 ports of the two Laya checkpoints
    "laya-mlx": ("aac6fef/laya-mlx", "20aed815fc6acde75733882e7ec0e3f28aeb9717", "FP16 MLX port of convaiinnovations/laya"),
    "laya-multilingual-mlx": ("aac6fef/laya-multilingual-mlx", "f2b4faf51023039425946074e2cf1361d2db11d5",
                              "FP16 MLX port of convaiinnovations/laya-multilingual"),
    # jevmlx's documented default model (alias "quality"); used only if jevmlx cannot run the Qwen3.5 checkpoints
    "jevmlx-default": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
                       "jevmlx default (alias quality)"),
}

# the allow_patterns each upstream loader itself uses, so a pre-download satisfies it offline
KEV_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]            # kev.checkpoint.resolve_run
LAYA_PATTERNS = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]   # laya.agent.Agent.__init__
LAYA_MLX_PATTERNS = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*",
                     "mlx_config.json"]                                            # laya_mlx.agent.resolve_model
NANOJEV_PATTERNS = ["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"]  # NanoJev README

QUANT_GROUP_SIZE = 64      # mlx-lm's default for mode="affine" (mlx_lm.utils.quantize_model)


def cache_dir() -> Path:
    return Path(os.environ.get("JEVLIKE_CACHE", str(Path.home() / "jevlike_cache"))).expanduser()


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def project_dir() -> Path:
    return code_dir().parent


def results_dir() -> Path:
    """results; JEVLIKE_RESULTS overrides it (used only by the harness self-tests, never for real runs)."""
    return Path(os.environ["JEVLIKE_RESULTS"]) if os.environ.get("JEVLIKE_RESULTS") else project_dir() / "results"


def repo_dir(name: str) -> Path:
    return cache_dir() / "repos" / name


def models_dir() -> Path:
    return cache_dir() / "models"


def hub_id(key: str) -> str:
    """'repo@sha', the form kev.checkpoint accepts for a pinned Hub checkpoint."""
    repo, sha, _ = HF[key]
    return f"{repo}@{sha}"


def assert_not_on_drive(path: Path) -> None:
    """Weights and clones must never land in Google Drive (protocol section 2)."""
    p = str(Path(path).expanduser().resolve())
    if "CloudStorage" in p or "Google Drive" in p or "GoogleDrive" in p:
        raise SystemExit(f"refusing to use {p}: model weights / clones must not be stored in Google Drive")
