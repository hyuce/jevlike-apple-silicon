# jevlike-apple-silicon

Code and results for the technical report

> **Jev-like Decision Models on a 16 GB Apple Silicon Laptop: Feasibility, and What 8- and 4-bit Quantization Do to Calibrated Decisions**
> Hüseyin Yüce, Department of Mechatronics Engineering, Faculty of Technology, Marmara University, Istanbul, Türkiye
> huseyin@marmara.edu.tr · ORCID [0000-0001-5525-7733](https://orcid.org/0000-0001-5525-7733)
>
> arXiv: *link will be added after publication*

## Overview

Open "Jev-like" typed decision models (Kev, Laya, NanoJev, AnyJev, jevmlx) answer yes/no, multiple-choice and rating questions with calibrated probabilities instead of generated text. This repository contains the benchmark harness and all result files used to test them on one Apple M2 Pro MacBook Pro with 16 GB of unified memory.

Main results (taken from the files in `results/`):

- **Feasibility.** 25 of 29 model/precision/backend combinations ran under a 14 GB memory guard. Kev-4B's default adapter path was stopped at 14.03 GB. A merged full-weight bf16 checkpoint peaked at 9.53 GB and is bitwise identical to Kev's official merge (checked on Kev-0.8B: 320/320 tensors).
- **Harness validation.** Kev-0.8B (fp32) reached accuracy 0.6479 and Brier score 0.4298 on Kev's transfer-v4 development split (656 questions). Kev's model card reports 0.648 / 0.430.
- **Quantization (pre-registered).** 8- and 4-bit Kev showed no Holm-significant change in accuracy, NLL, Brier score, ECE, coverage at 5% error or AURC. The only significant differences were in mean confidence.
- **Decision churn (exploratory).** 4-bit changed 10.7% (Kev-0.8B; 95% CI 8.4–13.1%) and 5.3% (Kev-4B; 3.7–7.2%) of top-1 decisions while leaving accuracy statistically unchanged. Every changed decision had a reference confidence below 0.8.

## Repository layout

```
code/                     benchmark harness and analysis (MIT)
  setup_mac.sh            pinned clones + Python venv in ~/jevlike_cache; writes results/environment.md
  run_smoke.sh            downloads pinned weights, prepares merged/quantized checkpoints, runs the smoke test
  prepare_models.py       weight downloads, Kev LoRA merges, MLX 4-bit conversions
  smoke_test.py           one guarded subprocess per model/precision -> results/feasibility_matrix*.csv
  bench_accuracy.py       accuracy/calibration on Kev transfer-v4 (development) -> results/bench/accuracy/
  bench_latency.py        latency, memory and context scaling -> results/bench/latency/
  analysis_quant.py       paired bootstrap / Holm / McNemar, decision churn, recalibration -> results/analysis/
  make_figures.py         figures and tables of the report (written to paper/)
  adapters.py, combos.py, pins.py, guard.py, kev_merge_stream.py, run_kev_script.py, serve_systemone.py, env_report.py
results/                  all result files (CC BY 4.0)
  feasibility_matrix.csv, feasibility_matrix_detail.csv   29 combinations: runs, load time, peak memory, errors
  metrics_accuracy.csv    accuracy/calibration summary for the 8 Kev combinations
  latency_summary.csv     partial latency run (one combination)
  environment.md          hardware and software record (serial numbers removed)
  environment_notes.md    how each candidate is launched
  analysis/               H2 test results and exploratory analyses
  bench/accuracy/<combo>/ answers.jsonl, rows.json (per question), report.json, failures.json, worker.json, system.json
  bench/latency/          partial latency run
  raw/smoke/              raw smoke-test responses, Kev merge parity report, AnyJev batch-16 NaN evidence
  logs/                   setup, prepare, smoke, benchmark logs and the venv freeze
```

## Reproducing

Requirements: an Apple Silicon Mac, [`uv`](https://docs.astral.sh/uv/), and at least ~38 GB of free disk space (45 GB recommended). Model weights and upstream clones go to `~/jevlike_cache` (override with `JEVLIKE_CACHE`).

```bash
cd code
bash setup_mac.sh
bash run_smoke.sh
~/jevlike_cache/venv/bin/python bench_accuracy.py --only kev-0.8b/fp32,kev-0.8b/bf16,kev-0.8b/8bit,kev-0.8b/4bit,kev-0.8b/full-bf16-merged,kev-4b/full-bf16-merged,kev-4b/8bit,kev-4b/4bit
~/jevlike_cache/venv/bin/python analysis_quant.py --kev-src ~/jevlike_cache/repos/kev
```

To re-run only the statistics on the published results, clone Kev at commit `2074cc44256200bb37540103fa4572901faa10f3` and run `python code/analysis_quant.py --kev-src <kev clone>`. This reproduces `results/analysis/quant_*.csv` exactly.

Every upstream repository and weight file is pinned to a commit in `code/pins.py`. The evaluation requests are rebuilt from Kev's frozen suite (`evals/v4/transfer-v4`, development split) and are not redistributed here. Model weights and datasets are not included; obtain them from their original publishers under their licences.

## Licence

- Code (`code/`): [MIT](LICENSE).
- Results (`results/`) and documentation: [CC BY 4.0](LICENSE-CC-BY-4.0.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
