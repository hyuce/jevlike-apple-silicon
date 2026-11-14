# Environment record

Generated 2026-11-13 15:34:12 +0300 by `code/env_report.py` on this machine. Serial/UUID/UDID lines removed.

## Hardware

```
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac14,10
      Model Number: MNW93TU/A
      Chip: Apple M2 Pro
      Total Number of Cores: 12 (8 Performance and 4 Efficiency)
      Memory: 16 GB
      System Firmware Version: 20457.41.3
      OS Loader Version: 20457.41.3
      Activation Lock Status: Enabled
```

### GPU (system_profiler SPDisplaysDataType)

```
Graphics/Displays:

    Apple M2 Pro:

      Chipset Model: Apple M2 Pro
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 19
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
      Displays:
        Color LCD:
          Display Type: Built-in Liquid Retina XDR Display
          Resolution: 3456 x 2234 Retina
          Main Display: Yes
          Mirror: Off
          Online: Yes
          Automatically Adjust Brightness: No
          Connection Type: Internal
        DELL U3417W:
          Resolution: 3440 x 1440 (UWQHD - Ultra-Wide Quad HD)
          UI Looks like: 3440 x 1440 @ 60.00Hz
          Mirror: Off
          Online: Yes
          Rotation: Supported
```

### sysctl

```
hw.memsize: 17179869184
hw.ncpu: 12
hw.perflevel0.physicalcpu: 8
hw.perflevel1.physicalcpu: 4
machdep.cpu.brand_string: Apple M2 Pro
vm.swapusage: total = 2048.00M  used = 868.50M  free = 1179.50M  (encrypted)
```

## Operating system

```
ProductName:		macOS
ProductVersion:		27.2.1
BuildVersion:		26B131
macOS-27.2.1-arm64-arm-64bit
```

## Power / thermal at recording time

```
Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
Now drawing from 'AC Power'
 -InternalBattery-0 (id=<redacted>)	80%; AC attached; not charging present: true
```

## Python and packages

- python: `3.12.16` at `~/jevlike_cache/venv/bin/python`
- uv: `uv 0.13.8 (Homebrew 2026-11-12 aarch64-apple-darwin)`
- git: `git version 2.54.0 (Apple Git-157)`
- mlx: `0.32.2`
- mlx-metal: `0.32.2`
- mlx-lm: `0.31.3`
- torch: `2.8.0`
- transformers: `5.17.0`
- accelerate: `1.15.0`
- safetensors: `0.8.0`
- huggingface-hub: `1.32.0`
- tokenizers: `0.23.2`
- peft: `0.21.0`
- numpy: `2.5.3`
- pandas: `3.0.5`
- scipy: `1.18.1`
- psutil: `7.2.2`
- fastapi: `0.141.1`
- uvicorn: `0.53.0`
- pydantic: `2.13.5`
- typesafe-sdk: `0.6.0`
- kev: `0.1.0`
- laya: `0.4.31`
- vegaml: `0.16.1`
- anyjev: `0.3.0`
- jevmlx: `0.1.0`

## MLX / Metal

```
{'metal_available': True, 'device_info': {'device_name': 'Apple M2 Pro', 'max_recommended_working_set_size': 12713115648, 'memory_size': 17179869184, 'architecture': 'applegpu_g14s', 'max_buffer_length': 9534832640, 'resource_limit': 499000}, 'default_device': 'Device(gpu, 0)', 'matmul_512_ok_ms': 79.41}
```

## torch MPS

```
{'mps_built': True, 'mps_available': True, 'mps_matmul_ok': True, 'mps_recommended_max_memory_gb': 12.713}
```

## Repositories (clones in the local cache, not in Google Drive)

| repo | URL | pinned SHA | checked-out SHA |
|---|---|---|---|
| kev | https://github.com/jaredpalmer/kev.git | `2074cc44256200bb37540103fa4572901faa10f3` | `2074cc44256200bb37540103fa4572901faa10f3` |
| laya | https://github.com/NandhaKishorM/laya.git | `3a38cbe83f6ef556f3125fb0065fdc5a55d05792` | `3a38cbe83f6ef556f3125fb0065fdc5a55d05792` |
| nanojev | https://github.com/TianyuCodings/NanoJev.git | `45b629283ddeb3b3784806f352bbf5e97f71dd40` | `45b629283ddeb3b3784806f352bbf5e97f71dd40` |
| jevmlx | https://github.com/bnsd55/jevmlx.git | `5f94ae262c1cf4f226b436433946be281ec1ac99` | `5f94ae262c1cf4f226b436433946be281ec1ac99` |
| anyjev | https://github.com/nokia-applied-research/AnyJev.git | `35311c58955716cb8955ce1368a665988bc63735` | `35311c58955716cb8955ce1368a665988bc63735` |
| laya-mlx | https://github.com/mizorewww/laya-mlx.git | `2159c6c9988e4cbfd6281db1888cef45de09cea9` | `2159c6c9988e4cbfd6281db1888cef45de09cea9` |

## Hugging Face revisions pinned for the smoke test

| key | repo | commit | note |
|---|---|---|---|
| kev-0.8b | jaredpalmer/kev-0.8b | `9a45d25eb2ab761841196625383fa1dff0e56c1e` | README Kev 1.0 weights rev |
| kev-4b | jaredpalmer/kev-4b | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` | README Kev 1.0 weights rev |
| qwen3.5-0.8b-base | Qwen/Qwen3.5-0.8B-Base | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Kev-0.8B base |
| qwen3.5-4b-base | Qwen/Qwen3.5-4B-Base | `1001bb4d826a52d1f399e183466143f4da7b741b` | Kev-4B base |
| qwen3.5-0.8b | Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | optional |
| qwen3.5-4b | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | optional |
| laya | convaiinnovations/laya | `85cc74ddae6109077ebe2853a9b3077fb6606a78` | ModernBERT-large encoder, ~421M |
| laya-multilingual | convaiinnovations/laya-multilingual | `1720e3e3357cfe1e281542e223f8273b0890ca34` | mmBERT-base encoder, ~322M |
| nanojev | C-Tianyu/NanoJev | `047b927b30882a1138fc504821b82ac145a4b81a` | tag unified-games-v1 |
| laya-mlx | aac6fef/laya-mlx | `20aed815fc6acde75733882e7ec0e3f28aeb9717` | FP16 MLX port of convaiinnovations/laya |
| laya-multilingual-mlx | aac6fef/laya-multilingual-mlx | `f2b4faf51023039425946074e2cf1361d2db11d5` | FP16 MLX port of convaiinnovations/laya-multilingual |
| jevmlx-default | mlx-community/Qwen2.5-7B-Instruct-4bit | `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed` | jevmlx default (alias quality) |

## Paths

- cache (weights, clones, venv): `~/jevlike_cache`
- HF_HOME: `~/jevlike_cache/hf`
- disk free at cache: 58.8 GB

