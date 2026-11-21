# Xolotl on Windows: Flash-Next XQ and DeepSeek-V4.1 XQ V4 on one RTX 3090

This branch (`windows-local`) is upstream Xolotl plus:

- **Three Windows build fixes** (CUDA 12.8 + MSVC), one commit each:
  - `src/core/dense_model.cpp`: a CUDA-13-only device attribute, now behind `CUDA_VERSION >= 13000` (upstream #138).
  - `src/kernels/cuda/expert_keep.cu`: MSVC has no `strtok_r`; it maps to `strtok_s` (upstream #143).
  - `tools/v41/engine/ds41_dense.cpp`: the 8-argument `cudaMemcpyBatchAsync` is CUDA 13's. CUDA 12.8 declares it with
    a `failIdx` argument, so the guard is now `>= 13000` instead of `>= 12080`.
- **This folder:** the scripts and configs used to run both models natively on Windows 11, and the speeds measured.

Both engines build and run natively with MSVC and CUDA 12.8. No WSL is needed.

## What it was tested on

- **GPU:** RTX 3090 24 GB (sm_86), PCIe 4.0 x16.
- **CPU and RAM:** Ryzen 9 9950X3D, 94 GB.
- **Disk:** WD_BLACK SN850X 8 TB NVMe for the models.
- **OS:** Windows 11 Pro.

| Model | Engine | Disk | RAM | Decode | Prompt reading |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next Swift-1.5 XQ 45 GB ("qfn") | `strata` | ~80 GB | ~37 GB resident (74 GB committed) | 162-167 tok/s on code (MTP draft accepts ~85%) | 2,400 tok/s (13K-token prompt) |
| DeepSeek-V4.1-Flash-1017 XQ V4, fast mode | `ds41_generate` | ~412 GB | ~65 GB (40 GiB expert cache) | 32-33 tok/s on code | ~170-185 tok/s |
| the same without fast mode | `ds41_generate` | | | 15-17 tok/s | ~115 tok/s |

All runs were greedy with a 40 GiB DeepSeek expert cache. Notes:

- **Only one model fits on a 24 GB card.** qfn leaves ~340 MiB of VRAM free, so stop one server before starting the other.
- **DeepSeek has a startup cost on every reply.** Any prompt with more than 16 new tokens takes about 11-14 s before the
  first token, because the batch path pulls ~13K experts through PCIe and from the SSD.
- **DeepSeek streams experts from the SSD.** In decode it reads ~0.5 GB/s, at a queue depth under 1, and waits on
  those reads for about half of each step. Put its models on your fastest NVMe drive.
- **DeepSeek thinks at length.** On an agent task it can spend 20+ minutes (~25K tokens) thinking on one turn.

## Requirements

- **NVIDIA GPU with 24 GB** (tested on sm_86). Other architectures: set `CUDA_ARCH` (89 = RTX 40xx, 120 = RTX 50xx).
- **CUDA toolkit 12.8**, with `CUDA_PATH` pointing at it.
- **Visual Studio 2022** with "Desktop development with C++" (the Build Tools are enough).
- **Python 3.12 and Git.**
- **RAM:** 64 GB is enough for qfn. DeepSeek wants 96 GB or more; see `make_configs.py` for how its cache is sized.
- **Disk:** NVMe. qfn ~80 GB; DeepSeek ~412 GB (163 GB model, 40 GB fast-mode copy, 203 GB Engram tables,
  6 GB draft and tokenizer). The models and the repo must be on the **same drive** for DeepSeek's fast mode.

## 1. Clone and build

```bat
git clone -b windows-local https://github.com/kev-ju/Xolotl.git D:\src\Xolotl
cd /d D:\src\Xolotl
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt psutil huggingface_hub hf_xet
python -m venv .venv-build
.venv-build\Scripts\pip install cmake ninja
windows-local\build-both.bat
```

The build takes a few minutes and produces `build-win\ds41_generate.exe` (DeepSeek) and `build-cuda12\strata.exe`
(Flash-Next). The commands below assume the models live in `D:\models`; use any folder on an NVMe drive.

## 2. Flash-Next XQ (qfn)

```bat
.venv\Scripts\python windows-local\fetch_qfn.py D:\models
.venv\Scripts\python windows-local\make_qfn_mtp.py D:\models
.venv\Scripts\python windows-local\make_configs.py D:\models
.venv\Scripts\python -m serve.server --engine strata --config D:\models\Qwen3.8-Flash-Next-Swift-1.5-XQ\xolotl-qfn45-win.json --port 8099
```

1. `fetch_qfn.py` downloads ~74 GB, resumably. It also makes two hard links: the Hub pack's `native_experts.txt` names
   the shards `Swift1.5-QFN-XQ-45GB-0000N-of-00002.gguf`, not the names they were uploaded under.
2. `make_qfn_mtp.py` builds the MTP draft model in `D:\models\xolotl-mtp\rt`.
3. `make_configs.py` writes all three serve configs with your paths. Run it again if you move anything.
4. The server is ready in about 20 s on http://127.0.0.1:8099 (OpenAI API at `/v1`, model `qfn45`, no API key).
   The engine log is `D:\models\xolotl-qfn45.log`.

## 3. DeepSeek-V4.1 XQ V4

```bat
.venv\Scripts\python windows-local\fetch_dsv41.py D:\models
.venv\Scripts\python windows-local\make_configs.py D:\models
.venv\Scripts\python -I -X utf8 tools\v41\serve\serve_ds41.py --engine strata --config D:\models\xolotl-dsv41-xq-v4-fast-win.json --port 8150
```

- **`fetch_dsv41.py`** downloads, checks the hashes, and joins the 16 model parts into one file. It deletes the parts
  only once the joined file matches its published sha256. Rerun it to resume an interrupted download.
- **The fast config** reads a 2-bit copy of V4's two largest expert formats. Here that doubled decode speed, mostly
  because the DSpark draft was accepted more often (4-5 tokens per step instead of ~2.8). It is lossy: the model card
  reports the same coding score with it, but outputs differ. The plain config is `xolotl-dsv41-xq-v4-win.json`.
- **Startup:** ready in under a minute on http://127.0.0.1:8150 (model `dsv41-xq-v4`).
- **Engine log:** `D:\models\xolotl-dsv41.log`. These lines should appear at startup:
  - `arena: pinned` (an unpinned arena decodes about a third slower);
  - `shadow: those experts ...` (fast mode in use);
  - `conversation pool 4096 MiB`.
- **Watch RAM on the first start.** Upstream's 3090 config was made for 128 GB: its 56-72 GiB cache plus the default
  8 GiB conversation pool left 1.3 GiB free on a 94 GB machine. `make_configs.py` sizes the cache as installed RAM
  minus 54 GiB, at most 72, and sets the pool to 4 GiB. Override it with `--arena-gib N`.
- **Harmless warning:** `dense requant cache: cannot write ./xolotl-cache/C:\...`. The cache file name is built from the
  absolute model path, so the requantization is redone at each start; it costs a few seconds.

To keep a server running after you close the terminal, start it through `windows-local\start_detached.ps1`. Its header
shows the command.

## 4. Zoo Code (VS Code)

- **Import the profiles.** Point VS Code's `settings.json` at the file and reload VS Code:
  `"zoo-code.autoImportSettingsPath": "D:\\src\\Xolotl\\windows-local\\zoo-code-profiles.json"`. Zoo's own Import
  button in its settings works too. It adds `xolotl-qfn45` and `xolotl-dsv41`; pick one in the profile dropdown.
- **Run commands in VS Code's terminal.** Set `terminalShellIntegrationDisabled: false`, which the import file does.
  Otherwise Zoo runs commands through its own cmd.exe runner. Make Git Bash VS Code's default terminal if your tasks
  use shell scripts.
- **Approve the commands your tasks use.** Auto-approve only runs a command if every part, split at `&&`, `;` and `|`,
  starts with an entry in the allowed list. The import's list did not stick here, so put it in VS Code's
  `settings.json`, which is read live:
  ```json
  "zoo-code.allowedCommands": ["git log", "git diff", "git status", "python", "ls", "cat", "grep", "./run_tests.sh"]
  ```
- **Retries:** if a server restarts mid-task, Zoo shows "API Request Failed". Click Retry, or turn on auto-approve for
  retries.

## 5. Measure

```bat
.venv\Scripts\python windows-local\speed_test.py 8099 qfn45 --runs 3 --long serve\server.py
.venv\Scripts\python windows-local\speed_test.py 8150 dsv41-xq-v4 --runs 3 --long serve\server.py
```

The script prints the engine's own prompt and decode rates. The DeepSeek engine reports its draft acceptance in its log
(`DSpark N passes, X tokens/pass`), not in the API's `timings`. A `max_tokens` of 400 caps thinking at half the budget,
so each reply becomes two passes; use a larger value for realistic numbers.
