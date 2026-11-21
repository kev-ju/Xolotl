"""Write the serve configs for this machine's folders (the ones measured on an RTX 3090 + 94 GB RAM, Windows 11).

    python windows-local\\make_configs.py D:\\models [--arena-gib N] [--cuda-bin DIR]     # from the repo root

Writes:
  <models>\\Qwen3.8-Flash-Next-Swift-1.5-XQ\\xolotl-qfn45-win.json   Flash-Next XQ 45 GB, strata, port 8099
  <models>\\xolotl-dsv41-xq-v4-win.json                            DeepSeek-V4.1 XQ V4, ds41_generate, port 8150
  <models>\\xolotl-dsv41-xq-v4-fast-win.json                       the same + the v4-fast Q2_K sidecar (fast mode)

--arena-gib: DeepSeek's pinned RAM expert cache. Default: installed RAM in GiB - 54, at most 72 (94 GB -> 40 GiB).
The engine holds about arena + 25 GB in all (4 GiB conversation pool, dense weights, staging); Xolotl's own
3090 config uses 72 GiB on a 128 GB machine. With 64 GB of RAM, DeepSeek doesn't fit well; Flash-Next does.

Fast mode's STRATA_SHADOW is a ':'-separated list, so its paths are relative to the repo root (an absolute C:\\ path
would be split at the drive colon): the models must be on the same drive as the repo for fast mode.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path

XOLOTL = Path(__file__).resolve().parent.parent
FAST_FILES = ("V41-XQ-1017-v4.mxfp4-q3k.q2k.im.x41s.g10d42:V41-XQ-1017-v4.mxfp4-q3k.q2k.im.x41s.g42d10:"
              "V41-XQ-1017-v4.mxfp4-q3k.q2k.im.x41s:V41-XQ-1017-v4.mxfp4-q3k.q2k.im.x41s.g10d143:"
              "V41-XQ-1017-v4.mxfp4-q3k.q2k.im.x41s.g143d10")


def ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("total", "avail", "ptotal", "pavail", "vtotal", "vavail", "x")]
    m = MS()
    m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2**30


def p(path):
    return str(path).replace("\\", "/")


def write(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", type=Path)
    ap.add_argument("--arena-gib", type=int)
    ap.add_argument("--cuda-bin", default=os.path.join(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU "
                                                                       r"Computing Toolkit\CUDA\v12.8"), "bin"))
    a = ap.parse_args()
    models = a.models.resolve()
    arena = a.arena_gib or max(16, min(72, round(ram_gib()) - 54))
    serve = (r".venv\Scripts\python -X utf8 tools\v41\serve\serve_ds41.py --engine strata --config <this file> "
             "--port 8150")

    q = models / "Qwen3.8-Flash-Next-Swift-1.5-XQ"
    write(q / "xolotl-qfn45-win.json", {
        "_comment": "Flash-Next XQ 45 GB on strata (build-cuda12). Serve from the repo root: .venv\\Scripts\\python -m "
                    "serve.server --engine strata --config <this file> --port 8099. Ready in ~20 s; leaves ~340 MiB "
                    "of a 24 GB card free.",
        "exe": p(XOLOTL / "build-cuda12" / "strata.exe"),
        "args": ["--pack", p(q / "pack-qfn45"),
                 "--native", p(q / "Qwen3.8-Flash-Next-Swift-1.5-XQ-45GB-Uncensored-00001-of-00002.gguf"),
                 "--ple-gguf", p(q / "Qwen3.8-Flash-Next-Swift-1.5-XQ-45GB-Uncensored-00001-of-00002.gguf"),
                 "--ple-io", "direct",
                 "--expert-profile", p(XOLOTL / "data" / "expert-profile.bin"),
                 "--expert-cache", "auto", "--prefill", "auto",
                 "--spec", "4", "--spec-min-p", "0.5", "--mtp", p(models / "xolotl-mtp" / "rt"),
                 "--max-context", "262144", "--kv", "int8", "--kv-resident", "32768"],
        "env": {"STRATA_ROUTE_RESIDENT": "0.8", "STRATA_ROUTE_RESIDENT_RANKS": "5-9"},
        "lib_dirs": [p(a.cuda_bin)],
        "cwd": p(XOLOTL),
        "tokenizer": p(q / "pack-qfn45" / "tokenizer"),
        "model_name": "qfn45",
        "aliases": "qwen38-flash-next-xq-abl,qwen/qwen3.8-flash",
        "log": p(models / "xolotl-qfn45.log"),
        "port": 8099, "gpu": 0,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "fit_max_tokens": True,
        "reasoning_budget_tokens": 32768,
    })

    xq, engram = models / "dsv41-1017-xq", models / "dsv41-1017-engram"
    ds = {
        "_comment": f"DeepSeek-V4.1-Flash-1017 XQ V4 + the official Engram shards (read in place). From Xolotl's "
                    f"tools/v41/serve/strata-ds41-1017-win.json; arena {arena} GiB in 4 GiB pinned slices and a 4 GiB "
                    f"conversation pool (arena 56 + the default 8 GiB pool left 1.3 GiB free on a 94 GB machine), "
                    f"--tier-batch-pcie 0.25 (the V4 model card). Serve from the repo root: {serve}",
        "exe": p(XOLOTL / "build-win" / "ds41_generate.exe"),
        "args": ["-m", p(xq / "v4" / "V41-XQ-1017-v4.gguf"), "--backend", "cuda", "--experts", "gpu",
                 "--engram", p(engram), "--rank", p(xq / "v4" / "rank-1017-v4.bin"),
                 "--ctx", "262144", "--stop", "1", "--arena-adapt", "--kv-fp4",
                 "--route-bias", "0.5", "--route-bias-file", "0.3", "--arena-gib", str(arena),
                 "--prompt-batch", "12288", "--tier-batch-pcie", "0.25", "--conversation-cache-mib", "4096"],
        "env": {"GGML_NO_BACKTRACE": "1", "STRATA_NO_LARGEPAGES": "1", "STRATA_SERVE_STEP_MAX": "96",
                "STRATA_SERVE_PASS_BIAS": "0",
                "STRATA_SERVE_DSPARK": p(xq / "V41-MTP-Vision-1017.Q2_K.gguf"),
                "STRATA_DSPARK_KEEP": p(XOLOTL / "tools" / "v41" / "serve" / "dspark_expert_counts.txt") + ":64",
                "STRATA_DSPARK_DEPTH": "4", "STRATA_DSPARK_CONF": "0.6", "STRATA_SERVE_LOOKUP": "14",
                "STRATA_SERVE_DSPARK_EVICT": "4096", "STRATA_DENSE_REQUANT": "q6_K", "STRATA_PRESTAGE_GIB": "1.5",
                "STRATA_TEXT_LOG": "0", "STRATA_ARENA_SLICE_GIB": "4"},
        "lib_dirs": [p(a.cuda_bin)],
        "cwd": p(XOLOTL),
        "tokenizer": p(xq / "tokenizer"),
        "model_name": "dsv41-xq-v4",
        "aliases": ["dsv41-xq-v4-exact", "dsv41-xq-v4-light"],
        "log": p(models / "xolotl-dsv41.log"),
        "port": 8150, "host": "127.0.0.1", "gpu": 0,
        "sampling": {"temperature": 1, "top_p": 0.95},
        "reasoning_budget_tokens": 32768,
    }
    write(models / "xolotl-dsv41-xq-v4-win.json", ds)

    fast_dir = xq / "v4-fast"
    lst = fast_dir / "V41-XQ-1017-v4.fast.list"
    names = (lst.read_text(encoding="utf-8").strip() if lst.exists() else FAST_FILES).split(":")
    try:
        rel = p(os.path.relpath(fast_dir, XOLOTL))
    except ValueError:
        print("skipped the fast-mode config: the models are on another drive than the repo (STRATA_SHADOW is a ':' "
              "list, so its paths must be relative)")
        return
    fast = json.loads(json.dumps(ds))
    fast["_comment"] = ("Fast mode: the same as xolotl-dsv41-xq-v4-win.json plus the v4-fast Q2_K sidecar (lossy; the "
                        "model card: +26% decode on agent tasks, same Gauntlet score). The startup log says 'shadow: "
                        "those experts ...' when it is used. " + ds["_comment"])
    fast["env"]["STRATA_SHADOW"] = ":".join(f"{rel}/{n.strip()}" for n in names if n.strip())
    write(models / "xolotl-dsv41-xq-v4-fast-win.json", fast)


if __name__ == "__main__":
    main()
