"""Download DeepSeek-V4.1-Flash-1017 XQ V4 (IndrasMirror) + the official Engram shards for ds41_generate.

    python windows-local\\fetch_dsv41.py D:\\models      # from the repo root; resumable, safe to rerun (about 412 GB)

Needs: pip install huggingface_hub hf_xet

1. IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored @ ed44acec -> <models>\\dsv41-1017-xq
   v4/ (16 parts, 163 GB), v4-fast/ (40 GB, optional fast mode), tokenizer/, V41-MTP-Vision-1017.Q2_K.gguf (DSpark draft).
2. deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 08c8657f -> <models>\\dsv41-1017-engram
   the Engram shards model-00047/00048 (101.5 GB each, read in place), model.safetensors.index.json, config.json.
3. Checks the drafter and the Engram shards against the Hub's LFS sha256 and v4-fast against its SHA256SUMS, joins v4
   with tools/xolotl/win_join_parts.py (it hashes while writing and checks the result against v4/SHA256SUMS) and
   deletes the parts only once the joined file matches. Took 1.6 h at ~70 MB/s.

Streams everything (hashing reads 16 MiB at a time); no model data is held in memory.
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

XOLOTL = Path(__file__).resolve().parent.parent
XQ_REPO, XQ_REV = "IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored", "ed44acec8175a06485c55f0f8915fb57eea9d7ee"
DS_REPO, DS_REV = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "08c8657f02291e11b6f6b63e0e16014f8e748518"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_sums(d, skip):
    """Every file named in d/SHA256SUMS, except the joined model (checked while joining) and its parts."""
    ok = True
    for ln in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        h, name = ln.split(maxsplit=1)
        p = d / name.strip().lstrip("*")
        if p.name == skip or ".part-" in p.name:
            continue
        got = sha256(p)
        log(f"  {'ok ' if got == h else 'BAD'} {p.name}")
        ok &= got == h
    return ok


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    models = Path(sys.argv[1])
    xq, engram = models / "dsv41-1017-xq", models / "dsv41-1017-engram"
    joined = xq / "v4" / "V41-XQ-1017-v4.gguf"
    lfs_sha = {   # the Hub's LFS sha256 at those revisions
        xq / "V41-MTP-Vision-1017.Q2_K.gguf": "5f92242e2ebed06e6831bbcae4e35e679c9138cdf6df0c8b36e8013c99c79a34",
        engram / "model-00047-of-00048.safetensors": "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5",
        engram / "model-00048-of-00048.safetensors": "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd",
    }
    t0 = time.time()
    v4 = ["v4/SHA256SUMS", "v4/rank-*.bin", "v4/*.experts.json", "v4/reassemble.sh"] if joined.exists() else ["v4/*"]
    log(f"1/3 {XQ_REPO} -> {xq}")
    snapshot_download(XQ_REPO, revision=XQ_REV, local_dir=xq, max_workers=8,
                      allow_patterns=v4 + ["v4-fast/*", "tokenizer/*", "V41-MTP-Vision-1017.Q2_K.gguf", "README.md"])
    log(f"2/3 {DS_REPO} -> {engram}")
    snapshot_download(DS_REPO, revision=DS_REV, local_dir=engram, max_workers=8,
                      allow_patterns=["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors",
                                      "model.safetensors.index.json", "config.json"])
    log(f"downloads done in {(time.time() - t0) / 3600:.2f} h")

    log("3/3 checks")
    ok = True
    for p, want in lfs_sha.items():
        got = sha256(p)
        log(f"  {'ok ' if got == want else 'BAD'} {p.name}")
        ok &= got == want
    log("  v4-fast/SHA256SUMS:")
    ok &= check_sums(xq / "v4-fast", joined.name)
    if not joined.exists():
        log("joining v4 (win_join_parts.py checks the joined file against v4/SHA256SUMS)")
        r = subprocess.run([sys.executable, str(XOLOTL / "tools" / "xolotl" / "win_join_parts.py"), str(xq / "v4"),
                            joined.name])
        if r.returncode != 0 or not joined.exists():
            log("JOIN FAILED: the parts are kept")
            return 1
        for p in sorted((xq / "v4").glob(joined.name + ".part-*")):
            p.unlink()
        log("joined file verified; parts deleted")
    log(("ALL OK" if ok else "SOME CHECKS FAILED") + f" in {(time.time() - t0) / 3600:.2f} h")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
