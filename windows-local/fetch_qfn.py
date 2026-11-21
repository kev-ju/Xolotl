"""Download Qwen3.8-Flash-Next Swift-1.5 XQ 45 GB (IndrasMirror) for Xolotl's strata engine. Anonymous and resumable.

    python fetch_qfn.py D:\\models          # -> D:\\models\\Qwen3.8-Flash-Next-Swift-1.5-XQ (about 74 GB)

Needs: pip install huggingface_hub hf_xet

After the download it makes two NTFS hard links: the Hub pack's native_experts.txt names the shards
Swift1.5-QFN-XQ-45GB-0000N-of-00002.gguf, not the uploaded names, and strata fails to open them otherwise.
Hard links take no extra space; don't delete them.
"""
import os
import sys
import time
from pathlib import Path

os.environ.pop("HF_TOKEN", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import snapshot_download  # noqa: E402

REPO = "IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored"
UPLOADED = "Qwen3.8-Flash-Next-Swift-1.5-XQ-45GB-Uncensored-0000{n}-of-00002.gguf"
PACK_NAME = "Swift1.5-QFN-XQ-45GB-0000{n}-of-00002.gguf"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    dst = Path(sys.argv[1]) / "Qwen3.8-Flash-Next-Swift-1.5-XQ"
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(REPO, local_dir=dst, token=False, max_workers=8)
            break
        except Exception as e:  # network hiccups: retry, the download resumes
            print(f"attempt {attempt} failed: {e!r}; retrying in 30 s", flush=True)
            time.sleep(30)
    else:
        sys.exit("download failed 5 times")
    print(f"downloaded in {time.time() - t0:.0f} s", flush=True)
    for n in (1, 2):
        link, target = dst / PACK_NAME.format(n=n), dst / UPLOADED.format(n=n)
        if not link.exists():
            os.link(target, link)
            print(f"hard link {link.name} -> {target.name}")
    print("OK:", dst)


if __name__ == "__main__":
    main()
