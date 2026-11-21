"""Build Flash-Next's MTP draft (speculative decoding, ~85% accepted on code) the way setup.py does on Linux.

    .venv\\Scripts\\python windows-local\\make_qfn_mtp.py D:\\models     # run from the repo root -> D:\\models\\xolotl-mtp\\rt

Steps: tools/mtp_fetch.py fetch + verify -> tools/mtp_pack.py (experts q2_0) -> tools/mtp_rt.py -> copy data/draft_vocab.bin.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(*args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    if len(sys.argv) != 2 or not Path("tools/mtp_fetch.py").exists():
        sys.exit(__doc__)
    m = Path(sys.argv[1]) / "xolotl-mtp"
    m.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            run("tools/mtp_fetch.py", "fetch", "--out", m)
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(5)
    run("tools/mtp_fetch.py", "verify", "--out", m)
    run("tools/mtp_pack.py", "--src", m, "--experts", "q2_0", "--out", m / "mtp-q2_0.gguf")
    run("tools/mtp_rt.py", "--gguf", m / "mtp-q2_0.gguf", "--out", m / "rt")
    shutil.copy("data/draft_vocab.bin", m / "rt" / "draft_vocab.bin")
    print("OK:", m / "rt")


if __name__ == "__main__":
    main()
