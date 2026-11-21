"""Quick speed check of a local Xolotl server (OpenAI API), from the engine's own `timings`.

    python speed_test.py PORT MODEL [--long FILE] [--runs N]

Short test: N greedy code requests, max_tokens 400 (decode tok/s, draft acceptance).
Long test: the first ~52 KB of FILE (about 13K tokens) plus a one-paragraph question (prompt read tok/s, then decode).
Thinking tokens count as decoded tokens, as the server's own speed figure does.
"""
import argparse
import json
import time
import urllib.request

CODE_PROMPTS = [
    "Write a Python function that parses an INI file into a dict of dicts without using configparser. Code only.",
    "Write a C function that reverses a singly linked list in place, with a short test in main(). Code only.",
    "Write a SystemVerilog module for a parameterized synchronous FIFO with full/empty flags. Code only.",
    "Write a Rust function that merges two sorted Vec<i64> into one sorted Vec<i64>. Code only.",
]


def ask(port, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    wall = time.time() - t0
    t = out.get("timings") or {}
    u = out.get("usage") or {}
    acc = (f"{t['draft_n_accepted']}/{t['draft_n']} drafts"
           if t.get("draft_n") else "no drafts")
    print(f"  prompt {u.get('prompt_tokens')} tok (read {t.get('prompt_n')}) at {t.get('prompt_per_second')} tok/s; "
          f"decode {u.get('completion_tokens')} tok at {t.get('predicted_per_second')} tok/s ({acc}); "
          f"wall {wall:.1f} s; finish {out['choices'][0].get('finish_reason')}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("model")
    ap.add_argument("--long", help="a text file whose first ~52 KB make the long prompt")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    print(f"short: {a.runs} greedy code requests, max_tokens 400")
    rates = []
    for i in range(a.runs):
        t = ask(a.port, a.model, CODE_PROMPTS[i % len(CODE_PROMPTS)], 400)
        if t.get("predicted_per_second"):
            rates.append(t["predicted_per_second"])
    if rates:
        print(f"  decode tok/s: min {min(rates)}, max {max(rates)}")
    if a.long:
        text = open(a.long, encoding="utf-8", errors="replace").read()[:52000]
        print(f"long: {len(text)} chars of {a.long}, max_tokens 300")
        ask(a.port, a.model, text + "\n\nIn one paragraph, what does this code do?", 300)


if __name__ == "__main__":
    main()
