#!/usr/bin/env python3
"""Evaluate the packaged Sukhan-Nano and print fixed generation samples.

Writes assets/eval_results.json with the held-out perplexity and the verbatim
samples (selected by a fixed prompt/seed list -- nothing is cherry-picked:
for each prompt one greedy and one sampled continuation, printed as-is).

    python3 scripts/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukhan_nano.infer import SukhanNano  # noqa: E402

PROMPTS = [
    "پاکستان کا دارالحکومت",
    "اردو شاعری میں",
    "علم حاصل کرنا",
    "ایک دفعہ کا ذکر ہے کہ",
    "محبت ایک",
]


def main():
    t0 = time.time()
    lm = SukhanNano(ROOT)
    val_text = (ROOT / "corpus" / "val.txt").read_text(encoding="utf-8")
    ppl = lm.perplexity(val_text)
    print(f"held-out perplexity (full val split, {len(val_text)} chars): {ppl:.2f}")

    samples = []
    for prompt in PROMPTS:
        greedy = lm.generate_text(prompt, max_new_tokens=60, temperature=0, seed=0)
        sampled = lm.generate_text(prompt, max_new_tokens=60, temperature=0.9,
                                   top_k=40, seed=42)
        samples.append({"prompt": prompt, "greedy": greedy, "sampled_t0.9": sampled})
        print("\n=== PROMPT:", prompt)
        print("--- greedy:", greedy)
        print("--- sampled:", sampled)

    results = {
        "held_out_perplexity": round(ppl, 2),
        "val_chars": len(val_text),
        "samples": samples,
        "eval_seconds": round(time.time() - t0, 1),
    }
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "assets" / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nwrote assets/eval_results.json")


if __name__ == "__main__":
    main()
