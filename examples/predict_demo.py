#!/usr/bin/env python3
"""Generate Urdu text with Sukhan-Nano.

    python3 examples/predict_demo.py
    python3 examples/predict_demo.py "پاکستان" --tokens 60 --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukhan_nano.infer import SukhanNano  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="اردو ایک خوبصورت")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lm = SukhanNano(ROOT)
    print(lm.generate_text(args.prompt, max_new_tokens=args.tokens,
                           temperature=args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
