#!/usr/bin/env python3
"""Train Sukhan-Nano from scratch on the packaged corpus.

Fixed seed everywhere; model.safetensors + config.json + tokenizer.json are
(re)written at the package root at every evaluation, so the newest evaluated
weights are always on disk. Full log goes to training_log.txt.

    python3 scripts/train.py --max-steps 1000
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sukhan_nano.model import GPT, GPTConfig          # noqa: E402
from sukhan_nano.tokenizer import BPETokenizer        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261028


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "training_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def get_batch(data, batch_size, block_size, gen):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, batches=24):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size, gen)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    (ROOT / "training_log.txt").write_text("", encoding="utf-8")

    train_text = (ROOT / "corpus" / "train.txt").read_text(encoding="utf-8")
    val_text = (ROOT / "corpus" / "val.txt").read_text(encoding="utf-8")
    log(f"corpus: train {len(train_text)} chars, val {len(val_text)} chars")

    tok_path = ROOT / "tokenizer.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        log(f"tokenizer loaded: vocab {tok.vocab_size}")
    else:
        log(f"training BPE tokenizer (vocab {args.vocab_size}) ...")
        t0 = time.time()
        tok = BPETokenizer.train(train_text, args.vocab_size)
        tok.save(tok_path)
        log(f"tokenizer trained in {time.time() - t0:.0f}s: vocab {tok.vocab_size}")

    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)
    log(f"tokens: train {len(train_ids)}, val {len(val_ids)} "
        f"({len(train_text) / len(train_ids):.2f} chars/token)")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=256,
                    n_layer=6, n_head=8, d_model=256, dropout=0.0)
    model = GPT(cfg)
    log(f"model params: {model.num_params() / 1e6:.2f}M  config: {cfg.to_dict()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    warmup, min_lr = 50, args.lr * 0.1

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.max_steps - warmup)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * t))

    gen = torch.Generator().manual_seed(SEED)
    model.train()
    t_start = time.time()
    best_val = float("inf")
    for step in range(args.max_steps + 1):
        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = estimate_loss(model, val_ids, args.batch_size, cfg.block_size)
            best_val = min(best_val, val_loss)
            elapsed = time.time() - t_start
            log(f"step {step}/{args.max_steps}  val_loss {val_loss:.4f}  "
                f"val_ppl {math.exp(val_loss):.2f}  elapsed {elapsed / 60:.1f}m")
            model.save_pretrained(ROOT)
            torch.save({"step": step, "val_loss": val_loss,
                        "config": cfg.to_dict()},
                       ROOT / "checkpoint.pt")
            if step == args.max_steps:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_ids, args.batch_size, cfg.block_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log(f"step {step}  train_loss {loss.item():.4f}  lr {lr_at(step):.2e}")

    total = time.time() - t_start
    log(f"DONE steps={args.max_steps} best_val_loss={best_val:.4f} "
        f"best_val_ppl={math.exp(best_val):.2f} train_time={total / 60:.1f}m")
    (ROOT / "work" / "train_summary.json").write_text(json.dumps({
        "steps": args.max_steps, "best_val_loss": best_val,
        "best_val_ppl": math.exp(best_val), "train_minutes": total / 60,
        "params": model.num_params(), "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
