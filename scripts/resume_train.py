#!/usr/bin/env python3
"""Resume Sukhan-Nano training from the last evaluated weights.

The original 700-step run was killed by the environment (VM process reaper)
at ~step 580 with no error in the training code; the step-500 weights in
model.safetensors were intact. This script loads those weights and runs the
remaining steps with the *same* cosine LR schedule evaluated at the global
step, so the schedule matches the original plan exactly. The optimizer
state is fresh and the batch sequence differs from an uninterrupted run --
that is disclosed in training_log.txt and REPORT.md rather than hidden.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukhan_nano.model import GPT, GPTConfig          # noqa: E402
from sukhan_nano.tokenizer import BPETokenizer        # noqa: E402
from train import estimate_loss, get_batch, log       # noqa: E402

SEED = 20261028
MAX_STEPS = 700
BATCH = 16
LR = 6e-4


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    ckpt = torch.load(ROOT / "checkpoint.pt", weights_only=True)
    start = int(ckpt["step"])
    log(f"RESUMED from step-{start} weights (original run killed by the "
        f"environment at ~step 580; see REPORT.md). Fresh optimizer state.")

    train_text = (ROOT / "corpus" / "train.txt").read_text(encoding="utf-8")
    val_text = (ROOT / "corpus" / "val.txt").read_text(encoding="utf-8")
    tok = BPETokenizer.load(ROOT / "tokenizer.json")
    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)

    cfg = GPTConfig(**ckpt["config"])
    model = GPT.from_pretrained(ROOT)
    assert model.cfg == cfg
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup, min_lr = 50, LR * 0.1

    def lr_at(step):
        if step < warmup:
            return LR * (step + 1) / warmup
        t = (step - warmup) / max(1, MAX_STEPS - warmup)
        return min_lr + 0.5 * (LR - min_lr) * (1 + math.cos(math.pi * t))

    gen = torch.Generator().manual_seed(SEED + start)
    t_start = time.time()
    best_val = float(ckpt["val_loss"])
    for step in range(start, MAX_STEPS + 1):
        if (step % 50 == 0 and step != start) or step == MAX_STEPS:
            val_loss = estimate_loss(model, val_ids, BATCH, cfg.block_size)
            best_val = min(best_val, val_loss)
            log(f"step {step}/{MAX_STEPS}  val_loss {val_loss:.4f}  "
                f"val_ppl {math.exp(val_loss):.2f}  (resumed run)")
            model.save_pretrained(ROOT)
            torch.save({"step": step, "val_loss": val_loss,
                        "config": cfg.to_dict()}, ROOT / "checkpoint.pt")
            if step == MAX_STEPS:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_ids, BATCH, cfg.block_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log(f"step {step}  train_loss {loss.item():.4f}  lr {lr_at(step):.2e}")

    minutes = (time.time() - t_start) / 60
    log(f"DONE (resumed) steps={MAX_STEPS} best_val_loss={best_val:.4f} "
        f"best_val_ppl={math.exp(best_val):.2f} resume_time={minutes:.1f}m")
    (ROOT / "work" / "train_summary.json").write_text(json.dumps({
        "steps_total": MAX_STEPS, "resumed_from_step": start,
        "best_val_loss": best_val, "best_val_ppl": math.exp(best_val),
        "resume_minutes": minutes, "params": model.num_params(),
        "train_tokens": int(len(train_ids)), "val_tokens": int(len(val_ids)),
        "note": "original run killed by environment at ~step 580; "
                "resumed from step-500 weights, same LR schedule, "
                "fresh optimizer state",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
