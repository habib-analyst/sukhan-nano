"""Load a packaged Sukhan-Nano model and generate Urdu text / score perplexity."""

import math
from pathlib import Path

import torch

from .model import GPT
from .tokenizer import BPETokenizer


class SukhanNano:
    def __init__(self, directory):
        directory = Path(directory)
        self.model = GPT.from_pretrained(directory)
        self.tokenizer = BPETokenizer.load(directory / "tokenizer.json")

    def generate_text(self, prompt, max_new_tokens=80, temperature=0.9, top_k=40, seed=0):
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt produced no tokens")
        gen = torch.Generator().manual_seed(seed)
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(
            x, max_new_tokens, temperature=temperature, top_k=top_k, generator=gen
        )
        return self.tokenizer.decode(out[0].tolist())

    @torch.no_grad()
    def perplexity(self, text, batch_size=16):
        """Mean-token perplexity of `text` under the model (lower = better)."""
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            raise ValueError("text too short to score")
        block = self.model.cfg.block_size
        chunks = []
        for start in range(0, len(ids) - 1, block):
            window = ids[start:start + block + 1]
            if len(window) >= 2:
                chunks.append(window)
        total_nll, total_tokens = 0.0, 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            maxlen = max(len(c) for c in batch)
            x = torch.zeros(len(batch), maxlen - 1, dtype=torch.long)
            y = torch.full((len(batch), maxlen - 1), -100, dtype=torch.long)
            for j, c in enumerate(batch):
                x[j, :len(c) - 1] = torch.tensor(c[:-1])
                y[j, :len(c) - 1] = torch.tensor(c[1:])
            logits, _ = self.model(x)
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            mask = y != -100
            total_nll += (nll * mask).sum().item()
            total_tokens += int(mask.sum())
        return math.exp(total_nll / total_tokens)
