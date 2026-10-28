"""Sukhan-Nano: a tiny decoder-only Transformer for Urdu, from scratch.

Standard GPT architecture (pre-norm blocks, learned positions, tied input /
output embeddings). No novel architecture is claimed -- the artifact's point
is a transparent, fully reproducible from-scratch training run on a
documented, openly licensed Urdu corpus.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 3000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc_proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(att)
        h = self.ln2(x)
        x = x + self.fc_proj(F.gelu(self.fc(h)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=40, generator=None):
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < vals[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    # ---------------------------------------------------------- persistence
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        # lm_head.weight is tied to tok_emb.weight; safetensors rejects
        # shared tensors, so only the embedding copy is stored. Loading
        # restores the tie automatically (same Parameter object).
        state.pop("lm_head.weight", None)
        save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
        (directory / "config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, directory):
        directory = Path(directory)
        cfg = GPTConfig.from_dict(
            json.loads((directory / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(cfg)
        from safetensors.torch import load_file

        # strict=False: lm_head.weight is absent from the file by design --
        # it is tied to tok_emb.weight, which load_state_dict fills in.
        missing, unexpected = model.load_state_dict(
            load_file(directory / "model.safetensors"), strict=False
        )
        assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)
        model.eval()
        return model
