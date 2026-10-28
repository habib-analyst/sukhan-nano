"""Tests for Sukhan-Nano: tokenizer round-trip, model load, deterministic
generation, and perplexity sanity. Run against the packaged weights:

    python3 -m pytest tests/ -q
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukhan_nano.infer import SukhanNano          # noqa: E402
from sukhan_nano.model import GPT, GPTConfig      # noqa: E402
from sukhan_nano.tokenizer import BPETokenizer    # noqa: E402


def test_tokenizer_roundtrip_on_corpus_sample():
    text = (ROOT / "corpus" / "val.txt").read_text(encoding="utf-8")[:5000]
    tok = BPETokenizer.load(ROOT / "tokenizer.json")
    assert tok.decode(tok.encode(text)) == text


def test_tokenizer_trained_from_scratch_roundtrip():
    sample = "اردو زبان پاکستان کی قومی زبان ہے۔ " * 20
    tok = BPETokenizer.train(sample, vocab_size=120, verbose=False)
    assert tok.decode(tok.encode(sample)) == sample


def test_model_loads_and_param_count_in_range():
    lm = SukhanNano(ROOT)
    n = lm.model.num_params()
    assert 5_000_000 <= n <= 15_000_000, n


def test_generation_is_deterministic_with_fixed_seed():
    lm = SukhanNano(ROOT)
    a = lm.generate_text("پاکستان", max_new_tokens=24, temperature=0.9, seed=7)
    b = lm.generate_text("پاکستان", max_new_tokens=24, temperature=0.9, seed=7)
    assert a == b
    assert a.startswith("پاکستان")
    assert len(a) > len("پاکستان")


def test_greedy_generation_deterministic():
    lm = SukhanNano(ROOT)
    a = lm.generate_text("شاعری", max_new_tokens=16, temperature=0, seed=0)
    b = lm.generate_text("شاعری", max_new_tokens=16, temperature=0, seed=0)
    assert a == b


def test_perplexity_sanity_real_text_beats_shuffled():
    lm = SukhanNano(ROOT)
    text = (ROOT / "corpus" / "val.txt").read_text(encoding="utf-8")[2000:5000]
    ppl_real = lm.perplexity(text)
    chars = list(text)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(chars), generator=g).tolist()
    shuffled = "".join(chars[i] for i in perm)
    ppl_shuffled = lm.perplexity(shuffled)
    assert 1.0 < ppl_real < ppl_shuffled


def test_tiny_model_forward_shapes():
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=64)
    model = GPT(cfg)
    x = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 64)
    assert loss.item() > 0


def test_safetensors_roundtrip_with_tied_embeddings(tmp_path):
    # Regression test: safetensors refuses shared (tied) tensors; the save
    # path must store the embedding once and restore the tie on load.
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=64)
    model = GPT(cfg)
    model.save_pretrained(tmp_path)
    loaded = GPT.from_pretrained(tmp_path)
    assert loaded.lm_head.weight is loaded.tok_emb.weight
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.equal(a, b)
