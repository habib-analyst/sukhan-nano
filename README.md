---
language: ur
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - urdu
  - tiny-lm
  - from-scratch
  - educational
---

# Sukhan-Nano

**A tiny Urdu language model, trained completely from scratch — tokenizer,
architecture code, and weights — on a corpus whose licensing is fully
documented.** 5.57M parameters. It is an educational / showcase artifact:
it demonstrates an honest, reproducible end-to-end training run for a
low-resource language, not a competitor to any production Urdu LM.

> سخن (*sukhan*) — Urdu for "speech / eloquence / poetry".

## Headline numbers (all measured from the artifacts in this package)

| | |
|---|---|
| Parameters | 5.57M (decoder-only Transformer, 6 layers, d_model 256, tied embeddings) |
| Tokenizer | Character-level BPE, vocab 3,000, implemented from scratch in this repo (1.90 chars/token on the corpus) |
| Training data | 3,977,932 chars / 2,090,702 tokens — Urdu Wikipedia + Urdu Wikisource, CC BY-SA / public domain (see `DATA.md`) |
| Held-out perplexity | **48.48** on 394,756 chars (207,943 tokens) of never-trained, article-disjoint text |
| Training | 700 steps × 4,096 tokens (~1.4 epochs), AdamW, cosine LR 6e-4 → 6e-5, seed 20261028, CPU-only, ~80 min wall-clock in total (37 min for steps 0-500, 42 min for the resumed steps 500-700; shared, contended VM — an upper bound) |

Perplexity is per BPE token (vocab 3,000, ~1.9 chars/token). For scale:
random guessing over this vocab is perplexity 3,000; a model that learned
nothing about Urdu would sit in the hundreds. 48.48 means the model has
learned real Urdu spelling, vocabulary and some local syntax — and nothing
more. It does **not** write reliable Urdu prose; see the verbatim samples
below, which include its failures.

## Verbatim generation samples

Fixed prompts, fixed seeds, output reproduced exactly by
`scripts/evaluate.py` (full set in `assets/eval_results.json`). Nothing
below was edited or cherry-picked — for each prompt the greedy and the
sampled continuation are both printed as generated.

**Prompt:** پاکستان کا دارالحکومت

- greedy: پاکستان کا دارالحکومت کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے
- sampled (t=0.9, seed 42): پاکستان کا دارالحکومت کے بھی علی جس میں کی الصور کے اس اور علی، پر اس کے پر کو کے حبی ہیں۔ کا کا کی ایک میں کی ہے کی ان

**Prompt:** اردو شاعری میں

- greedy: اردو شاعری میں کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے
- sampled (t=0.9, seed 42): اردو شاعری میں کے بھی علی جس میں کی وہ اس کہ اس اور علی، پر تو کے پر کو کے میں کو میں کا یہ کہ اور کہ آپ کی اس 

**Prompt:** علم حاصل کرنا

- greedy: علم حاصل کرنا کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے
- sampled (t=0.9, seed 42): علم حاصل کرنا کے بھی اس جس اس کی وہ اس کہ اس اور لیے، پر تو کے پر کو کیا میں اس میں کا یہ کہ اور کہ آپ کی اس 

**Prompt:** ایک دفعہ کا ذکر ہے کہ

- greedy: ایک دفعہ کا ذکر ہے کہ کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے
- sampled (t=0.9, seed 42): ایک دفعہ کا ذکر ہے کہ کے بھی اس جس اس کی وہ اس کہ اس اور لیے، پر تو کے پر کو کیا میں اس میں کا یہ کہ اور کہ آپ کہ اس 

**Prompt:** محبت ایک

- greedy: محبت ایک کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے کے
- sampled (t=0.9, seed 42): محبت ایک کے بھی کوئی جس میں کی وہ اس کہ اس اور کہ میں وہ بھی کے کہ کی ایک میں ایک ہی وہ کی ایک وہ کی جب کی جب

## Quickstart

```python
import sys
sys.path.insert(0, "src")
from sukhan_nano.infer import SukhanNano

lm = SukhanNano(".")                      # loads config.json + weights + tokenizer
print(lm.generate_text("پاکستان کا دارالحکومت", max_new_tokens=60, seed=0))
print(lm.perplexity("اردو پاکستان کی قومی زبان ہے۔"))
```

Or the CLI demo: `python3 examples/predict_demo.py "اردو ایک خوبصورت" --tokens 60`

## Architecture

Standard GPT, deliberately unoriginal: pre-norm Transformer blocks,
learned positional embeddings, GELU MLPs, causal self-attention
(`scaled_dot_product_attention`), final LayerNorm, and a language-model
head **tied** to the token embedding. Context 256 tokens. The point of the
artifact is the transparency of the whole pipeline — corpus card, tokenizer
training, training log, held-out evaluation — not architectural novelty.

## Corpus (summary — full card in `DATA.md`)

Only two sources, both Wikimedia dumps fetched 2026-10-28:

- **Urdu Wikipedia** (CC BY-SA 4.0, attribution: Urdu Wikipedia
  contributors) — modern prose; 2,600 articles in the cleaned pool.
- **Urdu Wikisource** (site text CC BY-SA 4.0; underlying works public
  domain — Ghalib d. 1869, Iqbal d. 1938, …) — classical poetry and prose;
  10,859 articles in the cleaned pool.

The cleaned pool (13,459 articles, 38.3M chars) was subsampled by whole
articles, in a seeded shuffle, to the packaged 4.0M-char train / 0.4M-char
validation caps; the split is article-level, so the validation text was
never trained on. Cleaning and subsampling are documented in `DATA.md` and
reproducible via `scripts/build_corpus.py`.

## Training

`scripts/train.py`, PyTorch, CPU (2 cores): batch 16 × 256 tokens,
AdamW (β 0.9/0.95, weight decay 0.1), cosine schedule with 50-step warmup,
gradient clipping 1.0, fixed seed 20261028. Full log: `training_log.txt`.
Tokenizer trained in 34s; the run took ~80 min wall-clock in total (37 min for steps 0-500, 42 min for the resumed steps 500-700; shared, contended VM — an upper bound) wall-clock.

## Limitations — read before using

- **It is tiny and briefly trained.** ~2.9M tokens seen, about 1.4 epochs.
  Output is Urdu-flavoured text with real words and fragments of syntax;
  it drifts, repeats, and invents words. The samples above show this.
- **The corpus skews classical.** Wikisource dominates the article count,
  so the model's Urdu leans literary/poetic and dated; modern informal,
  Roman Urdu, and code-mixed Urdu are essentially absent.
- **Poetry bias in generation:** line breaks and verse-like rhythm appear
  even for prose prompts, inherited from the divan-heavy corpus.
- **No safety tuning of any kind.** It is a base model trained on
  encyclopedia + classical literature text; do not deploy it anywhere
  user-facing.
- **Per-token perplexity is not comparable** to models with different
  tokenizers; the number is reported against this model's own tokenizer
  and this split only.
- Real Urdu language technology (larger corpora, real tokenizers, serious
  evaluation) lives elsewhere — e.g. the `urduhack` toolkit and the
  encoder models under the Urdu NLP community. This package's claim is
  only: *from scratch, openly licensed data, every number reproducible.*

## Files

| File | What |
|---|---|
| `model.safetensors` | trained weights |
| `checkpoint.pt` | step/config/val-loss record of the final eval |
| `config.json` | model hyperparameters |
| `tokenizer.json` | BPE vocab + merge table (the tokenizer itself) |
| `corpus/` | train/val splits + `sources.json` manifest |
| `scripts/` | `build_corpus.py`, `train.py`, `evaluate.py` — the whole pipeline |
| `src/sukhan_nano/` | model, tokenizer, inference code |
| `examples/predict_demo.py` | generation demo |
| `tests/` | tokenizer round-trip, deterministic generation, perplexity sanity |
| `training_log.txt` | the complete training log |
| `assets/eval_results.json` | perplexity + verbatim samples, machine-readable |

## License

Code: MIT (`LICENSE`). Corpus texts: CC BY-SA 4.0 / public domain as
documented in `DATA.md` — **not** MIT. Weights are released for research
and education; see `DATA.md` for the share-alike caveat.
