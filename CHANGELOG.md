# Changelog

## 0.1.0 - 2026-10-28

- Initial release of Sukhan-Nano as a standalone model package.
- From-scratch decoder-only Transformer (~5.6M parameters) and a
  from-scratch character-level BPE tokenizer (vocab 3,000), both implemented
  in this repository with no pretrained weights anywhere in the pipeline.
- Corpus built only from openly licensed sources (Urdu Wikipedia +
  Urdu Wikisource dumps; CC BY-SA / public-domain), with per-source
  documentation in `DATA.md` and a machine-readable manifest in
  `corpus/sources.json`. Packaged corpus is a documented seeded subsample
  of the cleaned pool.
- Training run: fixed seed, CPU-only, logged in `training_log.txt`;
  held-out perplexity measured on a never-trained article-level split and
  reported in `README.md` and `assets/eval_results.json` together with
  verbatim, unedited generation samples.
- Inference API (`SukhanNano.generate_text`, `SukhanNano.perplexity`)
  and a CLI demo (`examples/predict_demo.py`); test suite in `tests/`.
