# Sukhan-Nano corpus card (DATA.md)

The corpus is part of the artifact: its provenance is the claim. Every text
used for training or validation comes from a Wikimedia dump under an open
license — no scraped books, no unverifiable text collections.

## Sources

| # | Source | What it is | License | Articles kept (cleaned pool) |
|---|--------|-----------|---------|------------------------------|
| 1 | [Urdu Wikisource](https://ur.wikisource.org) full XML dump | Classical Urdu literature: poetry divans, classical prose, religious and historical texts (Ghalib, Iqbal, and other pre-20th-century / early-20th-century authors) | Site text CC BY-SA 4.0; the underlying works are public domain (e.g. Ghalib d. 1869, Iqbal d. 1938 — beyond Pakistan's life+50 copyright term) | 10,859 |
| 2 | [Urdu Wikipedia](https://ur.wikipedia.org) XML dump, streamed prefix | Modern Urdu encyclopedic prose | CC BY-SA 4.0 — attribution: Urdu Wikipedia contributors | 2,600 |

Dump URLs (files as fetched on 2026-10-28; dumps are the "latest" snapshots
published by the Wikimedia Foundation):

- `https://dumps.wikimedia.org/urwikisource/latest/urwikisource-latest-pages-articles.xml.bz2`
- `https://dumps.wikimedia.org/urwiki/latest/urwiki-latest-pages-articles-multistream.xml.bz2`
  (streamed; extraction stopped after 2,600 kept articles — the file itself
  was never downloaded in full)

## Processing (`scripts/build_corpus.py`, deterministic)

1. Parse the MediaWiki XML; keep main-namespace pages (ns=0) only; drop
   redirects.
2. Clean wikitext: strip templates, tables, references, file links, HTML
   and comments; unwrap internal links to their display text; drop bare
   URLs; remove dump encoding artifacts (U+FFFD).
3. Keep a line only if it has ≥ 20 characters and ≥ 60% of its letters are
   in the Arabic/Urdu Unicode block (U+0600–U+06FF); drop articles left
   with < 200 characters.
4. **No script normalization was applied** beyond whitespace collapsing:
   the corpus keeps the sources' own codepoints (Arabic-script quotes,
   diacritics, ZWNJ) as published.
5. Shuffle articles with seed `20261028`; assign 5% of articles to
   validation (article-level split — no article appears in both splits).
6. **Subsample (documented, not hidden):** the cleaned pool is 13,459
   articles / 38.3M characters — far more than a 5.6M-parameter model on a
   2-core CPU can touch in a bounded run. The packaged corpus keeps whole
   articles in the seeded shuffled order up to caps of 4.0M characters
   (train) and 0.4M characters (validation).

## Packaged splits (`corpus/`)

| File | Articles | Characters | Size (UTF-8) |
|------|----------|-----------|--------------|
| `train.txt` | 1,273 | 3,977,932 | 7.06 MB |
| `val.txt` (held out, never trained on) | 161 | 394,756 | 0.70 MB |
| `sources.json` | machine-readable manifest of everything above | | |

## Licensing notes

- The **code** in this repository is MIT (see `LICENSE`).
- The **corpus texts** are *not* MIT: Wikipedia/Wikisource text remains
  CC BY-SA 4.0 (share-alike) with attribution to the respective contributor
  communities, and Wikisource's underlying works are public domain. This
  package redistributes the cleaned splits under those same terms.
- Model weights trained on CC BY-SA text are published for research /
  education. Whether a trained model's weights inherit share-alike
  obligations is an unsettled question; we flag it rather than pretend it
  away, and we do not claim the weights are public domain.
