#!/usr/bin/env python3
"""Build the Sukhan-Nano corpus from Wikimedia dumps (all CC BY-SA / public domain).

Sources (see DATA.md for the full corpus card):
  1. Urdu Wikisource full dump  -- classical Urdu literature; site text CC BY-SA,
     underlying works public domain (authors such as Ghalib d.1869, Iqbal d.1938).
  2. Urdu Wikipedia dump (streamed prefix) -- modern Urdu prose, CC BY-SA.

Only main-namespace (ns=0) pages are used. Wikitext is cleaned with a small
deterministic cleaner; lines are kept only if they are predominantly
Urdu-script. Articles are shuffled with a fixed seed and split into
train/validation by whole article (no article appears in both).

Usage:
    python3 scripts/build_corpus.py
Downloads land in work/ (wikisource full file; wikipedia is streamed and
never stored). Outputs land in corpus/.
"""

import bz2
import json
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

USER_AGENT = "SukhanNanoCorpusBuilder/0.1 (research; contact: habib.gcuf.edu@gmail.com)"

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
CORPUS = ROOT / "corpus"
WORK.mkdir(exist_ok=True)
CORPUS.mkdir(exist_ok=True)

WIKISOURCE_URL = (
    "https://dumps.wikimedia.org/urwikisource/latest/"
    "urwikisource-latest-pages-articles.xml.bz2"
)
WIKIPEDIA_URL = (
    "https://dumps.wikimedia.org/urwiki/latest/"
    "urwiki-latest-pages-articles-multistream.xml.bz2"
)
WIKI_TARGET_ARTICLES = 2600          # stop streaming after this many kept articles
SEED = 20261028
VAL_FRACTION = 0.05
# The cleaned pool is ~38M chars (Wikisource holds complete books). A tiny
# model on a 2-core box cannot use it all, so the packaged corpus is a
# documented, article-granular subsample of the pool (seeded order preserved).
MAX_TRAIN_CHARS = 4_000_000
MAX_VAL_CHARS = 400_000

URDU_RE = re.compile(r"[\u0600-\u06FF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------- wikitext --
def strip_nested(text, open_tok, close_tok):
    """Remove balanced open/close spans (templates {{...}}, tables {|...|})."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith(open_tok, i):
            depth += 1
            i += len(open_tok)
        elif depth and text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def clean_wikitext(text):
    if not text:
        return ""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<ref[^>]*/>", " ", text, flags=re.I)
    text = strip_nested(text, "{{", "}}")
    text = strip_nested(text, "{|", "|}")
    # file/image links vanish entirely
    text = re.sub(r"\[\[(?:File|Image|تصویر|فائل)\s*:[^\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)   # [[a|b]] -> b
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)            # [[a]] -> a
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]*\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)                       # leftover html
    text = text.replace("�", "")                               # dump encoding artifacts
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^={1,6}\s*(.*?)\s*={1,6}$", r"\1", text, flags=re.M)
    text = re.sub(r"^\s*[*#;:]+ ?", "", text, flags=re.M)
    return text


def clean_article(raw):
    """Return cleaned text, or '' if the article is not predominantly Urdu prose."""
    text = clean_wikitext(raw)
    paragraphs = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 20:
            continue
        letters = LETTER_RE.findall(line)
        if not letters:
            continue
        urdu = sum(1 for ch in letters if URDU_RE.match(ch))
        if urdu / len(letters) >= 0.6:
            paragraphs.append(line)
    joined = "\n".join(paragraphs)
    return joined if len(joined) >= 200 else ""


# ------------------------------------------------------------------- dumps --
def local(tag):
    return tag.rsplit("}", 1)[-1]


def iter_pages(stream):
    """Yield (title, ns, wikitext) from a MediaWiki XML export stream."""
    title, ns, buf = None, None, None
    for event, elem in ET.iterparse(stream, events=("end",)):
        name = local(elem.tag)
        if name == "title":
            title = elem.text or ""
        elif name == "ns":
            ns = elem.text or ""
        elif name == "text":
            buf = elem.text or ""
        elif name == "page":
            yield title, ns, buf
            title, ns, buf = None, None, None
            elem.clear()


def from_wikisource():
    path = WORK / "urwikisource.xml.bz2"
    if not path.exists():
        print("downloading Urdu Wikisource dump (full, ~10 MB) ...", flush=True)
        # curl, not urllib: this VM's egress proxy rejects Python HTTP clients,
        # and Wikimedia requires a descriptive User-Agent in any case.
        subprocess.run(
            ["curl", "-sL", "--max-time", "600", "-A", USER_AGENT,
             "-o", str(path), WIKISOURCE_URL],
            check=True,
        )
    arts = []
    with bz2.open(path, "rb") as fh:
        for title, ns, raw in iter_pages(fh):
            if ns != "0" or not raw:
                continue
            if raw.lstrip().startswith("#REDIRECT") or raw.lstrip().startswith("#رجوع"):
                continue
            cleaned = clean_article(raw)
            if cleaned:
                arts.append({"title": title, "source": "urwikisource", "text": cleaned})
    return arts


def from_wikipedia():
    print("streaming Urdu Wikipedia dump prefix ...", flush=True)
    proc = subprocess.Popen(
        f"curl -s --max-time 1200 -A '{USER_AGENT}' '{WIKIPEDIA_URL}' | bzcat",
        shell=True, stdout=subprocess.PIPE,
    )
    arts = []
    try:
        for title, ns, raw in iter_pages(proc.stdout):
            if ns != "0" or not raw:
                continue
            if raw.lstrip().startswith("#REDIRECT") or raw.lstrip().startswith("#رجوع"):
                continue
            cleaned = clean_article(raw)
            if cleaned:
                arts.append({"title": title, "source": "urwikipedia", "text": cleaned})
                if len(arts) % 500 == 0:
                    print(f"  kept {len(arts)} wikipedia articles", flush=True)
                if len(arts) >= WIKI_TARGET_ARTICLES:
                    break
    finally:
        proc.terminate()
    return arts


def main():
    articles = from_wikisource()
    print(f"wikisource: {len(articles)} articles kept")
    wiki = from_wikipedia()
    print(f"wikipedia: {len(wiki)} articles kept")
    articles += wiki

    rng = random.Random(SEED)
    rng.shuffle(articles)
    n_val = max(1, int(len(articles) * VAL_FRACTION))
    val_pool, train_pool = articles[:n_val], articles[n_val:]
    pool_stats = {
        "articles": len(articles),
        "chars": sum(len(a["text"]) for a in articles),
    }

    def cap(split, max_chars):
        kept, total = [], 0
        for a in split:
            if kept and total + len(a["text"]) > max_chars:
                break
            kept.append(a)
            total += len(a["text"])
        return kept

    val, train = cap(val_pool, MAX_VAL_CHARS), cap(train_pool, MAX_TRAIN_CHARS)

    def dump(split, name):
        text = "\n\n".join(a["text"] for a in split)
        (CORPUS / name).write_text(text, encoding="utf-8")
        return len(text.encode("utf-8")), len(text)

    stats = {}
    for split, name in ((train, "train.txt"), (val, "val.txt")):
        b, c = dump(split, name)
        stats[name] = {"articles": len(split), "bytes": b, "chars": c}
        print(f"{name}: {len(split)} articles, {b/1e6:.2f} MB, {c} chars")

    manifest = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "sources": [
            {
                "name": "Urdu Wikisource (urwikisource) full dump",
                "url": WIKISOURCE_URL,
                "license": "Site text CC BY-SA 4.0; underlying works public domain",
                "articles_kept": sum(1 for a in articles if a["source"] == "urwikisource"),
            },
            {
                "name": "Urdu Wikipedia (urwiki) dump, streamed prefix",
                "url": WIKIPEDIA_URL,
                "license": "CC BY-SA 4.0 (attribution: Urdu Wikipedia contributors)",
                "articles_kept": sum(1 for a in articles if a["source"] == "urwikipedia"),
            },
        ],
        "splits": stats,
        "pool_before_subsample": pool_stats,
        "subsample_caps_chars": {"train": MAX_TRAIN_CHARS, "val": MAX_VAL_CHARS},
        "titles_sample": [a["title"] for a in articles[:25]],
    }
    (CORPUS / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_mb = sum(s["bytes"] for s in stats.values()) / 1e6
    print(f"TOTAL corpus: {total_mb:.2f} MB")
    if total_mb < 1.0:
        print("ERROR: corpus below the 1 MB hard gate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
