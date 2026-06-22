"""
Build a clean, single-label English lyrics dataset for genre classification.

Streams the public HF dataset `Yegor25/lyrics_genre_dataset_large` (no token needed),
maps its granular multi-genre tags to a few broad buckets, balances the classes,
cleans the lyrics, and writes data/full/lyrics.csv.

Usage:
    python -m src.build_dataset --per-class 1200
"""
import argparse
import csv
import os
import re

from datasets import load_dataset

# Broad genre buckets. Order matters: the first matching keyword wins.
GENRE_KEYWORDS = {
    "hip-hop": ["hip hop", "hip-hop", "rap", "trap"],
    "country": ["country"],
    "metal": ["metal"],
    "pop": ["pop"],
}

def map_genre(tags):
    joined = " ".join(tags).lower()
    for broad, kws in GENRE_KEYWORDS.items():
        if any(kw in joined for kw in kws):
            return broad
    return None


def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", " ", text)   # remove [Verse], [Chorus], etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(per_class, out_path, max_scan):
    ds = load_dataset(
        "Yegor25/lyrics_genre_dataset_large", split="train", streaming=True
    )

    counts = {g: 0 for g in GENRE_KEYWORDS}
    rows = []
    target_total = per_class * len(GENRE_KEYWORDS)
    scanned = 0

    for ex in ds:
        scanned += 1
        if scanned > max_scan:
            break
        if ex.get("language") != "english":
            continue
        genre = map_genre(ex.get("genre") or [])
        if genre is None or counts[genre] >= per_class:
            continue
        lyrics = clean_lyrics(ex.get("lyrics") or "")
        if len(lyrics) < 100:
            continue
        rows.append((lyrics, genre))
        counts[genre] += 1
        if sum(counts.values()) >= target_total:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lyrics", "genre"])
        w.writerows(rows)

    print(f"Scanned {scanned:,} rows")
    print("Class counts:", counts)
    print(f"Saved {len(rows):,} rows to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=1200,
                   help="max rows to keep per genre")
    p.add_argument("--out", default="data/full/lyrics.csv")
    p.add_argument("--max-scan", type=int, default=400000,
                   help="stop after scanning this many source rows")
    args = p.parse_args()
    main(args.per_class, args.out, args.max_scan)
