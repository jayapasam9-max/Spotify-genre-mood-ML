# Data

This folder ships with **small synthetic samples** so the whole pipeline runs
out of the box without any downloads:

- `sample_lyrics.csv` — columns: `lyrics, genre, mood` (~440 rows, 4 genres)
- `sample_charts.csv` — columns: `date, region, rank, track, artist, streams`

The samples are toy data with planted vocabulary — good enough to prove the code
works, **not** good enough for portfolio metrics. Swap in real data for the real run.

## Getting the real (free) datasets

All free, no paid API needed. Download to `data/full/` and update `config.yaml`.

**Lyrics + genre/mood (for fine-tuning):**
- Hugging Face Datasets — search "lyrics" / "genius lyrics" / "music genre lyrics".
  Example: `datasets` library → `load_dataset("...")`.
- Kaggle — "Genius Song Lyrics", "Music Dataset: 1950 to 2019" (has lyrics + genre),
  "MetroLyrics".

Make sure the final CSV has a text column (lyrics) and a label column (`genre` or `mood`).
Set those in `config.yaml`.

**Spotify charts (for the dashboard):**
- Kaggle — "Spotify Charts" (Top 200 daily, multiple regions).
- Spotify audio-features datasets on Kaggle (danceability, energy, valence...) if you
  want to enrich the dashboard or derive mood labels.

> Note on lyrics & copyright: keep full lyrics out of the public repo. Train locally /
> on Colab, commit only the **model**, metrics, and a few short snippets for the demo.
