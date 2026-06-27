# 🎵 Spotify Genre & Mood ML

Fine-tune **DistilBERT** to classify song **genre/mood from lyrics**, benchmarked
against a classic TF-IDF baseline — plus an interactive **Spotify charts** analytics
dashboard. End-to-end, reproducible, and **$0** to build (free models, free GPU, free hosting).

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/Jayasimha01/spotify-genre-mood-demo)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Model](https://img.shields.io/badge/model-DistilBERT-orange)

**▶️ Try it live:** [paste lyrics → get a genre prediction](https://huggingface.co/spaces/Jayasimha01/spotify-genre-mood-demo)

---

## ✨ What this project demonstrates

- A real **fine-tuning** workflow (not just calling an API): tokenization, `Trainer`, eval.
- **Honest benchmarking**: fine-tuned model vs. a TF-IDF + LogisticRegression baseline.
- **Error analysis**: confusion matrix + the model's most confident mistakes.
- **Productionisation**: a deployed Gradio demo and a Streamlit dashboard.

## 📊 Results

Classifying **4 genres** (hip-hop, country, metal, pop) from lyrics alone, on a held-out
test set of 1,600 songs. Random guessing would be 25%.

| Model | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| Baseline (TF-IDF + Logistic Regression) | 64.0% | 0.637 |
| **Fine-tuned DistilBERT** | **68.3%** | **0.686** |

DistilBERT improves on a strong TF-IDF baseline by ~4 points. The takeaway: bag-of-words
is surprisingly competitive for lyrics, but a fine-tuned transformer still adds a
meaningful, consistent gain.

**Error analysis:** hip-hop and country are the most reliably classified genres, thanks to
their distinctive vocabularies. **Pop is the hardest class** — country, metal, and pop
songs are frequently pulled toward "pop" because it is the broadest, least lexically
distinct category. The largest confusions are metal→pop and country→pop. The model's most
confident mistakes are saved to `reports/worst_errors.csv`.

## 🏗️ Architecture

```
lyrics.csv ──► data_prep ──► train/val/test ──► DistilBERT fine-tune ──► model
                                  │                       │
                                  └──► TF-IDF baseline     └──► evaluate (CM + errors)
                                                                   │
charts.csv ──────────────────────────────────────────► Streamlit dashboard
model ────────────────────────────────────────────► Gradio demo (HF Spaces)
```

## 🚀 Quickstart

```bash
git clone https://github.com/jayapasam9-max/Spotify-genre-mood-ML.git
cd Spotify-genre-mood-ML
pip install -r requirements.txt

# 1) Prepare data (uses bundled sample by default)
python -m src.data_prep --config config.yaml

# 2) Baseline ("before" numbers)
python -m src.baseline --config config.yaml

# 3) Fine-tune DistilBERT (use Colab free GPU for the real dataset)
python -m src.train --config config.yaml

# 4) Evaluate: confusion matrix + error analysis
python -m src.evaluate --config config.yaml

# 5) Demos
python app/gradio_app.py            # classifier demo
streamlit run app/dashboard.py      # charts dashboard
```

Switch between predicting **genre** and **mood** with one line in `config.yaml` (`target:`).

## 🆓 Going free, end to end

| Need | Free option |
|------|-------------|
| GPU for training | Google Colab (open `notebooks/train_colab.ipynb`) |
| Model | `distilbert-base-uncased` (no API key) |
| Datasets | Kaggle / Hugging Face (see `data/README.md`) |
| Demo hosting | Hugging Face Spaces (Gradio) |
| Dashboard hosting | Streamlit Community Cloud |

## 📁 Structure

```
spotify-genre-mood-ml/
├── config.yaml            # one place to change task / model / training
├── data/                  # bundled samples + how to get real data
├── src/                   # data_prep, baseline, train, evaluate, utils
├── app/                   # gradio_app.py (classifier), dashboard.py (charts)
├── notebooks/             # train_colab.ipynb (free GPU)
└── reports/               # generated: confusion matrix, error CSV
```

## 🗺️ 7-day plan

1. Get real lyrics + charts datasets (`data/README.md`)
2. `data_prep` + explore class balance
3. Run `baseline` → record "before"
4. `train` on Colab GPU
5. `evaluate` → confusion matrix + error analysis
6. Deploy Gradio demo + Streamlit dashboard
7. Fill in this README (results, demo GIF, screenshots)

## 📝 License

MIT. Lyrics datasets keep their own licenses — don't commit full lyrics; see `data/README.md`.
