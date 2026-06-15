# 🎵 Spotify Genre & Mood ML

Fine-tune **DistilBERT** to classify song **genre/mood from lyrics**, benchmarked
against a classic TF-IDF baseline — plus an interactive **Spotify charts** analytics
dashboard. End-to-end, reproducible, and **$0** to build (free models, free GPU, free hosting).

<!-- Replace with your own once deployed -->
[![Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/spotify-genre-mood)
[![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://YOUR-APP.streamlit.app)
![Python](https://img.shields.io/badge/python-3.10+-blue)

> ⚠️ Replace every `YOUR_USERNAME` / metric placeholder (`__`) below with your real values.

---

## ✨ What this project demonstrates

- A real **fine-tuning** workflow (not just calling an API): tokenization, `Trainer`, eval.
- **Honest benchmarking**: fine-tuned model vs. a TF-IDF + LogisticRegression baseline.
- **Error analysis**: confusion matrix + the model's most confident mistakes.
- **Productionisation**: a deployed Gradio demo and a Streamlit dashboard.

## 📊 Results

> Run on the **real** dataset, then paste your numbers here. (Sample data is toy data.)

| Model | Accuracy | Macro F1 |
|-------|:--------:|:--------:|
| Baseline (TF-IDF + LogReg) | `__%` | `__` |
| **Fine-tuned DistilBERT**  | **`__%`** | **`__`** |

![Confusion matrix](reports/confusion_matrix.png)

**Error analysis (example):** the model most often confuses `__` with `__`, usually on
short lyrics where vocabulary overlaps. See `reports/worst_errors.csv`.

## 🏗️ Architecture

```
lyrics.csv ──► data_prep ──► train/val/test ──► DistilBERT fine-tune ──► model
                                  │                       │
                                  └──► TF-IDF baseline     └──► evaluate (CM + errors)
                                                                   │
charts.csv ──────────────────────────────────────────► Streamlit dashboard
model ─────────────────────────────────────────────► Gradio demo (HF Spaces)
```

## 🚀 Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/spotify-genre-mood-ml.git
cd spotify-genre-mood-ml
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
