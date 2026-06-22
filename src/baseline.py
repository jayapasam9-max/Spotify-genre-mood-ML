"""
Baseline = TF-IDF + Logistic Regression on the SAME splits.
This is your "before" number to compare the fine-tuned model against.

Usage:
    python -m src.baseline --config config.yaml
"""
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.utils import load_config


def main(config_path):
    cfg = load_config(config_path)
    text_col = cfg["data"]["text_col"]

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(train[text_col], train["label"])
    preds = pipe.predict(test[text_col])

    acc = accuracy_score(test["label"], preds)
    f1 = f1_score(test["label"], preds, average="macro")
    print("=== Baseline (TF-IDF + LogisticRegression) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nUse these as the 'before' numbers in your README.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
