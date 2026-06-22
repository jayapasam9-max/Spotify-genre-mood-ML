"""
Evaluate the fine-tuned model on the test set:
  - classification report (precision/recall/F1 per class)
  - confusion matrix PNG
  - a CSV of the worst misclassifications for error analysis

Usage:
    python -m src.evaluate --config config.yaml
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import load_config, load_label_map


def predict(texts, model, tok, max_length, device):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = tok(list(batch), truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            probs = model(**enc).logits.softmax(-1).cpu().numpy()
            preds.extend(probs.argmax(-1))
            confs.extend(probs.max(-1))
    return np.array(preds), np.array(confs)


def main(config_path):
    cfg = load_config(config_path)
    text_col = cfg["data"]["text_col"]
    model_dir = cfg["train"]["output_dir"]

    label2id = load_label_map("data/processed/label_map.json")
    id2label = {v: k for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(len(id2label))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    test = pd.read_csv("data/processed/test.csv")
    y_true = test["label"].values
    y_pred, conf = predict(test[text_col].values, model, tok,
                           cfg["model"]["max_length"], device)

    os.makedirs("reports", exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {cfg['target']}")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    print("Saved reports/confusion_matrix.png")

    # Error analysis: confident but wrong
    test = test.copy()
    test["pred"] = [id2label[p] for p in y_pred]
    test["confidence"] = conf
    wrong = test[test["label"] != y_pred].sort_values("confidence", ascending=False)
    wrong[[text_col, cfg["target"], "pred", "confidence"]].head(50).to_csv(
        "reports/worst_errors.csv", index=False
    )
    print("Saved reports/worst_errors.csv (review these in your README error analysis)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
