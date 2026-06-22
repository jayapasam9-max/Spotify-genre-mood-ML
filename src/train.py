"""
Fine-tune DistilBERT for genre/mood classification.

Usage (Colab free GPU recommended):
    python -m src.train --config config.yaml
"""
import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils import load_config, load_label_map


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main(config_path):
    cfg = load_config(config_path)
    text_col = cfg["data"]["text_col"]
    label2id = load_label_map("data/processed/label_map.json")
    id2label = {v: k for k, v in label2id.items()}

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    def load_split(name):
        df = pd.read_csv(f"data/processed/{name}.csv")[[text_col, "label"]]
        return Dataset.from_pandas(df, preserve_index=False)

    ds = {s: load_split(s) for s in ["train", "val", "test"]}

    def tokenize(batch):
        return tok(batch[text_col], truncation=True, max_length=cfg["model"]["max_length"])

    ds = {s: d.map(tokenize, batched=True) for s, d in ds.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["batch_size"],
        learning_rate=float(cfg["train"]["lr"]),
        weight_decay=cfg["train"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=cfg["train"]["logging_steps"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Test set (fine-tuned DistilBERT) ===")
    print(trainer.evaluate(ds["test"]))

    trainer.save_model(cfg["train"]["output_dir"])
    tok.save_pretrained(cfg["train"]["output_dir"])
    print(f"Model saved to {cfg['train']['output_dir']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
