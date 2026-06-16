"""
Load + clean the lyrics dataset and produce train/val/test splits.

Usage:
    python -m src.data_prep --config config.yaml
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, save_label_map


def clean(df, cfg):
    target = cfg["target"]
    text_col = cfg["data"]["text_col"]

    df = df[[text_col, target]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() >= cfg["data"]["min_chars"]]
    df = df[df[target].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=[text_col])

    # Optional class balancing cap
    cap = cfg["data"].get("max_per_class")
    if cap:
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), cap), random_state=cfg["data"]["seed"]))
        )

    # Drop ultra-rare classes (need >= 3 rows to split)
    counts = df[target].value_counts()
    keep = counts[counts >= 3].index
    df = df[df[target].isin(keep)].reset_index(drop=True)
    return df


def split(df, cfg):
    target = cfg["target"]
    seed = cfg["data"]["seed"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[target], random_state=seed
    )
    return train, val, test


def main(config_path):
    cfg = load_config(config_path)
    target = cfg["target"]

    df = pd.read_csv(cfg["data"]["lyrics_csv"])
    df = clean(df, cfg)

    labels = sorted(df[target].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    df["label"] = df[target].map(label2id)

    train, val, test = split(df, cfg)

    out = "data/processed"
    os.makedirs(out, exist_ok=True)
    train.to_csv(f"{out}/train.csv", index=False)
    val.to_csv(f"{out}/val.csv", index=False)
    test.to_csv(f"{out}/test.csv", index=False)
    save_label_map(label2id, f"{out}/label_map.json")

    print(f"Target: {target}  | classes: {labels}")
    print(f"Rows -> train {len(train)} | val {len(val)} | test {len(test)}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)
