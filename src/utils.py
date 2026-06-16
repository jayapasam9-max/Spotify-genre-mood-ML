"""Shared helpers: config loading and label maps."""
import json
import os
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_label_map(label2id, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(label2id, f, indent=2)


def load_label_map(path):
    with open(path, "r") as f:
        return json.load(f)
