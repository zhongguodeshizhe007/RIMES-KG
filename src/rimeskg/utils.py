from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch


def seed_all(seed: int):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f: json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


def read_json(path):
    with open(path, encoding="utf8") as f: return json.load(f)


def append_jsonl(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf8") as f: f.write(json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")


def assert_only_train_graph(graph, splits):
    s = set(map(tuple, splits.train))
    if not set(map(tuple, graph)).issubset(s):
        raise AssertionError("Message graph contains triples not in TRAIN (possible leakage)")
