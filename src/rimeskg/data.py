"""Official CoDEx data, stable entity/relation indexing, train-only C/L split.

Important trust boundary: training code accepts only (train, valid) subsets;
TEST is never passed into training or mask sampler. The full truth pool is
created ONLY inside the final evaluation routine.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import requests
from .utils import sha256, digest_array, save_json

ROOT = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples"
OFFICIAL = {"s": (2034, 42, 32888, 1827, 1828), "m": (17050, 51, 185584, 10310, 10311)}


def download_codex(dst, dataset="s", timeout=60):
    if dataset not in OFFICIAL: raise ValueError("Only CoDEx-S or CoDEx-M")
    folder = Path(dst) / f"codex-{dataset}"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        out = folder / f"{split}.txt"
        if not out.exists():
            url = f"{ROOT}/codex-{dataset}/{split}.txt"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.content
            if not content or b"\t" not in content: raise ValueError(f"Unexpected download for {url}")
            (out.with_suffix(".part")).write_bytes(content)
            out.with_suffix(".part").replace(out)
    manifest = {f"{name}.txt": sha256(folder / f"{name}.txt") for name in ("train", "valid", "test")}
    save_json(folder / "download_sha256.json", {"url": ROOT, "files": manifest})
    load_codex(Path(dst), dataset, verify=True)
    return folder


def load_tsv(p):
    lines = Path(p).read_text(encoding="utf8").splitlines()
    rows = [l.split("\t") for l in lines if l.strip()]
    if any(len(x) != 3 for x in rows): raise ValueError(f"Invalid CoDEx triples: {p}")
    return rows


@dataclass
class Splits:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    entities: list[str]
    relations: list[str]
    source: str
    official: bool

    @property
    def n_entities(self): return len(self.entities)
    @property
    def n_relations(self): return len(self.relations)
    def metadata(self):
        return {"source": self.source, "official": self.official,
                "entities": self.n_entities, "relations": self.n_relations,
                "train":len(self.train), "valid":len(self.valid), "test":len(self.test),
                "train_sha256_encoded":digest_array(self.train),
                "valid_sha256_encoded":digest_array(self.valid),
                "test_sha256_encoded":digest_array(self.test)}


def load_codex(root, dataset="s", verify=True):
    folder = Path(root) / f"codex-{dataset}"
    raw = {k:load_tsv(folder / f"{k}.txt") for k in ("train", "valid", "test")}
    ents = sorted({x for triples in raw.values() for h, r, t in triples for x in (h,t)})
    rels = sorted({r for triples in raw.values() for h, r, t in triples})
    em, rm = {x:i for i,x in enumerate(ents)}, {x:i for i,x in enumerate(rels)}
    arr = {k:np.asarray([(em[h],rm[r],em[t]) for h,r,t in triples],dtype=np.int64).reshape(-1,3) for k,triples in raw.items()}
    actual = (len(ents), len(rels), len(arr["train"]), len(arr["valid"]), len(arr["test"]))
    if verify and actual != OFFICIAL[dataset]:
        raise ValueError(f"CoDEx-{dataset.upper()} counts mismatch. Observed {actual}, expected {OFFICIAL[dataset]}. Are these official full splits?")
    splits = Splits(arr["train"],arr["valid"],arr["test"],ents,rels,str(folder),verify)
    if len(set(map(tuple,splits.train))) != len(splits.train): raise AssertionError("Duplicate train triples")
    if set(map(tuple,splits.train)) & set(map(tuple,splits.test)): raise AssertionError("Train/test overlap")
    manifest = splits.metadata()
    manifest["original_file_sha256"] = {k:sha256(folder / f"{k}.txt") for k in raw}
    save_json(folder / "verified_manifest.json", manifest)
    return splits


def split_context_labels(train, fraction=0.25, seed=11):
    """Split unordered endpoint-pair GROUPS, so explicit reciprocal edges stay together.
    No validation/test input. Coverage repair for relations with >=2 groups.
    """
    if not 0 < fraction < 1: raise ValueError("fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    pair_to_idxs = {}
    for i,(h,r,t) in enumerate(train):
        pair = (min(int(h),int(t)),max(int(h),int(t)))
        pair_to_idxs.setdefault(pair,[]).append(i)
    groups = list(pair_to_idxs.values())
    perm = rng.permutation(len(groups))
    state = np.zeros(len(groups),dtype=bool)   # True = L
    acc = 0
    for gi in perm:
        if acc >= round(len(train)*fraction): break
        state[gi] = True
        acc += len(groups[gi])
    rel_groups = {}
    for gi,inds in enumerate(groups):
        for r in set(map(int,train[inds,1])): rel_groups.setdefault(r,[]).append(gi)
    # Greedy group-safe coverage repair (only when relation appears in >=2 distinct endpoint groups).
    for r, candidates in sorted(rel_groups.items()):
        if len(candidates) < 2: continue
        if not any(state[g] for g in candidates):
            state[int(rng.choice([g for g in candidates if not state[g]]))] = True
        if all(state[g] for g in candidates):
            state[int(rng.choice([g for g in candidates if state[g]]))] = False
    l_idx = np.array(sorted(i for gi,inds in enumerate(groups) if state[gi] for i in inds),dtype=np.int64)
    c_idx = np.array(sorted(i for gi,inds in enumerate(groups) if not state[gi] for i in inds),dtype=np.int64)
    C,L = train[c_idx],train[l_idx]
    assert len(C)+len(L)==len(train) and not(set(map(tuple,C)) & set(map(tuple,L)))
    # Group-level audit: no pair shared by C and L, even if swapped or relation changed.
    pairs = lambda X:{(min(int(h),int(t)),max(int(h),int(t))) for h,_,t in X}
    assert not pairs(C)&pairs(L), "Reciprocal/context leakage"
    return C,L


def train_positive_set(C,L):
    return set(map(tuple,np.concatenate([C,L],axis=0)))


def sample_queries(labels, n_entities, positive_train_set, batch_size, negative_count, rng, return_indices=False):
    """Positive + equal candidate negatives for both teacher and student.
    Negative set is based on TRAIN only, never Tva/Tte (open-world assumption).
    """
    chosen_indices=rng.integers(0,len(labels),size=batch_size)
    sample = labels[chosen_indices]
    K=negative_count+1
    cands = np.repeat(sample[:,None,:],K,axis=1)
    for i in range(len(sample)):
        for j in range(1,K):
            for attempt in range(500):
                tr = sample[i].copy()
                side = int(rng.integers(0,2))
                tr[0 if side==0 else 2] = int(rng.integers(n_entities))
                if tuple(tr) not in positive_train_set and not np.array_equal(tr,sample[i]): break
            else: raise RuntimeError("Could not sample a negative from training-only pool")
            cands[i,j] = tr
    return (sample,cands,chosen_indices) if return_indices else (sample,cands)
