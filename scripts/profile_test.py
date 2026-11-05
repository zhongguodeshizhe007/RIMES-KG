import time
import torch
from rimeskg.data import load_codex, split_context_labels
from rimeskg.training import TrainSettings, train_p1, train_p2
from rimeskg.evaluation import evaluate_final

print("Loading CoDEx-S...")
splits = load_codex("data", "s", verify=True)
st = TrainSettings(dim=64, layers=2, bases=4, steps=1500, batch_size=96, negatives=32, device="cuda")

print("Testing P1...")
t0 = time.perf_counter()
model_p1, log_p1 = train_p1(splits.train, splits.n_entities, splits.n_relations, st, seed=11, decoder="complex")
t_p1_train = time.perf_counter() - t0

t0 = time.perf_counter()
m_p1, _ = evaluate_final(model_p1, splits.train, splits.train, splits.valid, splits.test, splits.n_entities)
t_p1_eval = time.perf_counter() - t0
print(f"P1 train time: {t_p1_train:.2f}s, eval time: {t_p1_eval:.2f}s, test MRR: {m_p1['MRR']:.4f}")

print("Testing P2 ERM...")
C, L = split_context_labels(splits.train, 0.25, seed=123)
t0 = time.perf_counter()
model_p2, log_p2, _ = train_p2(C, L, splits.n_entities, splits.n_relations, st, seed=11, method="erm")
t_p2_train = time.perf_counter() - t0

t0 = time.perf_counter()
m_p2, _ = evaluate_final(model_p2, C, splits.train, splits.valid, splits.test, splits.n_entities)
t_p2_eval = time.perf_counter() - t0
print(f"P2 ERM train time: {t_p2_train:.2f}s, eval time: {t_p2_eval:.2f}s, test MRR: {m_p2['MRR']:.4f}")

print("Testing P2 RIMES...")
t0 = time.perf_counter()
model_rimes, log_rimes, _ = train_p2(C, L, splits.n_entities, splits.n_relations, st, seed=11, method="rimes")
t_rimes_train = time.perf_counter() - t0
print(f"P2 RIMES train time: {t_rimes_train:.2f}s")
