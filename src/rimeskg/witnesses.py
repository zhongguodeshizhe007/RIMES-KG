"""Exact relation-typed length-2 witness counting, with train-only templates."""
from __future__ import annotations
from collections import Counter,defaultdict
import numpy as np


class TwoHopIndex:
    def __init__(self,graph):
        self.out=defaultdict(lambda:defaultdict(set))
        self.ins=defaultdict(lambda:defaultdict(set))
        for h,r,t in graph:
            h,r,t=map(int,(h,r,t))
            self.out[h][t].add(r)
            self.ins[t][h].add(r)

    def all_pair_counts(self,h,t):
        # Only paths h -r1-> u -r2-> t; direct triples never count.
        left=self.out.get(int(h),{});right=self.ins.get(int(t),{})
        pairs=Counter()
        for u in left.keys()&right.keys():
            for r1 in left[u]:
                for r2 in right[u]:
                    pairs[(r1,r2)]+=1
        return pairs

    def count(self,h,t,allowed):
        if not allowed: return 0
        pairs=self.all_pair_counts(h,t)
        return sum(pairs[p] for p in allowed)


class TemplateBank:
    """The only inputs are TRAIN context and TRAIN label facts. No test triples."""
    def __init__(self,templates): self.templates={int(k):[tuple(map(int,p)) for p in pairs] for k,pairs in templates.items()}

    @classmethod
    def fit(cls,context,labels,n_relations,top_k=8):
        idx=TwoHopIndex(context)
        # Prefer pairs with the highest number of supporting training LABEL queries.
        hist=defaultdict(Counter)
        for h,r,t in labels:
            # train-only label exists outside C by protocol; no query edge leakage.
            hist[int(r)].update(idx.all_pair_counts(h,t))
        bank={r:[p for p,n in hist[r].most_common(top_k)] for r in range(n_relations)}
        return cls(bank)

    def counts(self,queries,graph):
        if len(queries)==0: return np.empty(0,dtype=np.int64)
        heads = set(queries[:, 0].tolist())
        tails = set(queries[:, 2].tolist())
        out = defaultdict(lambda: defaultdict(set))
        ins = defaultdict(lambda: defaultdict(set))
        for h, r, t in graph:
            h, r, t = int(h), int(r), int(t)
            if h in heads: out[h][t].add(r)
            if t in tails: ins[t][h].add(r)
        res = np.zeros(len(queries), dtype=np.int64)
        for i, (h, r, t) in enumerate(queries):
            h, r, t = int(h), int(r), int(t)
            allowed = self.templates.get(r, [])
            if not allowed: continue
            left = out.get(h, {})
            right = ins.get(t, {})
            mids = left.keys() & right.keys()
            if not mids: continue
            pairs = Counter()
            for u in mids:
                for r1 in left[u]:
                    for r2 in right[u]:
                        pairs[(r1, r2)] += 1
            res[i] = sum(pairs[p] for p in allowed)
        return res

    def loss(self,queries,graph,reference_counts):
        current=self.counts(queries,graph)
        support=reference_counts>0
        damage=np.zeros(len(queries),dtype=np.float64)
        damage[support]=np.clip(1-current[support]/reference_counts[support],0,1)
        return damage,support,current

    def describe(self): return {str(k):[list(pair) for pair in v] for k,v in self.templates.items()}
