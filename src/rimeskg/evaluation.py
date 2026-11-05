"""Head+tail full-entity filtered ranks. Training functions never receive test triples."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch


class FilteredEvaluator:
    def __init__(self,known_truth,n_entities):
        self.n_entities=int(n_entities)
        self.tails=defaultdict(set); self.heads=defaultdict(set)
        for h,r,t in known_truth:
            h,r,t=map(int,(h,r,t))
            self.tails[(h,r)].add(t)
            self.heads[(r,t)].add(h)

    @torch.no_grad()
    def evaluate(self,model,graph,queries,batch_size=96,max_queries=None,graph_off=False,with_per_query=False):
        model.eval()
        if max_queries is not None: queries=queries[:int(max_queries)]
        device=next(model.parameters()).device
        ctx=torch.as_tensor(graph,dtype=torch.long,device=device)
        z=model.encode(ctx,graph_off=graph_off)
        metrics={"MRR":0.,"Hits@1":0.,"Hits@3":0.,"Hits@10":0.,"mean_rank":0.}
        per=[]; n_ranks=0
        for direction in ("head","tail"):
            for start in range(0,len(queries),batch_size):
                block=queries[start:start+batch_size]
                rows=torch.as_tensor(block,dtype=torch.long,device=device)
                scores=model.candidate_scores(rows[:,0],rows[:,1],rows[:,2],z,direction)
                if scores.shape != (len(block),self.n_entities):
                    raise AssertionError("All-entity ranking shape mismatch")
                targets=[]
                for j,(h,r,t) in enumerate(block):
                    h,r,t=map(int,(h,r,t))
                    target=t if direction=="tail" else h
                    known=self.tails[(h,r)] if direction=="tail" else self.heads[(r,t)]
                    others=[v for v in known if v!=target]
                    if others: scores[j,others]=float('-inf')
                    targets.append(target)
                target_idx=torch.as_tensor(targets,device=device,dtype=torch.long)
                gold=scores.gather(1,target_idx[:,None])
                # Average-rank tie policy (consistent for head and tail).
                greater=(scores>gold).sum(dim=1)
                ties=(scores==gold).sum(dim=1)
                ranks=(1+greater).float()+0.5*(ties.float()-1)
                for j,rank in enumerate(ranks.cpu().numpy()):
                    rr=1./float(rank)
                    metrics["MRR"]+=rr
                    for n in (1,3,10): metrics[f"Hits@{n}"]+=float(rank<=n)
                    metrics["mean_rank"]+=float(rank)
                    n_ranks+=1
                    if with_per_query:
                        h,r,t=map(int,block[j])
                        per.append({"q_idx":int(start+j),"direction":direction,"h":h,"r":r,"t":t,
                                    "rank":float(rank),"rr":rr})
        if not n_ranks: raise ValueError("No evaluation queries")
        metrics={k:v/n_ranks for k,v in metrics.items()}
        metrics.update({"n_queries":len(queries),"n_ranks":n_ranks,
                        "evaluation":"head_tail_full_entity_filtered",
                        "tie_policy":"average", "partial":max_queries is not None})
        return metrics,per


def evaluate_validation(model,context,train,valid,n_entities,**kwargs):
    """Validation never reads test; filtering pool is train+valid."""
    pool=FilteredEvaluator(np.concatenate([train,valid]),n_entities)
    return pool.evaluate(model,context,valid,**kwargs)


def evaluate_final(model,context,train,valid,test,n_entities,**kwargs):
    """Call only AFTER model/config/seeds are frozen. Test truth used for filtering only."""
    pool=FilteredEvaluator(np.concatenate([train,valid,test]),n_entities)
    return pool.evaluate(model,context,test,**kwargs)


def relation_macro(rows,query_relations):
    if not rows: return None
    by_relation=defaultdict(list)
    for row in rows: by_relation[row["r"]].append(row["rr"])
    # Both prediction directions per unique query; no query-weighted micro bias.
    return float(np.mean([np.mean(xs) for r,xs in by_relation.items()]))
