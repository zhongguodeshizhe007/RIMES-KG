"""Independent P1 retraining and fixed-L P2 with KL-mass-normalized consistency."""
from __future__ import annotations
from dataclasses import dataclass
import math, time
import numpy as np
import torch
from torch.nn import functional as F
from .models import KGCModel
from .masks import generate_mask,degrees,degree_bins
from .data import train_positive_set,sample_queries
from .witnesses import TemplateBank
from .utils import seed_all


@dataclass
class TrainSettings:
    dim:int=24
    layers:int=1
    bases:int=4
    steps:int=100
    batch_size:int=48
    negatives:int=16
    lr:float=0.005
    weight_decay:float=0.0001
    rho:float=0.20
    lambda_kl:float=0.7
    temperature:float=2.0
    gate_temperature:float=0.5
    gate_delta:float=0.0
    warmup_steps:int=20
    top_templates:int=8
    clip_grad:float=5.0
    max_weight:float=1.0
    dropout:float=0.0
    device:str="cpu"
    def __post_init__(self):
        if self.steps<1 or self.batch_size<1 or self.negatives<1: raise ValueError("Invalid run settings")
        if self.rho<=0 or self.rho>=1: raise ValueError("rho must be between 0 and 1")
        if self.dropout!=0: raise ValueError("Main reproducible teacher/student experiment fixes dropout=0; otherwise model modes differ")


def _tensor(arr,device): return torch.as_tensor(arr,dtype=torch.long,device=device)


def train_p1(observed,n_entities,n_relations,settings,seed=11,decoder="complex"):
    """One P1 condition: a fresh model, trained ONLY on observed training edges."""
    seed_all(seed)
    model=KGCModel(n_entities,n_relations,settings.dim,encoder="static",decoder=decoder).to(settings.device)
    opt=torch.optim.Adam(model.parameters(),lr=settings.lr,weight_decay=settings.weight_decay)
    pos=set(map(tuple,observed))
    rng=np.random.default_rng(seed+913)
    trace=[];t0=time.perf_counter()
    for step in range(settings.steps):
        _,cands=sample_queries(observed,n_entities,pos,settings.batch_size,settings.negatives,rng)
        c=_tensor(cands,settings.device)
        logits=model.score(c,model.encode(None))
        loss=F.cross_entropy(logits,torch.zeros(len(c),dtype=torch.long,device=settings.device))
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),settings.clip_grad)
        opt.step()
        if step%max(1,settings.steps//10)==0 or step==settings.steps-1:
            trace.append({"step":step,"rank_loss":float(loss.item()),"grad_norm":float(gn)})
    return model,{"steps":settings.steps,"train_time_s":time.perf_counter()-t0,"training_triples":len(observed),"trace":trace}


def consistency_weights(method,damage,support,gate,queries,C,n_entities,rng,max_weight=1.0,shuffled_damage=None):
    """Same eligibility (W0>0 and d>0) for all evidence-weighted KL controls.
    Each method is normalized by the total batch weight. Teacher gate and
    weight distribution effects are exposed as separate baselines.
    """
    active=support & (damage>0)
    if method in ("erm","dropedge","relation_dropedge","group_dro","graph_off"):
        return np.zeros(len(queries),dtype=np.float32)
    g=gate.detach().cpu().numpy().astype(np.float32)
    d=np.clip(damage,0,max_weight).astype(np.float32)
    if method=="uniform_full": raw=np.ones(len(queries),dtype=np.float32)
    elif method=="uniform": raw=active.astype(np.float32)
    elif method=="gate_only": raw=active.astype(np.float32)*g
    elif method=="rimes": raw=d*g*active
    elif method=="shuffled_d":
        # Full train-L stratum shuffle passed by the caller. The small-batch fallback
        # exists for isolated unit tests; it is NOT used in P2 training.
        if shuffled_damage is not None:
            shuffled=np.clip(np.asarray(shuffled_damage),0,max_weight)
        else:
            hd,td=degrees(C,n_entities);hb=degree_bins(hd);tb=degree_bins(td)
            by={}
            for i,(h,r,t) in enumerate(queries):
                if active[i]:by.setdefault((int(r),int(hb[h]),int(tb[t])),[]).append(i)
            shuffled=d.copy()
            for ids in by.values():
                if len(ids)>1:shuffled[ids]=d[rng.permutation(ids)]
        raw=shuffled*g*active
    elif method=="degree":
        hd,td=degrees(C,n_entities)
        degree_score=np.array([1/np.sqrt(1+hd[h]+td[t]) for h,r,t in queries],dtype=np.float32)
        raw=degree_score*g*active
    elif method=="relation":
        freqs=np.bincount(C[:,1],minlength=int(max(C[:,1].max(),queries[:,1].max())+1))
        rel_score=np.asarray([1/np.sqrt(1+freqs[int(r)]) for _,r,_ in queries],dtype=np.float32)
        raw=rel_score*g*active
    else: raise ValueError(f"Unknown P2 method: {method}")
    return np.asarray(raw,dtype=np.float32)


def normalized_consistency(teacher_logits,student_logits,w,temperature):
    """Temperature-squared KL with exact weight-mass normalization.
    Returns zero iff no valid positive weight.
    """
    weights=torch.as_tensor(w,dtype=student_logits.dtype,device=student_logits.device)
    if not bool((weights>0).any()):return student_logits.sum()*0.0
    pt=F.softmax(teacher_logits.detach()/temperature,dim=-1)
    logps=F.log_softmax(student_logits/temperature,dim=-1)
    per=F.kl_div(logps,pt,reduction="none").sum(dim=-1)
    return temperature**2 * torch.sum(weights*per)/weights.sum().clamp_min(1.e-12)



def shuffle_damage_global(labels,C,n_entities,damage,support,seed):
    """Permute POSITIVE damage values among all active TRAIN labels in matched strata.

    Matching exact (relation, original-head-degree-bin, original-tail-degree-bin)
    prevents a minibatch containing singleton strata from silently disabling
    the negative control. No validation/test queries are used.
    """
    hdeg,tdeg=degrees(C,n_entities);hb=degree_bins(hdeg);tb=degree_bins(tdeg)
    eligible=np.asarray(support,dtype=bool)&(np.asarray(damage)>0)
    by={}
    for i,(h,r,t) in enumerate(labels):
        if eligible[i]:by.setdefault((int(r),int(hb[h]),int(tb[t])),[]).append(i)
    rng=np.random.default_rng(seed)
    donor=np.arange(len(labels))
    for ids in by.values():
        if len(ids)>1:donor[ids]=rng.permutation(ids)
    shuffled=np.asarray(damage)[donor]
    assert np.count_nonzero(shuffled)>0 or not eligible.any()
    return shuffled, float(np.mean((donor!=np.arange(len(labels)))[eligible])) if eligible.any() else 0.

def train_p2(C,L,n_entities,n_relations,settings,seed=11,method="rimes",mask_families=("M1","M2"),
             *, templates=None, graph_off=False):
    """All supervised positives are L. All graph contexts C_m subset C.
    NO validation or test triples are accepted by this function.
    """
    if method not in ("erm","dropedge","relation_dropedge","uniform","uniform_full","gate_only","degree","relation","shuffled_d","group_dro","rimes","graph_off"):
        raise ValueError(method)
    seed_all(seed)
    model=KGCModel(n_entities,n_relations,settings.dim,encoder="rgcn",decoder="complex",
                   layers=settings.layers,bases=settings.bases,dropout=settings.dropout).to(settings.device)
    opt=torch.optim.Adam(model.parameters(),lr=settings.lr,weight_decay=settings.weight_decay)
    if templates is None: templates=TemplateBank.fit(C,L,n_relations,settings.top_templates)
    W0=templates.counts(L,C)
    pos=train_positive_set(C,L)
    t0=time.perf_counter(); history=[];mean_grad=[]
    # Fixed scheduler streams for all methods: same seed -> same masks, samples & negatives.
    rng=np.random.default_rng(seed+177)
    group_weights=np.ones(n_relations,dtype=np.float64)/n_relations
    for step in range(settings.steps):
        _,cands,batch_idxs=sample_queries(L,n_entities,pos,settings.batch_size,settings.negatives,rng,return_indices=True)
        # A separate mask RNG seed is independent of masks generated on other steps.
        # Uniform DropEdge uses only M1; relation DropEdge and KL arms share
        # alternating M1/M2. This keeps the augmentation-type control meaningful.
        family="M1" if method=="dropedge" else mask_families[step%len(mask_families)]
        base_mask=generate_mask(C,family,settings.rho,seed*1000003+step,n_entities)
        C_m=base_mask.apply(C)
        q=cands[:,0,:]
        if method=="erm": C_m=C
        c=_tensor(cands,settings.device)
        teacher=None
        if method not in ("erm","dropedge","relation_dropedge","group_dro","graph_off"):
            model.eval()
            with torch.no_grad():
                zt=model.encode(_tensor(C,settings.device))
                teacher=model.score(c,zt)
                margin=teacher[:,0]-teacher[:,1:].max(dim=-1).values
                gate=torch.sigmoid((margin-settings.gate_delta)/settings.gate_temperature).detach()
        model.train()
        z=model.encode(_tensor(C_m,settings.device),graph_off=(method=="graph_off" or graph_off))
        logits=model.score(c,z)
        per_rank=F.cross_entropy(logits,torch.zeros(len(c),device=settings.device,dtype=torch.long),reduction="none")
        if method=="group_dro":
            # Group DRO across training-label relations, exponential-weight updates.
            qs=q[:,1].astype(int)
            g_losses=[];g_ids=[]
            for rel in np.unique(qs):
                g_ids.append(int(rel));g_losses.append(per_rank[torch.as_tensor(qs==rel,device=settings.device)].mean())
            if g_ids:
                for rel,val in zip(g_ids,g_losses):group_weights[rel]*=np.exp(0.03*min(float(val.detach().item()),30))
                group_weights/=group_weights.sum()
                norm=sum(group_weights[x] for x in g_ids)
                rank_loss=sum(float(group_weights[x]/norm)*val for x,val in zip(g_ids,g_losses))
            else:rank_loss=per_rank.mean()
        else:rank_loss=per_rank.mean()
        loss=rank_loss;kl_val=0.0;w_mass=0.0;n_active=0
        if teacher is not None:
            # Exact W count for q in the current context, not the full L matrix cache.
            wref=W0[batch_idxs]
            damage,support,_=templates.loss(q,C_m,wref)
            global_shuffled=None;shuffled_fraction=None
            if method=="shuffled_d":
                all_damage,all_support,_=templates.loss(L,C_m,W0)
                shuffled_all,shuffled_fraction=shuffle_damage_global(
                    L,C,n_entities,all_damage,all_support,seed*1000003+step+997)
                global_shuffled=shuffled_all[batch_idxs]
            weights=consistency_weights(method,damage,support,gate,q,C,n_entities,
                                        np.random.default_rng(seed*1000003+step+997),settings.max_weight,
                                        shuffled_damage=global_shuffled)
            kl=normalized_consistency(teacher,logits,weights,settings.temperature)
            warmup=min(1.,(step+1)/max(1,settings.warmup_steps))
            loss=rank_loss+settings.lambda_kl*warmup*kl
            kl_val=float(kl.detach().item());w_mass=float(weights.sum());n_active=int(np.count_nonzero(weights))
        opt.zero_grad(set_to_none=True);loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings.clip_grad)
        opt.step()
        mean_grad.append(float(grad_norm))
        if step % max(1,settings.steps//10)==0 or step==settings.steps-1:
            history.append({"step":step,"family":family,"loss":float(loss.detach()),
                            "rank_loss":float(rank_loss.detach()),"kl_loss":kl_val,
                            "weight_mass":w_mass,"effective_fraction":n_active/len(c),
                            "global_shuffled_fraction":shuffled_fraction if teacher is not None and method=="shuffled_d" else None,
                            "grad_norm":float(grad_norm)})
    info={"method":method,"steps":settings.steps,"train_time_s":time.perf_counter()-t0,
          "mean_grad_norm":float(np.mean(mean_grad)),"train_labels":len(L),"context_edges":len(C),
          "num_support_labels":int(np.count_nonzero(W0)),"support_fraction":float(np.mean(W0>0)),
          "history":history,"templates":templates.describe(),
          "scope":"Train-only C/L; no val/test access; dropout disabled"}
    return model,info,templates
