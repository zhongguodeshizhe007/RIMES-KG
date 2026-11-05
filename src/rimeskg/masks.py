"""Pre-registered, fixed-budget train-only masks and stratified matched null."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import jensenshannon


@dataclass
class Mask:
    family: str
    indices: np.ndarray
    seed: int
    k: int
    details: dict = field(default_factory=dict)

    def apply(self, graph):
        keep = np.ones(len(graph),dtype=bool)
        keep[self.indices] = False
        return graph[keep]


def degrees(graph,n_entities):
    h = np.bincount(graph[:,0],minlength=n_entities)
    t = np.bincount(graph[:,2],minlength=n_entities)
    return h,t


def degree_bins(degree):
    return np.searchsorted(np.array([0,1,3,7,15,31,63,127,255]),degree,side="left").astype(np.int16)


def strata(graph,n_entities):
    hd,td = degrees(graph,n_entities)
    hb,tb = degree_bins(hd),degree_bins(td)
    return [(int(r),int(hb[h]),int(tb[t])) for h,r,t in graph]


def generate_mask(graph, family, rho, seed, n_entities, *, relation_exponent=1.0,
                  motif_max_attempt_factor=30, null_source: Mask | None=None):
    if not 0 <= rho <= 1: raise ValueError("rho outside [0,1]")
    N=len(graph); K=round(rho*N)
    rng=np.random.default_rng(seed)
    if K==0: return Mask(family,np.empty(0,dtype=np.int64),seed,0)
    if family == "M1":
        inds=rng.choice(N,size=K,replace=False)
        detail={}
    elif family == "M2":
        counts=np.bincount(graph[:,1]); cnt=counts[graph[:,1]]
        # Explicit relation-conditioned weights: low-frequency relations get more weight.
        weights=np.power(np.maximum(cnt,1),-relation_exponent,dtype=np.float64)
        weights/=weights.sum()
        inds=rng.choice(N,size=K,replace=False,p=weights)
        detail={"relation_exponent":float(relation_exponent),"rule":"frequency^-exponent"}
    elif family == "M3":
        hd,td=degrees(graph,n_entities)
        weights=(1/np.sqrt(hd[graph[:,0]]+1.0) + 1/np.sqrt(td[graph[:,2]]+1.0))
        weights/=weights.sum()
        inds=rng.choice(N,size=K,replace=False,p=weights)
        detail={"rule":"inverse_sqrt_endpoint_degree"}
    elif family == "M4":
        out=defaultdict(list); incoming=defaultdict(list)
        for i,(h,r,t) in enumerate(graph):
            out[int(h)].append(i); incoming[int(t)].append(i)
        mids=[v for v in out if v in incoming]
        if not mids: raise ValueError("Graph has no 2-hop motifs")
        # Sampling midpoints proportional to the number of length-2 walks.
        middle_probs=np.asarray([len(out[m])*len(incoming[m]) for m in mids],dtype=np.float64)
        middle_probs/=middle_probs.sum()
        selected=set(); groups=0; attempts=0
        while len(selected)+2<=K and attempts < motif_max_attempt_factor*K:
            attempts+=1
            mid=int(rng.choice(mids,p=middle_probs))
            i=int(rng.choice(incoming[mid])); j=int(rng.choice(out[mid]))
            if i!=j and i not in selected and j not in selected:
                selected.add(i);selected.add(j);groups+=1
        missing=K-len(selected)
        if missing:
            remaining=np.asarray([i for i in range(N) if i not in selected],dtype=np.int64)
            selected.update(map(int,rng.choice(remaining,size=missing,replace=False)))
        inds=np.fromiter(selected,dtype=np.int64)
        detail={"motif_groups":groups,"grouped_removed_edges":2*groups,
                "residual_single_edges":missing,"grouped_fraction":2*groups/K,"attempts":attempts,
                "pattern":"two consecutive directed edges centered on a training-visible node"}
    elif family=="Null-M4":
        if null_source is None or null_source.family!="M4": raise ValueError("Null-M4 requires an existing M4 mask")
        if null_source.k!=K: raise ValueError("Mismatch K")
        groups=defaultdict(list); wanted=defaultdict(int)
        keys=strata(graph,n_entities)
        for i,key in enumerate(keys): groups[key].append(i)
        for i in null_source.indices: wanted[keys[int(i)]]+=1
        sampled=[]; forced=0
        for key,indices in groups.items():
            count=wanted.get(key,0)
            if count:
                sampled.extend(rng.choice(indices,size=count,replace=False).tolist())
                if count==len(indices): forced+=count
        inds=np.asarray(sampled,dtype=np.int64)
        detail={"matching":"exact (relation, head-degree-bin, tail-degree-bin) removal counts",
                "strata":len(groups),"forced_edges_fraction":forced/K,
                "matched_total":int(len(inds)),"reference_seed":null_source.seed}
        if len(inds)!=K: raise AssertionError("Null-M4 failed exact matching")
        # Distinct grouping is not always possible when strata are singleton/saturated.
    else:
        raise ValueError(f"Unknown mask family: {family}")
    inds=np.sort(np.asarray(inds,dtype=np.int64))
    if len(inds)!=K or len(np.unique(inds))!=K: raise AssertionError("Non-exact deletion budget")
    return Mask(family,inds,seed,K,detail)


def _js(p,q):
    p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    if not p.sum() or not q.sum(): return float('nan')
    return float(jensenshannon(p/p.sum(),q/q.sum(),base=2.0)**2)


def mask_profile(graph,mask,n_entities, *, reference_mask=None, witness_retained=None):
    removed=graph[mask.indices]
    hd,td=degrees(graph,n_entities)
    kept=mask.apply(graph)
    kh,kt=degrees(kept,n_entities)
    full_activity=(hd+td)>0
    isolates=int(np.count_nonzero(full_activity & ((kh+kt)==0)))
    # Two-hop co-removal intensity; compare M4 and stratified null at SAME K.
    removed_indeg=np.bincount(removed[:,2],minlength=n_entities)
    removed_outdeg=np.bincount(removed[:,0],minlength=n_entities)
    joint_removed=float(np.dot(removed_indeg.astype(np.float64),removed_outdeg.astype(np.float64)))
    all_twohop=float(np.dot(td.astype(np.float64),hd.astype(np.float64)))
    joint_fraction=joint_removed/all_twohop if all_twohop>0 else float('nan')
    rho_actual=mask.k/len(graph)
    joint_lift=joint_fraction/(rho_actual*rho_actual) if rho_actual>0 else float('nan')
    rel_n=int(graph[:,1].max())+1
    rel_ref=np.bincount(graph[:,1],minlength=rel_n)
    rel_removed=np.bincount(removed[:,1],minlength=rel_n) if len(removed) else np.zeros(rel_n)
    ref_h_bins=degree_bins(hd[graph[:,0]])
    ref_t_bins=degree_bins(td[graph[:,2]])
    rem_h_bins=degree_bins(hd[removed[:,0]]) if len(removed) else []
    rem_t_bins=degree_bins(td[removed[:,2]]) if len(removed) else []
    nbin=11
    profile={"family":mask.family,"seed":mask.seed,"K":mask.k,"rho_actual":mask.k/len(graph),
       "relation_js_vs_graph":_js(rel_removed,rel_ref),
       "head_degree_js_vs_graph":_js(np.bincount(rem_h_bins,minlength=nbin),np.bincount(ref_h_bins,minlength=nbin)),
       "tail_degree_js_vs_graph":_js(np.bincount(rem_t_bins,minlength=nbin),np.bincount(ref_t_bins,minlength=nbin)),
       "new_isolates":isolates,"active_nodes":int(full_activity.sum()),
       "removed_twohop_pair_count":joint_removed,"removed_twohop_walk_fraction":joint_fraction,
       "removed_twohop_lift_vs_independent_budget":joint_lift,
       **mask.details}
    if reference_mask is not None:
        orig=graph[reference_mask.indices]
        p=np.bincount(orig[:,1],minlength=rel_n)
        profile["relation_js_vs_m4"]=_js(rel_removed,p)
        for end,key in ((0,"head"),(2,"tail")):
            deg=hd if end==0 else td
            a=degree_bins(deg[removed[:,end]]);b=degree_bins(deg[orig[:,end]])
            profile[f"{key}_degree_js_vs_m4"]=_js(np.bincount(a,minlength=nbin),np.bincount(b,minlength=nbin))
        profile["overlap_with_m4_fraction"]=len(set(mask.indices)&set(reference_mask.indices))/mask.k
    if witness_retained is not None: profile["witness_retained_mean"]=float(witness_retained)
    return profile
