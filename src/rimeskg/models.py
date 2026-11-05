"""Static KGE (P1) and explicitly input-conditioned relational GNN (P2)."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def complex_score(z,relations,triples,d):
    h,r,t=triples[...,0],triples[...,1],triples[...,2]
    hr,hi=z[h].split(d,dim=-1)
    tr,ti=z[t].split(d,dim=-1)
    rr,ri=relations[r].split(d,dim=-1)
    return ((hr*rr-hi*ri)*tr + (hr*ri+hi*rr)*ti).sum(-1)


def distmult_score(z,relations,triples):
    h,r,t=triples[...,0],triples[...,1],triples[...,2]
    return (z[h]*relations[r]*z[t]).sum(-1)


class RelationalConv(nn.Module):
    """R-GCN-style 2R directional typed basis messages, destination-degree normalized."""
    def __init__(self,dim,n_relations,num_bases=4):
        super().__init__()
        self.bases=nn.Parameter(torch.empty(num_bases,dim,dim))
        self.coeffs=nn.Parameter(torch.empty(2*n_relations,num_bases))
        self.self_layer=nn.Linear(dim,dim,bias=True)
        self.norm=nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeffs)

    def forward(self,x,graph,n_relations):
        n=len(x)
        if len(graph)==0: return self.norm(self.self_layer(x))
        h,r,t=graph.unbind(-1)
        src=torch.cat((h,t)); dst=torch.cat((t,h))
        types=torch.cat((r,r+n_relations))
        # Basis factorization: x @ sum_b(a[rel,b] * B[b]).
        # Transform node embeddings ONCE per basis (not once per edge), then scatter.
        # This is both memory-light on CoDEx-M and differentiable w.r.t. bases/coeffs.
        aggr=torch.zeros_like(x)
        for b in range(len(self.bases)):
            transformed=x@self.bases[b]
            msg=transformed[src]*self.coeffs[types,b].unsqueeze(-1)
            aggr=aggr.index_add(0,dst,msg)
        deg=torch.bincount(dst,minlength=n).to(device=x.device,dtype=x.dtype).clamp_min(1).unsqueeze(1)
        return self.norm(self.self_layer(x)+aggr/deg)


class KGCModel(nn.Module):
    def __init__(self,n_entities,n_relations,dim=32,encoder="rgcn",decoder="complex",layers=1,bases=4,dropout=0.0):
        super().__init__()
        if encoder not in ("rgcn","static"): raise ValueError(encoder)
        if decoder not in ("complex","distmult"): raise ValueError(decoder)
        self.n_entities=n_entities;self.n_relations=n_relations
        self.dim=dim;self.decoder=decoder;self.encoder=encoder
        self.feature_dim=dim*2 if decoder=="complex" else dim
        self.entities=nn.Embedding(n_entities,self.feature_dim)
        self.relations=nn.Embedding(n_relations,self.feature_dim)
        self.convs=nn.ModuleList([RelationalConv(self.feature_dim,n_relations,bases) for _ in range(layers if encoder=="rgcn" else 0)])
        self.dropout=float(dropout)
        nn.init.normal_(self.entities.weight,std=0.08)
        nn.init.normal_(self.relations.weight,std=0.08)

    def encode(self,graph=None,graph_off=False):
        x=self.entities.weight
        if self.encoder=="rgcn" and not graph_off:
            if graph is None: raise ValueError("R-GCN requires graph input")
            for conv in self.convs:
                x=F.leaky_relu(conv(x,graph,self.n_relations),negative_slope=0.10)
                x=F.dropout(x,p=self.dropout,training=self.training)
        return x

    def score(self,triples,z):
        if self.decoder=="complex": return complex_score(z,self.relations.weight,triples,self.dim)
        return distmult_score(z,self.relations.weight,triples)

    def candidate_scores(self,h,r,t,z,direction):
        """All-entity ranking for arbitrary batch of queries, direction=head/tail."""
        if self.decoder=="distmult":
            if direction=="tail": v=z[h]*self.relations(r)
            else: v=z[t]*self.relations(r)
            return v@z.T
        er,ei=z.split(self.dim,dim=-1)
        rr,ri=self.relations(r).split(self.dim,dim=-1)
        if direction=="tail":
            hr,hi=er[h],ei[h]
            v1=hr*rr-hi*ri;v2=hr*ri+hi*rr
        else:
            tr,ti=er[t],ei[t]
            v1=rr*tr+ri*ti;v2=rr*ti-ri*tr
        return v1@er.T+v2@ei.T

    def score_with_context(self,triples,graph,graph_off=False):
        return self.score(triples,self.encode(graph,graph_off=graph_off))
