import numpy as np
import torch
from rimeskg.witnesses import TwoHopIndex,TemplateBank
from rimeskg.training import consistency_weights,normalized_consistency


def test_two_hop_count_damage_no_support():
    graph=np.array([[0,0,1],[1,1,2],[0,0,3],[3,1,2],[0,2,2],[5,0,6]],dtype=np.int64)
    idx=TwoHopIndex(graph)
    assert idx.count(0,2,[(0,1)])==2
    assert idx.count(5,2,[(0,1)])==0
    bank=TemplateBank({2:[(0,1)]})
    q=np.array([[0,2,2],[5,2,2]])
    ref=bank.counts(q,graph)
    assert list(ref)==[2,0]
    damaged=graph[[0,2,3,4,5]]
    d,support,cur=bank.loss(q,damaged,ref)
    assert support.tolist()==[True,False]
    assert np.allclose(d,[.5,0])
    assert not np.isnan(d).any()


def test_template_fit_uses_only_C_L(toy):
    train,valid,test,n,r=toy
    C=train[:16];L=train[16:]
    bank=TemplateBank.fit(C,L,r,top_k=4)
    assert set(bank.templates.keys())==set(range(r))
    assert all(len(v)<=4 for v in bank.templates.values())


def test_T20_kl_mass_normalization_invariant_under_rescaling():
    t=torch.tensor([[4.,2.,1.],[2.,1.,0.]])
    s=torch.tensor([[2.,1.,3.],[1.,2.,3.]],requires_grad=True)
    a=normalized_consistency(t,s,np.array([.3,.6]),2.0)
    b=normalized_consistency(t,s,np.array([3.,6.]),2.0)
    assert torch.allclose(a,b,atol=1e-6)
    a.backward()
    assert s.grad.abs().sum()>0
    z=normalized_consistency(t,s,np.array([0.,0.]),2.0)
    assert z.item()==0.


def test_matched_weight_eligibility(toy):
    train,_,_,n,_=toy
    q=train[:5]
    d=np.array([.1,0,.4,1.,.3]);a=np.array([1,1,0,1,1],dtype=bool)
    gate=torch.tensor([.5,.3,.8,.9,.2])
    for method in ("rimes","uniform","gate_only","degree","relation","shuffled_d"):
        w=consistency_weights(method,d,a,gate,q,train,n,np.random.default_rng(12))
        assert np.array_equal(np.nonzero(w)[0],np.array([0,3,4]))


def test_global_damage_shuffle_preserves_matched_strata(toy):
    from rimeskg.training import shuffle_damage_global
    train,_,_,n,r=toy
    # Several triples share degree/relation strata, even if some strata singleton.
    damage=np.linspace(.1,.9,len(train));support=np.ones(len(train),dtype=bool)
    res,fraction=shuffle_damage_global(train,train,n,damage,support,seed=19)
    assert sorted(res)==sorted(damage)
    assert 0<=fraction<=1
    res2,_=shuffle_damage_global(train,train,n,damage,support,seed=19)
    assert np.allclose(res,res2)
