import numpy as np
from rimeskg.masks import generate_mask,mask_profile,strata
from rimeskg.data import split_context_labels
from collections import Counter


def test_exact_budget_reproducible_m1_m4_and_null(toy):
    train,_,_,n,_=toy
    for fam in ("M1","M2","M3","M4"):
        x=generate_mask(train,fam,.25,11,n)
        y=generate_mask(train,fam,.25,11,n)
        assert x.k==6 and len(x.apply(train))==len(train)-6
        assert np.array_equal(x.indices,y.indices)
    x=generate_mask(train,"M4",.5,15,n)
    y=generate_mask(train,"Null-M4",.5,101,n,null_source=x)
    keys=strata(train,n)
    assert Counter(keys[i] for i in x.indices)==Counter(keys[i] for i in y.indices)
    z=mask_profile(train,y,n,reference_mask=x)
    assert z["relation_js_vs_m4"]==0
    assert z["head_degree_js_vs_m4"]==0
    assert z["tail_degree_js_vs_m4"]==0


def test_C_L_split_no_explicit_reciprocal_leakage(toy):
    train,_,_,_,_=toy
    C,L=split_context_labels(train,.35,seed=13)
    assert len(C)+len(L)==len(train)
    pairs=lambda arr:{(min(h,t),max(h,t)) for h,r,t in arr}
    assert not pairs(C)&pairs(L)


def test_deletion_scope_no_val_test_in_generator(toy):
    train,val,test,n,_=toy
    for fam in ("M1","M2","M3","M4"):
        masked=generate_mask(train,fam,.25,13,n).apply(train)
        assert set(map(tuple,masked)).issubset(set(map(tuple,train)))
        assert not set(map(tuple,masked))&set(map(tuple,test))
