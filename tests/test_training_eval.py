import numpy as np
import torch
from rimeskg.training import TrainSettings,train_p1,train_p2
from rimeskg.evaluation import FilteredEvaluator,evaluate_validation,evaluate_final
from rimeskg.models import KGCModel


def test_filtered_full_ranking_target_not_filtered(toy):
    train,val,test,n,r=toy
    model=KGCModel(n,r,dim=4,encoder="static",decoder="distmult")
    pool=FilteredEvaluator(np.concatenate([train,val,test]),n)
    metrics,rows=pool.evaluate(model,train,test,with_per_query=True)
    assert metrics["n_ranks"]==2*len(test)
    assert len(rows)==2*len(test)
    assert all(1<=x["rank"]<=n for x in rows)


def test_p1_retrain_and_p2_fixed_labels(toy):
    train,val,test,n,r=toy
    s=TrainSettings(dim=6,layers=1,bases=2,steps=2,batch_size=4,negatives=2,
                    rho=.2,device="cpu",warmup_steps=1)
    m,log=train_p1(train,n,r,s,seed=3)
    assert log["steps"]==2 and log["training_triples"]==len(train)
    v,_=evaluate_validation(m,train,train,val,n)
    assert v["n_ranks"]==2*len(val)
    C,L=train[:14],train[14:]
    for method in ("rimes","uniform","group_dro","graph_off"):
        p2,info,bank=train_p2(C,L,n,r,s,seed=13,method=method)
        assert info["steps"]==2 and info["train_labels"]==len(L)
        final,_=evaluate_final(p2,C,train,val,test,n)
        assert final["n_ranks"]==2*len(test)


def test_test_truth_is_not_input_to_training(toy):
    train,val,test,n,r=toy
    s=TrainSettings(steps=1,batch_size=2,negatives=1)
    # Function signatures deliberately have no `test` or `filtered_truth` parameter.
    import inspect
    for fn in (train_p1,train_p2):
        params=inspect.signature(fn).parameters
        assert "test" not in params and "filtered_truth" not in params


def test_full_official_counts_mandatory_for_benchmark(tmp_path,toy):
    from rimeskg.data import load_codex
    from pathlib import Path
    tr,va,te,n,r=toy
    folder=tmp_path/"codex-s";folder.mkdir()
    for arr,name in ((tr,"train"),(va,"valid"),(te,"test")):
        (folder/(name+".txt")).write_text("\n".join(f"Q{h}\tP{rel}\tQ{t}" for h,rel,t in arr)+"\n")
    import pytest
    with pytest.raises(ValueError,match="official full splits"):
        load_codex(tmp_path,"s",verify=True)
