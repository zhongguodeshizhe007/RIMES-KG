import numpy as np
import pytest
from pathlib import Path
from rimeskg.data import load_codex
from rimeskg.runner import run_experiment
from rimeskg.reporting import render_outputs


def test_T24_T25_pretest_config_freeze_and_all_outputs(tmp_path,toy):
    tr,va,te,n,r=toy
    root=tmp_path/"data"/"codex-s"
    root.mkdir(parents=True)
    for x,p in ((tr,"train"),(va,"valid"),(te,"test")):
        (root/f"{p}.txt").write_text("\n".join(f"Q{h}\tP{rel}\tQ{t}" for h,rel,t in x)+"\n")
    cfg={"dataset":"s","seeds":[7],"split_seed":4,"label_fraction":.3,"p1_rhos":[.2],
         "p1_decoders":["distmult"],"p2_methods":["erm","rimes"],"validation_queries":2,
         "train":{"device":"cpu","steps":1,"batch_size":3,"negatives":2,"dim":4,"bases":2,
                  "rho":.2,"warmup_steps":1}}
    out=tmp_path/"run"
    run_experiment(tmp_path/"data",out,cfg,allow_unverified=True,max_queries=2)
    import json,pandas as pd
    assert (out/"pretest_freeze.json").exists()
    assert len(pd.read_csv(out/"results.csv"))>0
    assert json.loads((out/"summary.json").read_text())["scope"]=="UNVERIFIED_EXCERPT"
    render_outputs(out,out)
    for k in range(1,5):assert (out/"figures"/f"Fig{k}.png").exists()
    for k in range(6):
        name="Table"+str(k)
        assert (out/"tables"/f"{name}.csv").exists()
    changed={**cfg,"seeds":[13]}
    with pytest.raises(RuntimeError,match="different frozen config"):
        run_experiment(tmp_path/"data",out,changed,allow_unverified=True,max_queries=2)
