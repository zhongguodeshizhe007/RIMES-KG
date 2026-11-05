"""Reproducible RQ1-RQ3 experiment runner. No test access during training."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import os, sys, time, platform, hashlib, json
import pandas as pd
import numpy as np
import torch
from .data import load_codex,split_context_labels,sample_queries,train_positive_set
from .training import TrainSettings,train_p1,train_p2
from .masks import generate_mask,mask_profile,degrees
from .witnesses import TemplateBank
from .evaluation import FilteredEvaluator,evaluate_final,evaluate_validation,relation_macro
from .utils import save_json,append_jsonl,digest_array,sha256


def settings_from_config(cfg):
    return TrainSettings(**cfg.get("train",{}))


def environment_report():
    return {"python":sys.version.split()[0],"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
            "gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy":np.__version__,"platform":platform.platform(),"torch_threads":torch.get_num_threads()}


def run_experiment(data_root,output,cfg,*,mode="all",allow_unverified=False,max_queries=None):
    """Test data enters only via evaluate_final after pre-test config freeze.

    allow_unverified is ONLY for locally sourced official excerpts or testing;
    all such outputs MUST be labeled EXCERPT, never published as CoDEx benchmark.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dataset=cfg.get("dataset","s")
    splits=load_codex(data_root,dataset,verify=not allow_unverified)
    if mode not in ("all","p1","p2"):raise ValueError(mode)
    st=settings_from_config(cfg)
    if st.device=="auto":st=replace(st,device="cuda" if torch.cuda.is_available() else "cpu")
    seeds=[int(v) for v in cfg.get("seeds",[11,17,23])]
    rhos=[float(v) for v in cfg.get("p1_rhos",[0.20])]
    methods=cfg.get("p2_methods",["erm","dropedge","uniform","gate_only","degree","relation","shuffled_d","group_dro","rimes","graph_off"])
    flag="OFFICIAL_FULL" if splits.official and max_queries is None else ("OFFICIAL_PARTIAL_EVAL" if splits.official else "UNVERIFIED_EXCERPT")
    config_payload={"config":cfg,"mode":mode,"scope":flag,"max_queries":max_queries,
                    "data_manifest":splits.metadata(),"environment":environment_report()}
    # Immutable run contract before ANY test results. A changed config requires a new out directory.
    frozen=out/"pretest_freeze.json"
    if frozen.exists():
        old=json.loads(frozen.read_text(encoding="utf8"))
        if old!=config_payload: raise RuntimeError("This output directory has a different frozen config: use a NEW output directory")
    else:save_json(frozen,config_payload)
    C,L=split_context_labels(splits.train, cfg.get("label_fraction",0.25),cfg.get("split_seed",123))
    c_rels=set(C[:,1].tolist()); l_rels=set(L[:,1].tolist())
    protocol={"scope":flag,"C_hash":digest_array(C),"L_hash":digest_array(L),"C_size":len(C),"L_size":len(L),
              "missing_relation_ids_in_C":sorted(set(range(splits.n_relations))-c_rels),
              "missing_relation_ids_in_L":sorted(set(range(splits.n_relations))-l_rels),
              "C_fraction":len(C)/len(splits.train),"L_fraction":len(L)/len(splits.train),
              "grouped_inverse_pairs":True,"official_counts":splits.metadata(),
              "p1":"retrain on independent Tobs","p2":"fixed context C, supervised labels L",
              "fit_templates_from":"C+L training facts only", "mask_train_families":["M1","M2"],
              "mask_heldout_families":["M3","M4"], "filtering":"evaluation-only all truths",
              "full_entity_ranking":True,"tie_policy":"average","max_queries":max_queries}
    save_json(out/"protocol.json",protocol)
    cfg_rho=float(st.rho)
    results=[];mask_rows=[];rq2_rows=[]
    if mode in ("all","p1"):
        for rho in rhos:
            for seed in seeds:
                mask_seed=seed+1001
                m4=generate_mask(splits.train,"M4",rho,mask_seed,splits.n_entities)
                mask_dict={"clean":None,
                           "M1":generate_mask(splits.train,"M1",rho,mask_seed,splits.n_entities),
                           "M2":generate_mask(splits.train,"M2",rho,mask_seed,splits.n_entities),
                           "M4":m4,
                           "Null-M4":generate_mask(splits.train,"Null-M4",rho,mask_seed+91,splits.n_entities,null_source=m4)}
                for name,mask in mask_dict.items():
                    if mask is not None:
                        row={"protocol":"P1","dataset":dataset,"rho":rho,"run_seed":seed,**mask_profile(splits.train,mask,splits.n_entities,reference_mask=m4 if name=="Null-M4" else None)}
                        mask_rows.append(row)
                    observed=splits.train if mask is None else mask.apply(splits.train)
                    for decoder in cfg.get("p1_decoders",["distmult","complex"]):
                        model,log=train_p1(observed,splits.n_entities,splits.n_relations,replace(st,rho=rho),seed=seed,decoder=decoder)
                        t0=time.perf_counter()
                        metric,rows=evaluate_final(model,observed,splits.train,splits.valid,splits.test,splits.n_entities,
                                                   max_queries=max_queries,with_per_query=True)
                        eval_time=time.perf_counter()-t0
                        print(f"[P1] dataset={dataset.upper()} seed={seed} rho={rho:.2f} family={name:<7} decoder={decoder:<8} train={log['train_time_s']:.1f}s MRR={metric['MRR']:.4f}", flush=True)
                        results.append({"protocol":"P1","dataset":dataset,"dataset_scope":flag,"seed":seed,
                          "rho":rho,"family":name,"method":decoder,"train_time_s":log["train_time_s"],
                          "eval_time_s":eval_time,"steps":st.steps,"labels":len(observed),
                          "relation_macro_mrr":relation_macro(rows,None),
                          "fullrank_latency_ms_amortized":eval_time*1000/max(1,metric["n_ranks"]),
                          "train_gpu_hours_estimate":log["train_time_s"]/3600 if st.device.startswith("cuda") else None,
                          "device":st.device,**metric})
                        append_jsonl(out/"train_audit.jsonl",{"protocol":"P1","family":name,"decoder":decoder,
                                                           "seed":seed,"rho":rho,"loss_trace":log["trace"],
                                                           "obs_sha256":digest_array(observed)})
                        del model
    if mode in ("all","p2"):
        templates=TemplateBank.fit(C,L,splits.n_relations,st.top_templates)
        save_json(out/"train_only_templates.json",templates.describe())
        W0_test=templates.counts(splits.test,C)  # offline diagnostics ONLY; never used by trainer.
        hc,tc=degrees(C,splits.n_entities)
        for seed in seeds:
            mask_seed=seed+70001
            m4=generate_mask(C,"M4",cfg_rho,mask_seed,splits.n_entities)
            masks={"clean":None,
                   "M3":generate_mask(C,"M3",cfg_rho,mask_seed,splits.n_entities),
                   "M4":m4,
                   "Null-M4":generate_mask(C,"Null-M4",cfg_rho,mask_seed+1,splits.n_entities,null_source=m4)}
            for name,mask in masks.items():
                if mask is None:continue
                c_m=mask.apply(C)
                wsurv=templates.counts(L,c_m)
                wbase=templates.counts(L,C)
                fraction=float(wsurv[wbase>0].sum()/max(1,wbase[wbase>0].sum()))
                mask_rows.append({"protocol":"P2","dataset":dataset,"rho":cfg_rho,"run_seed":seed,
                                  **mask_profile(C,mask,splits.n_entities,reference_mask=m4 if name=="Null-M4" else None,witness_retained=fraction)})
            for method in methods:
                model,log,_=train_p2(C,L,splits.n_entities,splits.n_relations,st,seed,method=method,templates=templates)
                # Train-label-only model sensitivity and teacher reliability audit.
                # Neither operation has access to Tte or uses held-out results for tuning.
                with torch.no_grad():
                    model.eval()
                    cv=torch.as_tensor(C,device=st.device,dtype=torch.long)
                    changed=torch.as_tensor(generate_mask(C,"M1",cfg_rho,seed+77711,splits.n_entities).apply(C),
                                            device=st.device,dtype=torch.long)
                    off=(method=="graph_off")
                    zclean=model.encode(cv,graph_off=off)
                    zshift=model.encode(changed,graph_off=off)
                    probe=torch.as_tensor(L[:min(64,len(L))],device=st.device,dtype=torch.long)
                    sensitivity={"mean_abs_z_change":float((zclean-zshift).abs().mean()),
                                 "mean_abs_score_change":float((model.score(probe,zclean)-model.score(probe,zshift)).abs().mean())}
                    # Sampled-candidate margin does not certify full-entity ranking.
                    rng_audit=np.random.default_rng(seed+909011)
                    _,kc=sample_queries(L,splits.n_entities,train_positive_set(C,L),min(48,len(L)),16,rng_audit)
                    kt=torch.as_tensor(kc,device=st.device,dtype=torch.long)
                    sc=model.score(kt,zclean)
                    margins=(sc[:,0]-sc[:,1:].max(dim=1).values).cpu().numpy()
                    sampled_good=(margins>0)
                # Training-side filter pool only; tail predictions on *training* L labels.
                training_evaluator=FilteredEvaluator(splits.train,splits.n_entities)
                _,lq=training_evaluator.evaluate(model,C,kc[:,0,:],max_queries=None,
                                 with_per_query=True,graph_off=(method=="graph_off"))
                tail_bad=np.array([entry["rank"]>1 for entry in lq if entry["direction"]=="tail"])
                reliability={"audited_L_queries":len(kc),"sampled_margin_positive_fraction":float(sampled_good.mean()),
                             "full_rank_error_fraction":float(tail_bad.mean()),
                             "high_margin_full_rank_error_fraction":float(np.mean(sampled_good & tail_bad))}
                append_jsonl(out/"train_audit.jsonl",{"protocol":"P2","method":method,"seed":seed,"log":log,
                                                      "sensitivity":sensitivity,"teacher_reliability":reliability})
                # Validation only M1/M2; no test-related masks or model tuning.
                for family in ("clean","M1","M2"):
                    vgraph=C if family=="clean" else generate_mask(C,family,cfg_rho,seed+83001,splits.n_entities).apply(C)
                    vm,_=evaluate_validation(model,vgraph,splits.train,splits.valid,splits.n_entities,
                                              max_queries=min(len(splits.valid),cfg.get("validation_queries",128)))
                    append_jsonl(out/"validation_audit.jsonl",{"method":method,"seed":seed,"family":family,"metrics":vm})
                clean_per=None
                for family,mask in masks.items():
                    graph=C if mask is None else mask.apply(C)
                    t0=time.perf_counter()
                    metrics,rows=evaluate_final(model,graph,splits.train,splits.valid,splits.test,splits.n_entities,
                                                max_queries=max_queries,graph_off=(method=="graph_off"),with_per_query=True)
                    eval_time=time.perf_counter()-t0
                    results.append({"protocol":"P2","dataset":dataset,"dataset_scope":flag,"seed":seed,
                      "rho":cfg_rho,"family":family,"method":method,"train_time_s":log["train_time_s"],
                      "eval_time_s":eval_time,"steps":st.steps,"labels":len(L),
                      "graph_edges":len(graph),"reference_support_fraction":log["support_fraction"],
                      "relation_macro_mrr":relation_macro(rows,None),
                      "fullrank_latency_ms_amortized":eval_time*1000/max(1,metrics["n_ranks"]),
                      "train_gpu_hours_estimate":log["train_time_s"]/3600 if st.device.startswith("cuda") else None,
                      "device":st.device,**metrics})
                    if family=="clean": clean_per={(x["q_idx"],x["direction"]):x for x in rows}
                    elif family in ("M3","M4","Null-M4"):
                        d,valid,_=templates.loss(splits.test,graph,W0_test)
                        for row in rows:
                            idx=row["q_idx"];base=clean_per[(idx,row["direction"])]
                            h,r,t=map(int,(row["h"],row["r"],row["t"]))
                            rq2_rows.append({"dataset":dataset,"method":method,"seed":seed,"family":family,
                                            "q_idx":idx,"direction":row["direction"],"r":r,
                                            "head_degree":int(hc[h]),"tail_degree":int(tc[t]),
                                            "W0":int(W0_test[idx]),"no_support":not bool(valid[idx]),
                                            "damage":float(d[idx]) if valid[idx] else None,
                                            "rank_clean":base["rank"],"rank_shift":row["rank"],
                                            "rr_change":row["rr"]-base["rr"]})
                del model
                print(f"[P2] dataset={dataset.upper()} seed={seed} method={method:<15} train={log['train_time_s']:.1f}s done", flush=True)
    pd.DataFrame(results).to_csv(out/"results.csv",index=False)
    pd.DataFrame(mask_rows).to_csv(out/"mask_profiles.csv",index=False)
    pd.DataFrame(rq2_rows).to_csv(out/"rq2_queries.csv",index=False)
    save_json(out/"summary.json",{"dataset":dataset,"scope":flag,"rows":len(results),"masks":len(mask_rows),
                                 "paired_queries":len(rq2_rows),"completed":True,
                                 "caveat":"Results with EXCERPT or PARTIAL scope are NEVER official benchmark claims."})
    return out
