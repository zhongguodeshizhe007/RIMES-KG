"""CLI: python -m rimeskg.cli COMMAND ... or installed `rimes`."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .data import download_codex,load_codex
from .runner import run_experiment
from .reporting import render_outputs
from .utils import save_json


def main(argv=None):
    p=argparse.ArgumentParser(prog="rimes",description="RIMES-KG V3 auditable reproduction")
    sub=p.add_subparsers(dest="cmd",required=True)
    dl=sub.add_parser("download",help="Download CoDEx official splits from authors' GitHub")
    dl.add_argument("--root",default="data");dl.add_argument("--dataset",choices=["s","m","both"],default="both")
    check=sub.add_parser("verify",help="Verify official full CoDEx counts and SHA256")
    check.add_argument("--root",default="data");check.add_argument("--dataset",choices=["s","m"],default="s")
    exp=sub.add_parser("run",help="Run full P1/P2 experiment matrix")
    exp.add_argument("--config",default="configs/full_s.yaml")
    exp.add_argument("--data-root",default="data")
    exp.add_argument("--output",required=True)
    exp.add_argument("--mode",choices=["all","p1","p2"],default="all")
    exp.add_argument("--max-queries",type=int,help="PILOT ONLY: truncates evaluation triples")
    exp.add_argument("--allow-unverified",action="store_true",help="EXCERPTS ONLY: bypass official split counts")
    render=sub.add_parser("render",help="Generate Fig.1-4, Table0-5, report, and CSVs")
    render.add_argument("--input",required=True);render.add_argument("--output",help="defaults to input folder")
    args=p.parse_args(argv)
    if args.cmd=="download":
        for ds in (["s","m"] if args.dataset=="both" else [args.dataset]):
            print("Downloading",download_codex(args.root,ds),flush=True)
    elif args.cmd=="verify":
        s=load_codex(args.root,args.dataset,verify=True)
        print(json.dumps(s.metadata(),indent=2))
    elif args.cmd=="run":
        cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
        root=run_experiment(args.data_root,args.output,cfg,mode=args.mode,
                            allow_unverified=args.allow_unverified,max_queries=args.max_queries)
        print("Completed",root)
    elif args.cmd=="render":
        output=render_outputs(args.input,args.output or args.input)
        print("Rendered figures/tables",output)


if __name__=="__main__":main()
