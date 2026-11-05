"""Combine independent official CoDEx-S/M run directories without mixing partial or excerpts."""
from pathlib import Path
import argparse,json
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True,help="two or more completed run dirs")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    summaries=[];protos=[]
    for ip in args.inputs:
        inp=Path(ip)
        summary=json.loads((inp/"summary.json").read_text())
        if summary["scope"]!="OFFICIAL_FULL":
            raise ValueError(f"Refusing non-official/partial run: {ip}: {summary['scope']}")
        summaries.append(summary)
        protos.append(json.loads((inp/"protocol.json").read_text()))
    datasets={x["dataset"] for x in summaries}
    if not {"s","m"}.issubset(datasets):
        raise ValueError("Both official CoDEx-S and CoDEx-M required")
    for file in ("results.csv","mask_profiles.csv","rq2_queries.csv"):
        df=pd.concat([pd.read_csv(Path(ip)/file) for ip in args.inputs],ignore_index=True)
        df.to_csv(dest/file,index=False)
    (dest/"dataset_protocols.json").write_text(json.dumps(protos,indent=2))
    (dest/"summary.json").write_text(json.dumps({"scope":"OFFICIAL_FULL","datasets":sorted(datasets),
         "result":"merged verified run directories","rule":"no excerpt or partial results allowed"},indent=2))
    print("Created",dest)

if __name__=="__main__":main()
