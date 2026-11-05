import sys
import os
import time
from pathlib import Path
import yaml
import subprocess

# Ensure rimes_kg_reproduction is in sys.path
repo_root = Path("rimes_kg_reproduction").resolve()
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from rimeskg.runner import run_experiment
from rimeskg.reporting import render_outputs

def run_all():
    data_dir = str((Path(__file__).parent / "data").resolve())
    # Change working directory to rimes_kg_reproduction so relative paths in configs match
    os.chdir(str(repo_root))
    
    # 1. CoDEx-S
    print("=" * 60, flush=True)
    print(">>> [STAGE 1/4] EXECUTING OFFICIAL CODEX-S MATRIX", flush=True)
    print("=" * 60, flush=True)
    t0 = time.perf_counter()
    cfg_s = yaml.safe_load(Path("configs/full_s.yaml").read_text(encoding="utf8"))
    out_s = run_experiment(data_dir, "results/official_s", cfg_s, mode="all")
    t_s = time.perf_counter() - t0
    print(f"\n>>> CoDEx-S Full Matrix finished in {t_s:.1f}s ({t_s/60:.1f} min)", flush=True)
    render_outputs("results/official_s", "results/official_s")
    
    # 2. CoDEx-M
    print("\n" + "=" * 60, flush=True)
    print(">>> [STAGE 2/4] EXECUTING OFFICIAL CODEX-M REPLICATION MATRIX", flush=True)
    print("=" * 60, flush=True)
    t0 = time.perf_counter()
    cfg_m = yaml.safe_load(Path("configs/full_m.yaml").read_text(encoding="utf8"))
    out_m = run_experiment(data_dir, "results/official_m", cfg_m, mode="all")
    t_m = time.perf_counter() - t0
    print(f"\n>>> CoDEx-M Full Matrix finished in {t_m:.1f}s ({t_m/60:.1f} min)", flush=True)
    render_outputs("results/official_m", "results/official_m")
    
    # 3. Merge
    print("\n" + "=" * 60, flush=True)
    print(">>> [STAGE 3/4] MERGING EXPERIMENTAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    from scripts.merge_runs import main as merge_main
    old_argv = sys.argv
    sys.argv = ["merge_runs.py", "--inputs", "results/official_s", "results/official_m", "--output", "results/official_combined"]
    merge_main()
    sys.argv = old_argv
    
    # 4. Render All Figures & Tables
    print("\n" + "=" * 60, flush=True)
    print(">>> [STAGE 4/4] RENDERING FIG 1-7 AND TABLE 0-6 ARTIFACTS", flush=True)
    print("=" * 60, flush=True)
    render_outputs("results/official_combined", "results/official_combined")
    
    print("\n" + "=" * 60, flush=True)
    print(">>> ALL OFFICIAL EXPERIMENTS AND ARTIFACTS SUCCESSFULLY PRODUCED!", flush=True)
    print(f">>> Total pipeline runtime: {(t_s + t_m)/60:.1f} minutes", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_all()
