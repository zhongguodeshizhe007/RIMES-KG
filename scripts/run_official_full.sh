#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e .
python -m pytest -q
rimes download --root data --dataset both
rimes verify --root data --dataset s
rimes verify --root data --dataset m
rimes run --config configs/full_s.yaml --data-root data --output results/official_s --mode all
rimes run --config configs/full_m.yaml --data-root data --output results/official_m --mode all
python scripts/merge_runs.py --inputs results/official_s results/official_m --output results/official_combined
rimes render --input results/official_combined
printf '\nDONE: official outputs in results/official_combined\n'
