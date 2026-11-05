#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python -m pytest -q
python -m rimeskg.cli run --config configs/pilot_s.yaml --data-root examples --output results/offline_smoke --allow-unverified --max-queries 30
python -m rimeskg.cli render --input results/offline_smoke
printf '\nDebug smoke complete; results are NOT official benchmark results.\n'
