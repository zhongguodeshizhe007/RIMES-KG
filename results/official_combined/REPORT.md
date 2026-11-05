# RIMES-KG automated figure/table report

Scope: **OFFICIAL_FULL**

All named outputs correspond to the V2/V3 proposal specifications.

## Output inventory (Figures)
- **Fig1**: Equal-K masks, unequal two-hop evidence (schematic) (`figures/Fig1.png`)
- **Fig2**: P1/P2 information boundaries (protocol diagram) (`figures/Fig2.png`)
- **Fig3**: RIMES teacher-student and normalized KL (method diagram) (`figures/Fig3.png`)
- **Fig4**: P1 fixed-test MRR under missingness (RQ1 empirical) (`figures/Fig4.png`)
- **Fig5**: Evidence survival vs ranking degradation (RQ2 empirical) (`figures/Fig5.png`)
- **Fig6**: P2 held-out family robustness (RQ3 empirical) (`figures/Fig6.png`)
- **Fig7**: Ablation and computational cost tradeoffs (efficiency frontier) (`figures/Fig7.png`)

## Output inventory (Tables)
- **Table0**: Prior-work novelty boundary (`tables/Table0.csv` & `.tex`)
- **Table1**: Datasets and protocol metadata (`tables/Table1.csv` & `.tex`)
- **Table2**: Mask profiles and matched null residuals (`tables/Table2.csv` & `.tex`)
- **Table3**: P1 training-observation effects and paired comparisons (`tables/Table3.csv` & `.tex`)
- **Table4**: RQ2 matched evidence and association analysis (`tables/Table4.csv` & `.tex`)
- **Table5**: P2 robust results, clean/held-out tradeoffs and GPU cost (`tables/Table5.csv` & `.tex`)
- **Table6**: RIMES ablation and evidence allocation controls (`tables/Table6.csv` & `.tex`)

## Interpretation boundary
Official CoDEx-S/M counts and full ranking are required for formal claims.
P1 and P2 use different supervision protocols; their scores must not be compared as a single leaderboard.
No p-values or narrow confidence intervals are inferred from dependent triples.
