# HPGR — Localized Hierarchical Probabilistic Granular-Ball Modeling

Reference implementation of the main pipeline of *Localized Hierarchical Probabilistic
Granular-Ball Modeling for One-Class Classification* (Pattern Recognition). Tabular
one-class classification / anomaly detection only.

## Files

```
main.py                  load -> standardize -> hierarchical splitting -> score -> evaluate
configs/default.yaml     all parameters (single source of truth)
requirements.txt
hpgr/
  granular_ball.py       a probabilistic granular-ball B_k = N(mu_k, Sigma_k)
  model.py               representation, local BIC splitting, hierarchy construction, scoring
  scoring.py             log-density, entropy-based level weights
  data.py                .mat loading (last column = label)
  metrics.py             AUC / AP / F1
  config.py              YAML loading
```

## Data

The datasets are the tabular `.mat` files distributed with GBMOD
(`https://github.com/BELLoney/Outlier-detection`). Each file holds one array: the
features followed by a label column (`1` = anomaly, `0` = normal). Place them in
`data/anomaly/`.

The `.mat` files are already min-max normalized upstream by GBMOD. This code applies a
`StandardScaler` fitted on the normal samples only. Training uses the normal samples;
evaluation uses all samples.

## Run

```
pip install -r requirements.txt
python main.py --dataset bands_band_34_variant1.mat
```

## Reproducibility

The results reported in the paper are averages over five runs with different random
seeds. Pass `--seed` to fix a seed.
