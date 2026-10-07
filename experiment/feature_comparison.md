# Feature comparison (E-1 hyperparameter study)

Same vertices (3,137 cells with ≥ 60 snow days per year, duplicates contracted), same
graph pipeline (kNN with k = 21, self-tuning Gaussian weights, normalized Laplacian,
K = 3 spectral clustering); only the feature changes.

Reproduce: `python -m experiment.compare_features` (about 15 min), then
`python -m experiment.make_compare_figures`. Raw numbers: `feature_comparison.json`.
Figures: `figures/compare_maps.png`, `figures/compare_lags.png`.

## Scores

No ground truth exists, so each feature is scored on independent checks.

- **Glacier purity**: share of the 74 Yao et al. (2012) glaciers (regions V+IV / II+VI+VII / I) in their group's majority region.
- **Contiguity**: share of map-neighbour pairs in the same region (chance ≈ 0.36). The features contain no location.
- **Climate**: |r(q₂, longitude)| among cells at 4,400–5,000 m.
- **Elev η²**: share of elevation variance explained by the regions (lower means less elevation-driven).
- **Split ARI**: agreement of partitions built from two disjoint halves of the years (mean ± sd over 5 random 7/7 splits).
- **Split kNN**: share of each cell's 21 nearest neighbours that the two half-year graphs have in common.
- **Era kNN**: the same overlap between 2000/01–2007/08 and 2008/09–2013/14, which straddles the SSM/I → SSMIS sensor change.
- **vs main**: ARI with the partition of the earlier draft's feature (smoothed HOG).

| Feature | d | Glacier purity | Contiguity | Climate | Elev η² | Split ARI | Split kNN | Era kNN | vs main |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw depth (no gradient) | 24 | 0.99 | 0.85 | 0.78 | 0.23 | 0.72 ± 0.07 | 0.37 | 0.30 | 0.47 |
| snow occurrence only | 24 | 0.93 | 0.76 | 0.55 | 0.05 | 0.74 ± 0.05 | 0.28 | 0.17 | 0.23 |
| teammate hf102 (default) | 240 | 0.91 | 0.87 | 0.67 | 0.15 | 0.73 ± 0.02 | 0.22 | 0.16 | 0.48 |
| teammate hf102 (gradient only) | 240 | 0.91 | 0.80 | 0.58 | 0.10 | 0.24 ± 0.06 | 0.15 | 0.11 | 0.26 |
| HOG, lag 1 | 288 | 0.95 | 0.84 | 0.75 | 0.13 | 0.73 ± 0.05 | 0.28 | 0.20 | 0.53 |
| HOG, smoothed (earlier draft) | 288 | 0.99 | 0.87 | 0.74 | 0.13 | 0.42 ± 0.13 | 0.39 | 0.31 | 1.00 |
| HOG, lag 3 | 288 | 0.99 | 0.85 | 0.75 | 0.14 | 0.77 ± 0.04 | 0.35 | 0.26 | 0.72 |
| HOG, lag 7 | 288 | 0.97 | 0.87 | 0.77 | 0.12 | 0.55 ± 0.16 | 0.42 | 0.32 | 0.90 |
| HOG, lag 15 | 288 | 0.99 | 0.88 | 0.80 | 0.11 | 0.77 ± 0.06 | 0.47 | 0.38 | 0.87 |
| HOG, lag 31 | 288 | 0.99 | 0.88 | 0.82 | 0.10 | 0.72 ± 0.07 | 0.48 | 0.39 | 0.78 |
| HOG, lag 61 | 288 | 0.99 | 0.88 | 0.77 | 0.10 | 0.57 ± 0.10 | 0.44 | 0.36 | 0.66 |
| **multi-scale HOG, lags 1–31** | 1440 | 0.99 | 0.88 | 0.80 | 0.11 | **0.80 ± 0.04** | 0.51 | 0.40 | 0.88 |
| **multi-scale HOG, lags 7–61** | 1152 | 0.99 | **0.88** | **0.83** | 0.10 | 0.76 ± 0.06 | **0.53** | **0.43** | 0.82 |
| **multi-scale HOG, lags 1, 7–61 (chosen for the report)** | 1440 | 0.99 | **0.88** | 0.82 | **0.09** | **0.78 ± 0.04** | **0.53** | 0.42 | 0.83 |
| Haar wavelet energy | 144 | 0.72 | 0.87 | 0.77 | 0.04 | 0.42 ± 0.08 | 0.46 | 0.39 | 0.33 |
| Fourier harmonics | 12 | 0.92 | 0.88 | 0.72 | 0.26 | 0.56 ± 0.03 | 0.31 | 0.26 | 0.33 |
| phenology statistics | 8 | 0.99 | 0.85 | 0.65 | 0.16 | 0.39 ± 0.17 | 0.20 | 0.14 | 0.65 |

## Principles

- **raw depth**: the share of the year's snow depth in each half-month. It keeps the
  amount of snow, so it follows elevation (η² = 0.23) and puts SE Tibet (region I) with the
  west.
- **snow occurrence**: the share of days with any snow, per half-month. It records only
  when snow is present, not how it changes.
- **teammate hf102**: each day with snow casts one vote for its daily-change bin
  (9 bins of atan(change / 1 cm/day)). Day counts per half-month are turned into
  distributions, square-rooted (Hellinger), and joined 50/50 with the snow-day share.
  Most days with snow have zero or tiny, noisy changes, so the gradient part alone is
  unstable (split ARI 0.24). The occurrence part supplies most of the stable structure.
  Re-implemented from `snow_divides/e1_features/histogram.py`, with windows = 24 equal
  parts of the snow year; on this repository's vertex set, not hf102's 3,941 cells.
- **HOG, lag L**: the rate (s(t + L/2) − s(t − L/2)) / L. Each day votes |rate| into
  one of 6 slope bins (fast melt … fast accumulation), with half-month cells and
  2-cell L2-Hys blocks. A difference over L days averages out changes shorter than L days.
  Daily passive-microwave depth flickers by several cm (`compare_lags.png` b), so short
  lags mostly see noise. Lags of 2–4 weeks see the seasonal accumulation and melt that
  separate the climate regimes.
- **multi-scale HOG**: HOG at several lags, concatenated, which is a scale-space
  description. Lags 7–61 give the most reliable graph, the strongest climate signal and
  the smallest elevation dependence.
- **Haar wavelet energy**: energy of box-smoothed gradients at 2–64-day scales, per
  month and sign. It discards the slope angle and is dominated by a few large events. It
  separates the Pamir (V) from the West Himalaya (IV), so the glacier purity is 0.72.
- **Fourier harmonics**: annual harmonics 1–6, relative to the mean depth. Amplitude
  measures seasonality, phase the timing. It follows elevation (η² = 0.26).
- **phenology**: onset, melt-out, number of snow days, peak day, number of episodes,
  accumulation and melt durations, and the share of accumulation on days gaining more than
  1 cm. These 8 numbers are interpretable but too few and too noisy (split ARI 0.39).

## Conclusions

1. **Longer and multiple lags are better than daily changes.** Lags of 15–31 days,
   and multi-scale lags 7–61, are best or near-best on every check.
2. **The earlier draft's feature (smoothed, about 2-day lag) is valid but unstable.** Its
   K = 3 partition flips between two solutions depending on the years used
   (split ARI 0.42 ± 0.13).
3. **Counting days instead of weighting by the size of the change** loses most of the
   gradient information. Without the occurrence term, the teammate feature is the least
   stable.
4. **The era overlap is below the split overlap for every feature.** The 2008/09 sensor
   change (or real decadal change) affects all features alike.
5. **Caveats.** K = 3 is imposed. Only 74 glaciers are available for the external check.
   The scores are proxies, not accuracy.
6. **Chosen feature.** The report uses the multi-scale HOG with lags 1, 7, 15, 31 and 61 days
   (`experiment/feature_comparison_lag1.json`). Adding the 1-day block to lags 7–61 does not
   hurt and slightly stabilises the partition.
