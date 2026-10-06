# Snow Divides

*Do snow seasons draw the climate boundaries of the Tibetan Plateau?*

CS-521 Graph-based Data Analysis (Duke, Fall 2026). Team project for HW1 Part II.

## Motivation

The Tibetan Plateau is split between two climates: the **westerlies** in the northwest and
the **Indian monsoon** in the southeast, with a transition zone in between. The divide
matters because glaciers behave differently on each side. Those in the northwest are
stable or advancing, while those in the southeast retreat fastest (Yao et al., 2012).

Snow records when precipitation falls and how cold it is. So the shape of the snow season
(when snow arrives, how fast it builds up, how fast it melts) should carry the same
climate signature. A kNN graph of these shapes can find the regions without supervision.
Its Laplacian spectrum then lets us study the boundaries between them.

## 1. Research question

> From snow alone, does a kNN graph of snow-season shapes recover the plateau's climate
> regions? Are the boundaries between them sharp divides, or broad transition zones that
> shift from year to year?

| | Hypothesis | Test |
|---|---|---|
| **H1** Regions | The graph's partition is spatially coherent and matches the westerly / transition / monsoon pattern, and elevation alone does not explain it. | Map the nodal domains and spectral clusters. Compare them with a DEM and with 82 glacier records (Yao et al., 2012). Repeat within a single elevation band. |
| **H2** Boundaries | Boundaries are broad, shifting belts, not lines. | Three checks: the gradient of the spectral embedding on the map; the clustering coefficient of border pixels; how often border pixels switch regime across per-year graphs. |
| **H3** Confounds | Part of the partition is not climate. | Elevation; glacier ice read as snow (use glacier fraction per pixel as a covariate); satellite sensor changes (look for jumps in graph statistics in the years the sensor changed). |

## 2. Method

- **Vertices:** about 3,800 grid cells (0.25°) on the plateau (26–40°N, 73–105°E), each
  with at least 30 snow days per year.
- **Attributes:** daily snow depth over 14 snow years (1 Aug – 31 Jul), 2000/01–2013/14.
- **Feature, temporal HOG** (HW1 1-b):
  - the orientation of each day is the slope angle of the depth curve;
  - 6 bins, from fast melt to fast accumulation;
  - half-month cells, blocks of 2 cells normalised with L2-Hys;
  - averaged over the 14 years, giving d = 288.

  It captures the timing and speed of snow change, not the amount of snow.
- **Distance:** square root of the Jensen–Shannon divergence; Hellinger distance as a check.
- **Graph:** kNN with k = 21–26, symmetrised by union, self-tuning Gaussian weights,
  normalized Laplacian, spectral clustering.

| Item | Question it answers |
|---|---|
| E-1 | Which temporal resolution reveals the regimes? (Cell length, and so d, is a hyperparameter; the raw depth curve is the baseline.) |
| E-2 | Are the results robust to the choice of metric? |
| E-3 | In what order do regions merge as k grows? This ranks how distinct each region is. |
| E-4 | Do border pixels have lower clustering coefficients? If so, boundaries can be found from graph structure alone. |
| E-5 | How many regimes are there (eigengap)? Do nodal domains follow climate rather than elevation? How sharp is each boundary? |
| E-6 | Per-year partitions: which areas are stable cores and which are shifting transition belts? Is there a jump when the sensor changes? |
| E-7 | What holds up, and how far each confound has been ruled out. |

## 3. Data

All datasets come from the National Tibetan Plateau Data Center (TPDC,
https://data.tpdc.ac.cn) and are licensed CC BY-NC-SA 4.0. Raw data are not included in
this repository.

| Dataset | Authors | Content | Size | Used for |
|---|---|---|---|---|
| [Long-term series of daily snow depth dataset in China (1979–2025)](https://data.tpdc.ac.cn/en/data/df40346a-0202-4ed2-bb07-b65dfcda9368) | Tao Che, Liyun Dai, Xin Li | Daily snow depth on a 0.25° grid, from passive microwave (SMMR, SSM/I, SSMIS) | 362 MB | Vertices and features (2000–2014) |
| [Snowmelt onset time of High Mountain Asia (1979–2018)](https://data.tpdc.ac.cn/en/data/01be1b50-d9b6-4189-8aa0-9e005514b6d1) | Chuan Xiong, Jiancheng Shi, Ruzhen Yao, Yonghui Lei, Jinmei Pan | Yearly melt-onset date, plus a DEM band | 2.8 MB | Elevation (DEM band only) |
| [Different glacier status with atmospheric circulations in Tibetan Plateau and surroundings (1970s–2000s)](https://data.tpdc.ac.cn/en/data/439b01bd-1799-4171-b9ed-16e82ccc43df) | Tandong Yao | Supplementary tables of Yao et al. (2012) | 40 KB | Glacier locations and length change (Table S4), as an external check |

**Citation.** Cite each dataset together with the papers listed under it, and
acknowledge: *"This dataset is provided by the National Tibetan Plateau / Third Pole
Environment Data Center (https://data.tpdc.ac.cn/)."*

- Che, T., Dai, L., & Li, X. (2015). Long-term series of daily snow depth dataset in China (1979–2025). National Tibetan Plateau Data Center. https://doi.org/10.11888/Geogra.tpdc.270194
  - Che, T., Li, X., Jin, R., Armstrong, R., & Zhang, T. (2008). Snow depth derived from passive microwave remote-sensing data in China. *Annals of Glaciology*, 49, 145–154.
  - Dai, L., Che, T., & Ding, Y. (2015). Inter-calibrating SMMR, SSM/I and SSMI/S data to improve the consistency of snow-depth products in China. *Remote Sensing*, 7(6), 7212–7230.
  - Dai, L., Che, T., Ding, Y., & Hao, X. (2017). Evaluation of snow cover and snow depth on the Qinghai–Tibetan Plateau derived from passive microwave remote sensing. *The Cryosphere*, 11(4), 1933–1948.
- Xiong, C., Shi, J., Yao, R., Lei, Y., & Pan, J. (2020). Snowmelt onset time of High Mountain Asia (1979–2018). National Tibetan Plateau Data Center. https://doi.org/10.11888/Snow.tpdc.270307
  - Xiong, C., Shi, J., Cui, Y., & Peng, B. (2017). Snowmelt pattern over High-Mountain Asia detected from active and passive microwave remote sensing. *IEEE Geoscience and Remote Sensing Letters*, 14, 1096–1100.
  - Xiong, C., Yao, R., Shi, J., Lei, Y., & Pan, J. (2019). Changes in snow and ice melt timing over High Mountain Asia. *Chinese Science Bulletin*, 64(27), 2885–2893 (in Chinese).
- Yao, T. (2019). Different glacier status with atmospheric circulations in Tibetan Plateau and surroundings (1970s–2000s). National Tibetan Plateau Data Center. https://doi.org/10.11888/Glacio.tpdc.270100
  - Yao, T., Thompson, L., Yang, W., et al. (2012). Different glacier status with atmospheric circulations in Tibetan Plateau and surroundings. *Nature Climate Change*, 2, 663–667.

## Current Part II pipeline (EXP-0009)

The method above is the original project proposal. The executable experiment now follows the homework's E-1 → E-2 → E-3 → E-4 → E-5 order. It uses a 24-window × 9-bin signed temporal-gradient histogram plus snow occurrence, a complete pairwise feature-distance matrix, a kNN graph derived from that matrix, combinatorial connectivity diagnostics, and normalized-Laplacian partitions. This differs from the proposed 288-dimensional HOG/Jensen–Shannon setup; the change and each attempt are recorded in [the append-only experiment log](docs/experiment-log.md).

```text
pipeline.py                  Root command and all implemented-stage parameters
snow_divides/e1_features/   Daily input and time-gradient features
snow_divides/e2_interactions/  All-to-all weighted feature distance matrix
snow_divides/e3_knn/        kNN adjacency matrices and connectivity scan
snow_divides/e4_connectivity/  Degree and local clustering distributions
snow_divides/e5_laplacian/  Spectrum, Fiedler cut, spectral partitions and maps
snow_divides/e6_optional/   Optional methods, not run yet
snow_divides/e7_discussion/ Discussion stage; observations are in docs
outputs/                    New pipeline runs, each in a separate directory
feature_extraction/         Preserved EXP-0005/0007 code and historical output
```

```bash
uv sync --locked
uv run python pipeline.py --help
uv run python pipeline.py --output outputs/my-part2-run
```

The root `pyproject.toml` and `uv.lock` manage the new environment. The source snow archives remain in `data/snow depth/`; each new run writes E-1 through E-5 artifacts under one root `outputs/` directory. Existing `feature_extraction/outputs/` files are historical and remain in place. See [the pipeline report](docs/part2-pipeline-exp-0009.md) for the EXP-0009 k and cluster-count criteria, reproduction commands, saved matrix formats, and geographic figures.

For an offline rotatable 3D spectral embedding, a reordered distance/adjacency matrix view, and a guide to interpreting every figure, see [the EXP-0010 figure guide](docs/figure-guide-exp-0010.md). Use `uv run python export_views.py <completed-run> <new-output-directory>` to export these views from an existing run.

The [CS-521 Part II experiment report](docs/cs521-part2-report.md) presents E-1 through E-7, including the E-4 connectivity analysis added after EXP-0009, and states the authors' and AI assistant's contributions. A six-page PDF is in `outputs/exp-0011-part2-report/`.
