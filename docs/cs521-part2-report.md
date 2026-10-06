# Snow-Season Feature Graphs and Preliminary Spectral Divides on the Tibetan Plateau

**CS-521 Homework 1, Part II - Experiment report**  
**Report authors:** Haoyang Feng and Feiyang Zhong  
**Date:** October 5, 2026  
**AI assistance:** OpenAI Codex assisted with data inspection, Python implementation, figure generation, experiment logging, and preparation of this report draft. The research question and iterative methodological decisions were directed by the human authors. The AI contribution is described in detail in Section 8.

## Abstract

We ask whether similarity in snow-season timing and daily snow-depth changes produces spatially meaningful graph partitions over the Tibetan Plateau. Daily passive-microwave snow-depth grids for 14 snow years were converted into a 240-dimensional feature per eligible grid cell. We constructed the full pairwise feature-distance matrix, a series of weighted k-nearest-neighbor (kNN) graphs, combinatorial connectivity summaries, and normalized-Laplacian partitions. The reference graph at `k=21` has 3,941 vertices, 59,373 undirected edges, and one connected component. Its three-class spectral partition displays geographic structure, while the Fiedler vector gives an exploratory two-sided cut. However, the eigengap does not uniquely favor three classes, and the result changes substantially when temporal resolution or snow-occurrence weighting changes. We therefore treat the mapped classes as snow-feature groups, not established climate regions.

## 1. Question, data, and feature construction (E-1)

The project question is whether snow-season shapes alone can reveal broad divisions associated with the plateau's differing circulation regimes, and whether any boundary is sharp or transitional. This is motivated by published evidence of heterogeneous glacier behavior and atmospheric circulation across the plateau (Yao et al., 2012), but that literature does not provide ground-truth class labels for our grid cells.

We used the long-term daily snow-depth product stored in `data/snow depth/snowdepth-2000.tar.gz` through `snowdepth-2014.tar.gz`. The analysis covers snow years 2000/01–2013/14, August 1, 2000 through July 31, 2014, in the 73–105°E, 26–40°N rectangle. The source grid spacing is 0.25° in both directions; the study rectangle contains 57 × 129 = 7,353 cells. This is a geographic rectangle rather than a verified Tibetan Plateau polygon. The 5,113-day calendar interval contains 5,109 available main-series daily files. Four December 31 dates are missing from those annual archives; we left them missing and did not calculate a temporal gradient across a missing day. Snow depth is a remotely retrieved value in centimeters, not measured snowfall or snow-water equivalent. Decimal storage precision and grid spacing do not imply equivalent physical accuracy; passive-microwave retrieval, temporal filling, terrain, and sensor differences matter (Che et al., 2008; Dai et al., 2015, 2017).

For each cell, we calculated daily change `g_t = depth_t − depth_{t−1}` only when both days are valid, at least one has positive reported snow depth, and the pair does not cross a snow-year boundary. We mapped `g_t` to `atan(g_t / (1 cm day⁻¹))`, then counted it in nine signed bins within each of 24 time windows per snow year. Counts were pooled across years, smoothed along the time-window axis with Gaussian width 0.5 window, and normalized within each window. We also retained the fraction of valid days reporting snow in each window. The resulting 240-dimensional vector contains 24 × 9 = 216 gradient-histogram coordinates and 24 snow-occurrence coordinates. In the Euclidean embedding, the gradient-distribution and occurrence terms each receive weight 0.5.

We retained cells with at least 300 valid days in at least 10 snow years and at least 30 reported snow days per qualifying year **on average**. This left 3,941 cells. The 24-window and nine-bin choices, filtering thresholds, and mixture weight are hyperparameters. The executed feature differs from the original README proposal of a 288-dimensional HOG with Jensen–Shannon distance; the experiment log records that change.

## 2. Pairwise interaction matrix (E-2)

For eligible feature vectors `x_i`, we computed every Euclidean feature distance `D_ij = ||x_i − x_j||₂`. The resulting `3,941 × 3,941` float32 matrix is symmetric and has a zero diagonal. Smaller values mean more similar snow-feature vectors; `D` is not geographic distance. The histogram part uses square-rooted per-window probabilities, yielding a Hellinger-style contribution to Euclidean distance, while the occurrence coordinates retain differences between snow-free and snow-present windows. Rows and columns are indexed by `selected_pixel_index` in `e1_features/features.npz`.

The left panel of Figure 1 shows `D` after grouping cells by the reference spectral-cluster labels. This sorting is for display only; `D` was calculated before clustering. The orange square boundaries are imposed by the row and column order, whereas the color variation within and between squares is observed.

![Figure 1. The same cluster order applied to the full feature-distance matrix D (left) and the nonzero pattern of the weighted kNN adjacency matrix W (right).](../outputs/exp-0010-views-from-exp-0009/06_clustered_matrices.png)

## 3. kNN graphs and connectivity across k (E-3)

For each row of `D`, we excluded the diagonal, stably ranked all other cells by distance, and selected the first `k`. When distances tie, lower original grid index takes priority. A pair becomes an undirected edge when either endpoint selects the other. Its weight is `W_ij = exp[−D_ij²/(σ_i σ_j)]`, where `σ_i` is cell `i`'s distance to its `k`th neighbor; zero-distance scales are bounded below by `10⁻⁶`. The sparse symmetric matrix `W` has no self-edges. Geographic position did not enter feature extraction, distance, or edge selection.

We scanned `k = 2, 3, 4, 8, 10, 16, 21, 26, 32, 64`, covering the assignment's illustrative range of 2–64. At `k=2`, there are 13 connected components. At every scanned `k≥3`, the graph is connected, so the observed connection threshold is `k_c=3`. The `k=21` reference lies in the project's initially proposed 21–26 range and gives one connected graph with 59,373 undirected edges. Its mean degree is 30.13, above 21 because union symmetrization adds incoming selections. The three-class labels at `k=21` have adjusted Rand index (ARI) 0.941 relative to `k=26`, 0.890 relative to `k=16`, and 0.905 relative to `k=32`; ARI here measures internal stability, not climate-label accuracy. At `k=64`, ARI relative to `k=21` falls to 0.747. We chose `k=21` as an exploratory reference, not as an optimized or uniquely correct value.

![Figure 2. Connected-component count and normalized-Laplacian Fiedler value across the scanned k values.](../outputs/exp-0009-part2-default-v3/e3_knn/connectivity_and_fiedler.png)

## 4. Combinatorial connectivity of the reference graph (E-4)

The reference graph is connected, so its single 3,941-vertex component is the large component analyzed here. We computed unweighted degree `d_i` from the nonzero pattern of `W` and the local clustering coefficient `C_i = 2T_i/[d_i(d_i−1)]`, where `T_i` counts triangles incident to vertex `i`. Edge weights determine which connections exist but do not enter these two combinatorial summaries.

The degree distribution has a lower bound of 21 imposed by union kNN construction, a median of 27, mean of 30.13, and a right tail reaching 95. It is right-skewed rather than approximately symmetric; this histogram alone does not establish a power law. Local clustering coefficients are concentrated in one broad central mode: median 0.338, mean 0.352, interquartile range 0.274–0.412, and no zeros in this reference graph. The distribution has a moderate right tail. These are graph-topology summaries, not direct measurements of geographic boundaries.

![Figure 3. Degree and unweighted local clustering-coefficient distributions for the connected k=21 reference graph.](../outputs/exp-0011-e4-from-exp-0009/degree_and_clustering.png)

## 5. Normalized-Laplacian spectrum and partitions (E-5)

For each connected scanned graph, we formed the symmetric normalized Laplacian `L = I − Δ⁻¹ᐟ² W Δ⁻¹ᐟ²`, with `Δ` the weighted-degree diagonal matrix. The Fiedler value `λ₂(k)` rose from approximately 0.00244 at `k=3` to 0.01048 at `k=21` and 0.01767 at `k=64` (Figure 2). Larger `λ₂` does not by itself choose `k`, since adding edges also changes the partition.

For `k=21`, we computed and plotted the 32 smallest eigenvalues. The gaps after candidate counts `m=2`, `m=3`, and `m=5` are about 0.01137, 0.01144, and 0.01112, respectively. They are close: this spectrum does **not** identify a uniquely well-separated nonzero simple eigenvalue or uniquely justify three regimes. To examine a reproducible nodal-domain cut anyway, we selected the nonzero Fiedler eigenvector `q₂` and labeled cells by its strict sign. This gives 2,375 negative and 1,566 positive cells. Excluding the 10% or 20% smallest absolute `q₂` values marks 395 or 789 cells as near-zero uncertainty; it does not move the underlying zero-sign boundary. A Fiedler near-zero band is an algorithmic sensitivity device, not a measured climate-transition width.

![Figure 4. The first 32 normalized-Laplacian eigenvalues and candidate eigengaps.](../outputs/exp-0009-part2-default-v3/e5_laplacian/01_laplacian_spectrum.png)

![Figure 5. The strict Fiedler sign domains and exploratory 10% and 20% near-zero bands on the original geographic grid.](../outputs/exp-0009-part2-default-v3/e5_laplacian/03_fiedler_maps.png)

The three-dimensional spectral display places each cell at `(q₂(i), q₃(i), q₄(i))` and colors it by the sign of `q₂` (Figure 6). These axes are graph eigenvector coordinates, not longitude, latitude, or snow depth. Separation along the `q₂` axis is partly definitional because color also encodes `q₂` sign. The [offline interactive version](../outputs/exp-0010-views-from-exp-0009/05_embedding_3d_interactive.html) permits rotation, zooming, hover inspection of source-grid coordinates, and switching to the three-class colors; browser dragging was not independently tested in the current computer-use environment.

![Figure 6. Static three-dimensional Laplacian embedding, colored by the strict Fiedler sign domains.](../outputs/exp-0009-part2-default-v3/e5_laplacian/04_embedding_3d.png)

Separately from the two Fiedler nodal domains, we applied row-normalized spectral KMeans to the first `m` eigenvectors for `m=2,...,6`, using a fixed random seed. The exploratory three-class map has class sizes 1,951, 1,611, and 379 (Figure 7). It shows spatial organization even though no geographic coordinates entered graph formation: one small class is concentrated largely in the northwest, and the two larger classes occupy broad, partly interleaved areas. Labels are arbitrary and cannot yet be assigned to westerly, transition, or monsoon regimes. In Figure 1, within-class blocks of `W` are denser than cross-class blocks, but that is an internal consistency property because the labels were obtained from a graph derived from `D`. It is not independent climate validation.

![Figure 7. Exploratory three-class spectral partition placed back on the original 0.25° geographic grid; gray cells did not pass the eligibility filter.](../outputs/exp-0009-part2-default-v3/e5_laplacian/02_reference_clusters.png)

## 6. Optional methods (E-6)

The assignment labels other graph-clustering methods as optional. We did not run an additional algorithm. The two Fiedler domains and the spectral KMeans maps are both analyses of the same normalized-Laplacian graph, and should not be portrayed as independent external validation.

## 7. Preliminary findings, uncertainty, and next tests (E-7)

The snow-feature graph becomes connected at a small `k`; its reference graph has a right-skewed degree distribution and a substantial concentration of local triangle closure. The Fiedler cut and three-class spectral map show spatial patterns despite location-free graph construction. These are promising structures for further study, but the current experiment does not establish climate boundaries or annual boundary motion.

Several checks limit interpretation. The three-class result has no uniquely compelling eigengap. In the earlier sensitivity experiment, changing 24 time windows to 12 while keeping nine gradient bins reduced three-class agreement to ARI 0.423; removing the snow-occurrence component reduced agreement to ARI 0.326. The current groups therefore depend materially on feature definition. The selected domain is a rectangular window without a verified plateau or glacier mask; high-elevation permanent snow/ice and terrain effects may influence the small persistent-snow class. The product contains missing days, sensor and retrieval uncertainty, and no independent climate, elevation, or glacier labels were used in this analysis. We have not tested the hypothesis of a shifting transition belt with matched per-year graphs in the new Part II pipeline.

The next scientific tests are to compare partition labels with independent geography and climate evidence, vary the temporal and snow-occurrence feature choices, and evaluate annual partitions with consistent cell sets. Those checks are needed before using climatic names or interpreting the near-zero Fiedler band as a physical transition region.

## 8. Reproducibility and contribution statement

The main run and configuration are in `outputs/exp-0009-part2-default-v3/manifest.json`; it contains parameter values, source-file SHA-256 hashes, missing dates, software versions, and stage summaries. E-4 was computed subsequently from its saved `graph_k21.npz` without changing the original run. The interactive and clustered-matrix figures were likewise exported from saved E-1/E-2/E-3/E-5 files. All attempts, corrections, and limitations are appended to `docs/experiment-log.md`.

From the repository root, the relevant commands are:

```bash
uv sync --locked
uv run python pipeline.py --output outputs/your-new-run
uv run python -m snow_divides.e4_connectivity.report \
  outputs/exp-0009-part2-default-v3 outputs/your-new-e4-export
uv run python export_views.py \
  outputs/exp-0009-part2-default-v3 outputs/your-new-view-export
uv run pytest -q tests
```

**Human authors:** Haoyang Feng and Feiyang Zhong directed the research question, chose the sequence of analyses, reviewed figures and interpretations, and retain responsibility for checking the final report. **AI participation:** OpenAI Codex inspected the dataset and literature already present in the project, implemented and tested the Python feature/graph/spectral and E-4 analysis code, generated and checked figures, maintained the append-only experiment log, and drafted this report. The AI did not supply independent climate labels or conduct physical validation. The statement distinguishes assistance from evidence: reported numerical results come from the saved runs and cited source files, not from the AI's authority.

## References

- CS-521/CS-321/Math-462, Fall 2026. *Homework 1, Part II: Experiment*, p. 9, [assignment PDF](CS521D_HW1_26Fall%20(1).pdf).
- Che, T., Li, X., Jin, R., Armstrong, R., & Zhang, T. (2008). Snow depth derived from passive microwave remote-sensing data in China. *Annals of Glaciology*, 49, 145–154. [Local PDF](papers/Che_2008_Snow_depth_China.pdf).
- Dai, L., Che, T., & Ding, Y. (2015). Inter-calibrating SMMR, SSM/I and SSMI/S data to improve the consistency of snow-depth products in China. *Remote Sensing*, 7, 7212–7230. [Local PDF](papers/Dai_2015_Inter_calibrating_snow_depth.pdf).
- Dai, L., Che, T., Ding, Y., & Hao, X. (2017). Evaluation of snow cover and snow depth on the Qinghai–Tibetan Plateau derived from passive microwave remote sensing. *The Cryosphere*, 11, 1933–1948. [Local PDF](papers/Dai_2017_Qinghai_Tibetan_snow_evaluation.pdf).
- Yao, T., Thompson, L., Yang, W., et al. (2012). Different glacier status with atmospheric circulations in Tibetan Plateau and surroundings. *Nature Climate Change*, 2, 663–667. [Local PDF](papers/Yao_2012_Glacier_status_circulations.pdf).
