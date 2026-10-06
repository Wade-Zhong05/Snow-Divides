# EXP-0009：按作业 E-1/E-2/E-3/E-5 组织的雪特征图流水线

本页记录 2026-10-05 的新流水线、参数选择和已运行结果。每次尝试与更改按时间追加到 [实验进程文档](experiment-log.md)。原 `feature_extraction/` 及其 EXP-0005/0007 结果保留为历史实现；新运行统一写入根目录 `outputs/`。

## 为什么增加 E-2

作业 PDF 第 9 页明确要求：E-1 特征向量，**E-2 全体样本两两加权关系矩阵**，E-3 从该矩阵生成 kNN 邻接矩阵。此前 EXP-0007 用 `NearestNeighbors` 直接找到近邻，未保存完整 E-2 矩阵。这次将两步分开：

```text
日雪深 → E-1 每点时间×梯度直方图及出现率 → 特征矩阵 X
      → E-2 完整距离矩阵 D (所有入选格点两两之间)
      → E-3 从 D 每行取 k 个邻居 → 稀疏加权邻接矩阵 W(k)
      → E-5 归一化拉普拉斯 L(k) → 谱分区/Fiedler cut → 地理网格图
```

默认使用 2000/01–2013/14 的 14 个雪年。E-1 保留 **3,941/7,353** 格点，生成 `X=(3941,240)`；24 个时间窗口 × 9 个有符号梯度 bin，加 24 个积雪出现率。梯度条件、`atan(g/1 cm/day)` 分箱、时间平滑 `0.5` 窗口、两部分各 `0.5` 的距离权重，与 [EXP-0007 方法说明](../feature_extraction/GRAPH_README.md)一致。四个未核实的 12 月 31 日仍缺失，不插值。E-2 把 `X` 每两行的欧氏距离保存为对称、对角为零的 `float32 D=(3941,3941)`，文件约 59 MiB；**数值小表示雪特征近**，没有地理距离。矩阵行列顺序由 E-1 文件 `selected_pixel_index` 确定。

E-3 对 `D` 每行先排除自己，再按距离稳定排序（同距时原网格行号较小者优先），取前 `k` 名作为有向候选。若任一方向选择对方，就保留无向边；权重为 `exp[-D(i,j)²/(σ_i σ_j)]`，`σ_i` 为节点 `i` 的第 `k` 近邻距离，最小截断 `10⁻⁶`。对称后的 `W(k)` 为 CSR 稀疏矩阵，自己到自己为零。经纬度仅在划分完成后用于展示。与旧 EXP-0007 相比，E-1 特征与格点顺序逐项相等，`k=21` 边数相同；由于第 21 名附近有同距候选，个别边在两种近邻实现中不同，但三类划分的 ARI 为 **1.0**。旧文件没有覆盖。

## 目录和复现

| 位置 | 用途与状态 |
| --- | --- |
| 根目录 `pipeline.py`、`pyproject.toml`、`uv.lock`、`.venv/` | 统一命令、默认参数与新环境。 |
| `snow_divides/e1_features/` | 读取主雪深档案、筛选、提取特征；已运行。 |
| `snow_divides/e2_interactions/` | 全体格点两两特征距离矩阵；已运行。 |
| `snow_divides/e3_knn/` | 从 E-2 矩阵选边、k 扫描和连通性；已运行。 |
| `snow_divides/e4_connectivity/` | 作业要求的度与局部聚类系数分布；**尚未执行**。 |
| `snow_divides/e5_laplacian/` | 拉普拉斯谱、类数扫描、Fiedler、三维嵌入和原网格地图；已运行。 |
| `snow_divides/e6_optional/` | 其他图划分方法；作业可选，本轮未执行。 |
| `snow_divides/e7_discussion/`、`docs/` | 记录观察及解释；气候验证仍待做。 |
| `feature_extraction/` | EXP-0005/0007 的代码、环境和历史输出，未搬迁或删除。 |

原始输入在 `data/snow depth/snowdepth-2000.tar.gz` 至 `snowdepth-2014.tar.gz`。从**项目根目录**运行：

```bash
uv sync --locked
uv run python pipeline.py --help
uv run python pipeline.py --output outputs/my-part2-run
```

输出目录必须事先不存在。默认输出会自动创建 `outputs/part2-<时间戳>/`。新运行统一包含 E-1、E-2、E-3、E-5 子目录和 `manifest.json`；历史 `feature_extraction/outputs/` 保留原位。一次调整 (k)、聚类数和 Fiedler 近零带宽的实际成功命令：

```bash
uv run python pipeline.py \
  --first-year 2000 --last-year 2000 \
  --k-values 2,3,8,16 --reference-k 16 \
  --cluster-counts 2,4 --reference-clusters 4 \
  --fiedler-bands 0,0.15 --eigen-count 8 \
  --output outputs/exp-0009-custom-controls
```

该自定义运行实际导出 `k=16` 的四类地图、8 个特征值和严格/15% 近零带 Fiedler 图。`--windows-per-month`、`--gradient-bins`、`--gradient-scale-cm-day`、`--time-smoothing-bins`、`--occurrence-weight`、`--minimum-snow-days` 也均由根命令控制；完整取值见 `--help`。

主运行产物在 `../outputs/exp-0009-part2-default-v3/`：

| 相对主运行目录的路径 | 内容 |
| --- | --- |
| `e1_features/features.npz` | `pooled_features=X`、直方图、出现率、经纬度、`selected_pixel_index`。 |
| `e2_interactions/distance_matrix.npy` | 完整 `D`，`float32 (3941,3941)`。 |
| `e3_knn/graph_k21.npz` 等 | 每个指定 (k) 的 CSR 邻接矩阵 `W(k)`。 |
| `e3_knn/k_scan.json`、`connectivity_and_fiedler.png` | 连通分量、边数、连通后的 (lambda_2(k))、与参考分区的 ARI。 |
| `e5_laplacian/spectral_results.npz` | 前 32 个特征值/向量、候选类数与 Fiedler 带宽及逐格标签。 |
| `e5_laplacian/01…04…png` | 谱、原网格类别图、Fiedler 地图、三维嵌入；另有单独的 `02_reference_clusters.png`。 |
| `manifest.json` | 参数、阶段状态、输入 SHA-256、缺日、统计与依赖版本。 |

读取 E-2 和参考图时，`D[i,j]`、`W[i,j]` 与 `X[i]` 的行号共用同一 `selected_pixel_index`：

```python
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

run = Path("outputs/exp-0009-part2-default-v3")
with np.load(run / "e1_features/features.npz") as data:
    x = data["pooled_features"]
    pixel_index = data["selected_pixel_index"]
d = np.load(run / "e2_interactions/distance_matrix.npy", mmap_mode="r")
w = load_npz(run / "e3_knn/graph_k21.npz")
print(x.shape, d.shape, w.shape, pixel_index.shape)
```

## (k) 的选取标准与本次扫描

默认扫描 `k=2,3,4,8,10,16,21,26,32,64`，覆盖老师示例范围 `[2,64]`，加入 `3` 定位连通阈值，以及 thesis 预设的 `21–26`。本次 `k=2` 有 **13** 个连通分量，`k=3` 起连通，因此扫描到的最小连通值为 **3**。但“刚好连通”不足以决定研究图：(k) 太小会产生弱连接和不稳定划分，太大会把原本分离的雪季形状混合。对连通图继续比较 (lambda_2(k)) 与参考三类分区的 ARI：

| (k) | 连通分量 | (lambda_2(k)) | 相对 (k=21) 三类 ARI |
| ---: | ---: | ---: | ---: |
| 2 | 13 | 不定义为连通图 Fiedler 值 | 不比较 |
| 3 | 1 | 0.00244 | 0.672 |
| 10 | 1 | 0.00706 | 0.792 |
| 16 | 1 | 0.00915 | 0.890 |
| **21** | **1** | **0.01048** | **1.000** |
| 26 | 1 | 0.01168 | 0.941 |
| 32 | 1 | 0.01290 | 0.905 |
| 64 | 1 | 0.01767 | 0.747 |

因此暂以 **`k=21`** 为可复现的参考：它处于 README 原定的 21–26 区间，已连通，且到 26 的三类划分 ARI 为 0.941；16–32 一带也较接近。`k=64` 虽有更大的 (lambda_2)，但分区已明显改变，不能简单通过最大化 (lambda_2) 选 (k)。`k=21` 的并集图有 59,373 条无向边，平均度约 30.1，**并集后平均度并非 21**。这是探索性选择；更换特征、筛选或时间范围应重新扫 (k)，命令行的 `--k-values` 和 `--reference-k` 可调。完整 10 个 (k) 的结果在 `k_scan.json`。

![从 E-2 矩阵生成的 kNN 图：连通性与 Fiedler 值](../outputs/exp-0009-part2-default-v3/e3_knn/connectivity_and_fiedler.png)

## 拉普拉斯分区、类数与 Fiedler cut

参考图 `W(21)` 的归一化拉普拉斯为 `L=I−Δ⁻¹ᐟ²WΔ⁻¹ᐟ²`，其中 `Δ` 是加权度对角矩阵。计算前 **32** 个最小特征值；候选类数 `m=2,3,4,5,6` 用前 `m` 个特征向量按行归一化后做固定随机种子的 KMeans。默认 `m=3` 来自项目对“西风、过渡、季风”三个可能区域的假设，**不是已经由谱证明的类别数**。`λ₃` 后的 eigengap 约 **0.01144**，`λ₂` 后约 **0.01137**，`λ₅` 后约 **0.01112**；三者接近，谱在这组参数下没有给出唯一清楚的三类证据。`--cluster-counts` 控制扫描候选，`--reference-clusters` 控制默认图及 (k) 稳定性比较。

![前 32 个拉普拉斯特征值与候选 eigengap](../outputs/exp-0009-part2-default-v3/e5_laplacian/01_laplacian_spectrum.png)

![默认三类谱聚类回填到原始地理网格](../outputs/exp-0009-part2-default-v3/e5_laplacian/02_reference_clusters.png)

![二至六类的原地理网格对照](../outputs/exp-0009-part2-default-v3/e5_laplacian/02_cluster_maps.png)

Fiedler 向量 `q₂` 的**严格 nodal sign cut** 以零为阈值，把节点分成正、负两侧；这是两侧划分，与默认三类 KMeans 是不同的图划分。为观察近零值可能形成的过渡带，另扫描 `|q₂|` 的 0%、10%、20% 分位数，把低于阈值的格点标为“待定”，**并未移动正负符号边界**。10% 与 20% 的绝对阈值分别约为 0.00173、0.00348，数值只属于该归一化特征向量，没有雪深单位；改变图后须重新计算。`--fiedler-bands` 控制该扫描。向量整体符号本可翻转，代码以最大绝对值处为正固定显示方向。

![Fiedler 正负切分和近零带回填到原始网格](../outputs/exp-0009-part2-default-v3/e5_laplacian/03_fiedler_maps.png)

![由 q2、q3、q4 构成的三维拉普拉斯嵌入](../outputs/exp-0009-part2-default-v3/e5_laplacian/04_embedding_3d.png)

彩色地图用原始网格位置展示，但经纬度没有进入 E-2 距离或 E-3 连边。类别编号可置换，不能将颜色直接命名为气候区。尚未用高原边界、DEM、冰川掩膜或独立气候标签验证三类，也未完成作业 E-4 的度/局部聚类系数分布。Fiedler 近零带是算法不确定区域的探索标记，不等于物理气候过渡带的测量值。
