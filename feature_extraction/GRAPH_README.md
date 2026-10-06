# 时间梯度二维直方图与雪特征图（EXP-0007）

本实验遵循 README 的研究对象：**节点是地理网格点**。建图只使用雪深时间特征；经纬度在建图完成后用于绘图、测量边的地理长度，以及比较原始栅格中的相邻点。没有半径 `r`，也没有把坐标拼入特征向量。

## 复现与调整

```bash
cd feature_extraction
uv sync --locked
uv run python run_graph.py --output outputs/my-graph-run
uv run python run_graph.py --help
```

自动扫描时间和梯度分箱，每个组合独立保存，不覆盖旧结果：

```bash
uv run python sweep_graph.py \
  --window-options 2,1 \
  --gradient-bin-options 9,7 \
  --k-value 21 \
  --output outputs/my-histogram-sweep
```

扫描脚本对每个组合运行同一套雪特征图流程，但关闭年度分区，以避免重复计算年度边界。它在**两次都符合筛选条件的格点**上计算调整兰德指数（ARI），比较分区稳定性。ARI 只量化两个计算结果是否相似，不能验证气候解释。也可以单独用 `--windows-per-month`、`--gradient-bins`、`--gradient-scale-cm-day`、`--time-smoothing-bins`、`--occurrence-weight`、`--minimum-snow-days`、`--k-values` 调整参数。`--k-values 10,21,26 --reference-k 21` 同时扫描三种连线数，地图使用 `k=21`。

输出目录必须不存在；已有实验不会被覆盖。Python 依赖由本目录 `pyproject.toml`、`uv.lock` 和 `.venv` 管理。

## 输入、筛选与缺日

- 默认雪年 2000/01–2013/14，共 14 个 8 月 1 日至次年 7 月 31 日的周期。读取 `../data/snow depth/snowdepth-YYYY.tar.gz` 中带头的主序列，仅覆盖 73–105°E、26–40°N 的 **57 × 129 = 7,353** 格点矩形。
- 四个缺失的 12 月 31 日补丁尚未配准核实；保留缺日，不插值，不跨越缺日计算梯度。每个 8 月 1 日也不跨雪年求差。
- 雪深单位 cm，产品负值按既有读取器转成 NaN。产品的零值沿用为“报告的零雪深”，不等于经过地面观测确认的无雪。
- 主图筛选：每格至少 10 个雪年各有 300 个有效日，且这些年份的平均雪深 `>0` 日数至少 30。运行范围少于 10 年时，要求所有选定雪年均达到 300 个有效日。年度分区另外要求该年至少 10 个雪日。因此当前筛选与项目 README 的“每年都至少 30 雪日”并不相同，须在论文章节明确；也未使用高原多边形、DEM 或冰川掩膜。

## 二维直方图与特征向量

对网格点 `i`、雪年 `y`、日期 `t`，时间梯度为 `g_i(t)=D_i(t)-D_i(t-1)`，单位 cm/day，归属于后一天。只在两日都有有效值、且至少一天 `D>0` 时计入。**梯度恰为零的有雪样本保留**；两日都无雪的零梯度不计入梯度直方图，而由出现率记录。

二维直方图的横轴是雪年内的时间窗口，纵轴是**有符号梯度**。默认每月两段，共 24 个时间窗口；每月根据该月的实际日数分段，所以闰年 2 月 29 日仍在 2 月的窗口里。为了让大幅变化不挤掉小幅变化，梯度先作单调变换 `a=atan(g/s)`，默认 `s=1 cm/day`，再在 `[-90°,90°]` 等宽分成 9 个 bin。角度是分箱坐标，原始量仍是 cm/day；改变 `s` 会改变厘米每天所落入的 bin。奇数个 bin 使零梯度处于中间 bin。按同一时间窗口跨年份汇总原始计数后，在每个有样本的窗口内归一化为概率 `p_i(w,b)`；空窗口保留全零向量。

另保存每窗口的积雪出现率 `o_i(w)=雪深>0 的有效日数/全部有效日数`。这防止“无雪无变化”和“有雪但深度稳定”产生相同表示。时间平滑先对直方图原始计数沿窗口轴进行高斯平滑（默认 `sigma=0.5` 个窗口，端点使用最近值），再做逐窗口归一化；出现率不平滑。

距离通过欧氏向量实现。令出现率权重为 `α`（默认 `0.5`），窗口数为 `W`，则

```text
d(i,j)^2 = (1-α)/W × Σ_w [ 1/2 × Σ_b (sqrt(p_i(w,b))-sqrt(p_j(w,b)))² ]
         + α/W × Σ_w [o_i(w)-o_j(w)]²
```

非空窗口的第一项是 Hellinger 距离平方；遇到空窗口时使用其全零向量，属于同一欧氏嵌入的定义，不能把这一项解释成两个完整概率分布之间的标准 Hellinger 距离。默认特征维数为 `24×9 + 24 = 240`。梯度概率按窗口归一化，因此变化**次数**未直接进入第一项；出现率及保存的原始计数仍可核查活动量。该描述子与之前逐公历月的 HOG 文件不同。

## 当前 kNN 建图顺序：三个不同的矩阵

这一节描述**已经运行的算法**，供方法审阅；不能把三种矩阵混为一谈。

1. **格点特征矩阵 `X`**：从 7,353 个矩形格点中筛出 3,941 个。每格的 24×9 条件梯度概率取平方根并缩放，再接上 24 个积雪出现率，得到 `X.shape == (3941, 240)`。第 `i` 行对应一个地理格点，但 `X` 中没有经纬度列。代码是 `time_histogram.feature_matrix()`；保存为 `snow_graph_results.npz` 的 `pooled_features`。
2. **两两特征距离 `d(i,j)`**：在 `X` 的行之间使用欧氏距离，数值上等于上一节给出的混合距离。当前代码调用 `sklearn.neighbors.NearestNeighbors(metric="euclidean")`，为每点搜索最多 26 个非自身近邻，再取指定 `k` 的前 `k` 个。**没有预先生成或保存完整的 `3941×3941` 距离矩阵**；近邻搜索只需要找每行最小的若干距离。代码是 `snow_graph.find_neighbors()`。
3. **加权邻接矩阵 `W`**：如果 `j` 在 `i` 的前 `k` 个雪特征近邻中，先生成有向候选 `i→j`。令 `σ_i=max(第 k 近邻距离, 10⁻⁶)`，候选权重为 `exp[-d(i,j)²/(σ_i σ_j)]`。把两个方向取最大值构成无向并集；没有候选的矩阵元素为零，自己到自己的对角元素强制为零。`W` 是**kNN 选边之后产生的稀疏矩阵**，保存为 `reference_graph_csr.npz`。默认 `k=21` 有 `3941×21=82,761` 个有向候选，经并集去重后为 59,373 条无向边，矩阵有 118,746 个非零元素。代码是 `snow_graph.build_graph()`。

后续谱分区才使用 `W`：先形成归一化拉普拉斯 `L=I-Δ⁻¹ᐟ²WΔ⁻¹ᐟ²`（`Δ` 为加权度的对角矩阵），再取特征向量聚类。经纬度只在这些步骤完成后用于地图与地理距离分析。当前流程是 `日雪深 → 逐格二维直方图 → X → 雪特征 kNN → W → 谱分区`。如果“先建立 matrix”指完整两两距离矩阵，可以由 `X` 显式计算并保存，近邻规则本身无需改变；如果指另一种按时间或地理关系定义的矩阵，则当前 `W` **不等价于那种先验关系矩阵**，需要另行设计。

## 数据和矩阵文件的具体位置

以下路径以项目根目录 `Snow-Divides/` 为起点：

| 文件 | 内容 |
| --- | --- |
| `data/snow depth/snowdepth-2000.tar.gz` 至 `snowdepth-2014.tar.gz` | 默认输入的原始逐日雪深档案；读取时不解压到磁盘。 |
| `feature_extraction/outputs/exp-0007-snow-graph-2000-2014-v5/snow_graph_results.npz` | `pooled_features` 即 `X`、`selected_pixel_index` 即 `X/W` 行到原网格行号的映射，以及直方图、坐标和分区。 |
| `feature_extraction/outputs/exp-0007-snow-graph-2000-2014-v5/reference_graph_csr.npz` | 默认 `k=21` 的稀疏加权邻接矩阵 `W`。 |
| `feature_extraction/outputs/exp-0007-snow-graph-2000-2014-v5/annual_graphs/` | 14 个年度邻接矩阵；每年的节点集合可能不同。 |
| `feature_extraction/outputs/exp-0007-snow-graph-2000-2014-v5/summary.json` | 参数、输入 SHA-256、缺日、各 `k` 统计、包版本和运行时间。 |

要**重新复现**而不覆盖既有结果，从项目根目录执行：

```bash
cd feature_extraction
uv sync --locked
uv run python run_graph.py --output outputs/my-graph-reproduction
```

读取当前 `X` 与 `W`，核对矩阵行到经纬度的映射：

```python
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

root = Path("outputs/exp-0007-snow-graph-2000-2014-v5")
with np.load(root / "snow_graph_results.npz") as data:
    x = data["pooled_features"]
    pixel_index = data["selected_pixel_index"]
    latitude = data["latitude"]
    longitude = data["longitude"]
w = load_npz(root / "reference_graph_csr.npz")
row, column = divmod(int(pixel_index[0]), longitude.size)
print(x.shape, w.shape, latitude[row], longitude[column])
```

这段代码应在 `feature_extraction/` 中运行；`X[i]` 和 `W` 的第 `i` 行是同一格点。原网格纬度从北向南递减，格点行号为 `row*129+column`。输出目录存在时 CLI 会拒绝覆盖。

## 连线、地理检验和年度边界

- 全时期图按上述雪特征距离找每点的 `k` 个最近邻，采用并集对称化；权重为 `exp[-d(i,j)^2/(σ_i σ_j)]`，其中 `σ_i` 是节点 `i` 的第 `k` 近邻距离，零距离用很小正值防止除零。计算归一化拉普拉斯前四个最小特征值；取前三个对应特征向量并逐行归一化，再用固定全一初始向量的特征求解器和随机种子 `0` 的 KMeans 分为三类。三类是探索设置，并未由外部气候标签确定；编号任意。
- 原图边可以跨越很远的地理距离。`02_graph_diagnostics.png` 给出 `k` 对连通性的影响，以及**建图后**才计算的边长分布（Haversine 公里）。这张距离图用于检查空间关系，不参与近邻选择。
- 每个雪年使用同样的特征定义、参考 `k` 与三类算法独立建图。年度节点须满足该年有效日数和至少 10 雪日。然后在原始 0.25° 网格的四邻接（东西、南北）上，数每个格点参与的“相邻两格类别不同”的次数，除以两格当年都有效时的比较次数，得到 `03_annual_boundary_frequency.png`。类别编号跨年可以置换，但同一年内两格是否同类不受置换影响。图上第二面板给出分母，防止低覆盖区域被误读。该统计是局部相邻分区差异，不能直接解释为真实气候边界移动概率。

## 输出文件

- `01_snow_features_and_regions.png`：雪特征图三类的经纬度地图，以及每类平均的**时间 × 梯度**条件分布；颜色为对数刻度，灰色是未通过筛选的格点。
- `02_graph_diagnostics.png`：`k` 扫描连通性、最大连通块比例、图边地理长度。
- `03_annual_boundary_frequency.png`：年度分区的四邻接差异比率及有效比较次数；`--no-annual` 时不生成。
- `04_cluster_occurrence_profiles.png`：每类逐窗口积雪出现率，以及有雪参与梯度对数/有效日数；用来判读零梯度。
- `snow_graph_results.npz`：所选格点的全局行号、特征矩阵、原始二维直方图计数、出现日数/有效日数、参考分区、逐年分区、边界分子分母和边长。行号为 `row*129+column`，纬度从北向南递减。
- `reference_graph_csr.npz`：参考 `k` 的加权稀疏邻接矩阵，可用 `scipy.sparse.load_npz()` 读取；其第 `j` 行对应 `snow_graph_results.npz` 中 `selected_pixel_index[j]`。
- `annual_graphs/snow_year_YYYY.npz`：每年的加权稀疏邻接矩阵。该年矩阵行顺序是年度 `annual_labels[year]` 按原网格 C-order 展平后，标签 `>=0` 的格点顺序；`--no-annual` 时不存在。
- `summary.json`：所有参数、定义、来源压缩包 SHA-256、缺日、筛选数、各 `k` 结果、逐年结果、版本与耗时。
- 扫描目录下每个组合保存自己的完整非年度运行文件；顶层 `sweep_summary.json` 和 `histogram_sensitivity.png` 汇总 ARI。

当前图仅说明**雪特征产生了何种分区**。尚未用 DEM、冰川表、高原边界或独立气候分区验证 H1，也未拆解传感器切换与雪深产品误差。年度边界图是进一步提出假设的依据，不能独立证明 H2。

检验积雪出现率对结果的影响，可用 `--occurrence-weight 0 --k-values 21 --reference-k 21 --no-annual` 运行纯梯度对照。该设置下出现率仍用于输出诊断图，但不进入图的距离。
