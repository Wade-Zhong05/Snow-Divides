# E-4 组合连通性

课程要求：在各较大连通分量上绘制度分布与局部聚类系数分布。本轮先由 E-3 输出图和连通性数据，E-4 的分布分析尚未执行；不得把 E-3 的连通分量数图当成 E-4 完成证据。

EXP-0011 起新增实际 E-4 分析。对连通的参考 kNN 图计算无权度数，以及局部聚类系数 `C_i = 2T_i/[d_i(d_i-1)]`（`d_i<2` 时记 0）。边权只决定边是否存在，不进入这两个组合指标。流水线把分布图、逐点数组和摘要写入 `e4_connectivity/`。

从已完成的旧运行导出，不更改旧目录：

```bash
uv run python -m snow_divides.e4_connectivity.report \
  outputs/exp-0009-part2-default-v3 \
  outputs/exp-0011-e4-from-exp-0009
```
