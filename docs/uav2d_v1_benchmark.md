# UAV2D-v1 路径规划基准

`uav2d-v1` 是与原有六臂算子进化实验相互独立的二维 UAV 路径规划基准。
它固定数据、预算、碰撞器、目标函数和硬约束验收器，用来比较完整路径规划器，
不会改变旧的 `run-baselines` 语义。

## 固定数据

- Train：180 张，六类各 30 张。
- Validation：60 张，六类各 10 张。
- Test：120 张，六类各 20 张。
- 六类为 `sparse`、`dense`、`corridor`、`clustered`、
  `rooms_maze`、`mixed`。
- `rooms_maze` 在三个 split 内分别严格平衡为
  rooms/maze = 15/15、5/5、10/10。
- 地图大小为 100×100，安全距离为 2.0，连通性栅格分辨率为 2.0。
- 主种子为 `20260725`。起终点、障碍布局和完整几何在 360 张地图间均不重复。

固定配置位于 `configs/uav_benchmark_v1.yaml`，数据位于
`data/benchmarks/uav2d-v1/`。Manifest 保存：

- `benchmark_id`
- `layout_subtype`
- `terminal_hash`
- `obstacle_layout_hash`
- `geometry_hash`

加载数据时会重新计算并核验这些哈希，拒绝重复或被修改的地图。

## Planner 分层

`planning_benchmarks` 不依赖旧的算子搜索执行器，分为三层：

1. Planner：只负责在给定预算和随机数发生器下寻找路径。
2. `BudgetedEvaluator`：统一记录目标评价、碰撞检查、节点扩展和规划耗时。
3. Runner：用独立的 `PathEvaluator` 重新验收返回路径，再负责重复实验和结果落盘。

实现的执行臂为：

- 确定性：`dijkstra`、`astar`、`theta_star`
- 随机性：`rrt`、`rrt_star`、`prm`、`ga`、`pso`、`de`、`aco_acor`
- 管线验证：`afl_uav_mock`
- 可选冻结求解器：`afl_uav`（需要 `--afl-artifact`）

`afl_uav_mock` 每张图只运行一次，所有记录都带
`research_claim_eligible=false`，不能用于 AFL-UAV 方法优劣结论。
冻结式 AFL-UAV Planner 已接入统一预算；离线 artifact 仍不具备研究结论资格。
真实 DeepSeek V4 Pro artifact 和离线 Evolutionary AFL-UAV 已接入；后者的结构、命令和 Validation 审计见 [evolutionary_afl_uav.md](evolutionary_afl_uav.md)。两者当前仍严格限制在 Train/Validation。

统一上限为每次规划 1 秒和 2000 次目标函数评价。种群算法使用 32 个个体、
10 个中间航点和最多 20 代；RRT/RRT* 使用步长 4.0、goal bias 0.1，
最多采样 1000 次；PRM 使用 180 个样本和 10 近邻。这些内部停止条件都低于
统一硬上限，最终排名只使用可信 `PathEvaluator` 的结果。

## 命令

生成或核验固定数据：

```powershell
python -m uav_operator_evolution.cli generate-maps `
  --config configs/uav_benchmark_v1.yaml
```

运行完整 Test Pilot：

```powershell
python -m uav_operator_evolution.cli benchmark-planners `
  --config configs/uav_benchmark_v1.yaml `
  --split test `
  --run-id uav2d-v1-test-pilot
```

运行每类一张图的小型 smoke：

```powershell
python -m uav_operator_evolution.cli benchmark-planners `
  --config configs/uav_benchmark_v1.yaml `
  --split test `
  --maps-per-class 1 `
  --time-limit 0.25 `
  --max-evaluations 50 `
  --repetitions 1 `
  --run-id uav2d-v1-smoke
```

`--planners`、`--maps-per-class`、`--time-limit`、`--max-evaluations` 和
`--repetitions` 的覆盖值都会写入运行元数据。

## 结果

每次运行生成：

- `benchmark_runs.csv`：每个 planner/map/seed 一行。
- `benchmark_paths.jsonl`：路径、状态与失败诊断。
- `benchmark_summary.json` 和 `benchmark_summary.csv`：总体及分地图类别统计。
- `benchmark_metadata.json`：Manifest hash、预算、参数、依赖和主机信息。

排序规则固定为：

1. 可行率降序；
2. 可行路径的统一总成本中位数升序。

失败、超时或预算耗尽不会通过惩罚成本伪装为可行结果；如果算法在到达预算时
已经找到 best-so-far，则该路径仍会经过同一硬约束验收，其运行状态仍保留为
`timeout` 或 `budget_exhausted`。
