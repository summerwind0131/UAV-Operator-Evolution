# Evolutionary AFL-UAV（离线进化增强）

Evolutionary AFL-UAV 不再次调用 LLM，也不修改冻结求解器源码。它把已经通过人工哈希批准和六类 Train 资格考试的 AFL-UAV 路径作为种子，在同一个规划预算内继续执行可信的路径进化。

## 方法结构

1. 冻结 AFL-UAV 生成一条可行种子路径。
2. 建立可变长度路径种群，而不是固定十个中间航点。
3. 使用五类路径算子：航点插入、删除、移动、交换和双亲交叉。
4. 保存八个质量—多样性精英：先保留统一总成本最小路径，再保留 Pareto 有效且几何形状不同的路径。
5. Pareto 目标同时包括路径长度、风险暴露、平滑度和中间航点数；最终提交路径仍按统一 `PathEvaluator` 总成本选择。
6. `rooms_maze` 使用较小步长的角点松弛、连续航点删除和确定性较低代数上限，改善狭窄门洞中的路径，同时给最终验证留出时间余量。

所有候选都经过统一碰撞检测器和目标函数评价，最终路径仍由 benchmark runner 独立进行硬约束验收。

## 运行方式

Train/Validation 可运行，Test 会在 artifact 加载或执行前被拒绝：

```powershell
$env:PYTHONPATH = "src"
python -m uav_operator_evolution.cli benchmark-planners `
  --config configs/uav_benchmark_v1.yaml `
  --split validation `
  --planners evolutionary_afl_uav `
  --time-limit 1 `
  --max-evaluations 2000 `
  --repetitions 5 `
  --evolutionary-afl-artifact deepseek_v4pro_evo=artifacts/planning_benchmarks/afl_uav_artifacts/deepseek-v4pro-frozen-strict-v2 `
  --run-id evolutionary-afl-uav-validation-v1
```

这条命令不会读取 `DEEPSEEK_API_KEY`，不会访问网络，也不会产生 token 费用。

## Validation v1 结果

- 60 张地图、每图 5 个共享种子，共 300 条，全部可行。
- 成本中位数：117.363；冻结 AFL-UAV 为 120.810。
- 按每张地图的五种子中位数比较：53 胜、7 平、0 负。
- `rooms_maze`：10 胜、0 平、0 负，中位相对改善约 1.91%。
- 54/60 张地图产生多个不同最终路径；每图不同路径数中位数为 5。
- 最慢运行 0.949 秒，最多 737 次可信目标评价。

结果和完整审计位于：

- `artifacts/planning_benchmarks/evolutionary-afl-uav-validation-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-validation-analysis-v1/`

这些结果只用于方法开发和 Validation 选择。由于参数和算子已经看过 Validation 表现，不能把它当作最终无偏论文结果；最终结论必须使用尚未打开的隐藏终测集。

## v1 冻结合同

Evolutionary AFL-UAV v1 已按源码和证据哈希冻结，冻结后禁止再根据 Validation 修改参数或算法：

- 核心源码 SHA256：`79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f`。
- 种群 32、质量—多样性精英档案 8、最多 20 代。
- 主变异算子插入/删除/移动/交换各占 0.25；交叉概率 0.40，额外变异概率 0.30。
- 可变长度路径，最多 64 个航点；最终仍选择统一可信总成本最低的路径。
- 冻结 artifact 同时锁定数据 manifest、基线 Validation receipt、消融/敏感性矩阵和实验分析 receipt。
- 允许的开发 split 只有 Train/Validation；`test_split_opened=false`，规划阶段 API 调用为 0。

冻结命令：

```powershell
$env:PYTHONPATH = "src"
python scripts/freeze_evolutionary_afl_v1.py
```

冻结目录：`artifacts/planning_benchmarks/evolutionary-afl-uav-methods/evolutionary-afl-uav-v1/`。

## 消融实验

固定五个消融臂，每个臂在 60 张 Validation 地图上运行 5 个共享种子，共 1,500 条：

| 消融臂 | 可信可行率 | 候选对完整 v1 的每图胜/平/负 | 候选−完整 v1 成本中位差 | rooms_maze 中位差 |
|---|---:|---:|---:|---:|
| 去质量—多样性档案 | 100% | 33/8/19 | -0.00275 | -0.0535 |
| 去交叉 | 100% | 26/7/27 | 0 | -0.0488 |
| 只移动航点 | 100% | 33/6/21 | -0.00241 | -0.1024 |
| 去 rooms_maze 专用策略 | 99.33% | 8/23/29 | 0 | +0.0936 |
| 固定长度种群 | 99.67% | 24/6/30 | +0.00006 | +0.1623 |

正差表示消融后成本更高，即完整 v1 更好。结果支持 rooms_maze 专用策略和可变长度在目标场景中的局部价值，但没有证明质量—多样性档案或交叉能独立降低本次 Validation 成本。完整组合相对原始冻结 AFL 仍是 53 胜、7 平、0 负；论文中应同时报告这些负结果，不能把所有组件都写成已被证明必要。

历史消融结果包含 3 条越过 1 秒边界的记录，其中 2 条由旧 runner 写成 `success`。原始数据没有重跑或删除；统一分析把三条都视为有效超时。runner 已增加可信边界后的状态重分类，后续实验不会继续产生这种旧标签。

## 参数敏感性

敏感性使用 OFAT（一次只改一个因素），不用于重选 v1 参数：

- 种群：16、24、32。
- 精英档案：4、8、12。
- 代数：6、12、20。
- 时间：0.25、0.5、1 秒。

种群/档案/代数共 9 个设置全部 100% 可信可行，运行级成本中位数处于 117.303–117.495。0.5 秒为 100% 可信可行、成本中位数 117.998；0.25 秒为 99.67% 可信可行、成本中位数 118.314。说明算法参数在所测邻域内稳定，时间缩短主要导致温和的成本退化；这不等于冻结参数是 Validation 最优。

运行与分析命令：

```powershell
$env:PYTHONPATH = "src"
python scripts/run_evolutionary_afl_matrix.py --section ablation
python scripts/run_evolutionary_afl_matrix.py --section sensitivity_algorithm
python scripts/run_evolutionary_afl_matrix.py --section sensitivity_time_025 --section sensitivity_time_050
python scripts/analyze_evolutionary_afl_experiments.py
python scripts/build_evolutionary_afl_report.py
```

主要证据：

- `artifacts/planning_benchmarks/evolutionary-afl-uav-ablation-validation-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-sensitivity-algorithm-validation-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-sensitivity-time025-validation-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-sensitivity-time050-validation-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-experiments-analysis-v1/`
- `artifacts/planning_benchmarks/evolutionary-afl-uav-technical-report-v1/report.html`
