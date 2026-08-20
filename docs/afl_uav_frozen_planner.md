# AFL-UAV 多 Provider 候选、冻结与 Validation

正式 `afl_uav` 与 `afl_uav_mock` 是不同执行臂。前者执行冻结的 Agent 生成求解器，后者只验证 benchmark 接口，不能用于真实 LLM 方法结论。

本阶段只使用 Train 生成和资格验收，只使用 Validation 比较；不要再访问 Test。OpenAI、DeepSeek、Gemini 是三条独立实验臂，使用同一套 Generation、Judgment、Revision Agent，不混用角色或在厂商间回退。

## 依赖和凭据

```powershell
python -m pip install -e ".[dev,llm]"

$env:OPENAI_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
```

可选依赖固定为 `openai>=2.46,<3` 和 `google-genai>=2.13,<3`。API key 只从上述三个环境变量读取，不进入候选、artifact、日志或异常文本。

本阶段固定模型：

- OpenAI：`gpt-4.1-2025-04-14`
- DeepSeek：`deepseek-v4-pro`
- Gemini：`gemini-2.5-pro`

## 第一步：只生成候选，不执行源码

三条命令分别生成独立候选：

```powershell
python -m uav_operator_evolution.cli generate-afl-uav-candidate `
  --config configs/uav_benchmark_v1.yaml `
  --provider openai `
  --model gpt-4.1-2025-04-14 `
  --run-id openai-gpt41-v1

python -m uav_operator_evolution.cli generate-afl-uav-candidate `
  --config configs/uav_benchmark_v1.yaml `
  --provider deepseek `
  --model deepseek-v4-pro `
  --run-id deepseek-v4pro-v1

python -m uav_operator_evolution.cli generate-afl-uav-candidate `
  --config configs/uav_benchmark_v1.yaml `
  --provider gemini `
  --model gemini-2.5-pro `
  --run-id gemini-25pro-v1
```

命令只保存 `candidate.json` 和 `candidate_solver.py`，并打印 `source_hash_to_approve`；不会启动生成代码。每个候选强制使用以下上限：

- 单次最多 16,384 输出 token
- 整个候选最多 250,000 token
- 最多 56 个逻辑 Agent 调用
- 每次调用 60 秒
- 最多重试 2 次

三家响应都要经过本地 Pydantic schema 校验。任何拒绝、空输出、截断、超时耗尽、token/调用预算耗尽或 SDK/凭据缺失都会失败关闭，不会换模型或退回 mock。

## 第二步：人工审查哈希后冻结

先人工查看 `candidate_solver.py`。确认源码后，把生成命令打印的完整哈希原样传给冻结命令：

```powershell
python -m uav_operator_evolution.cli freeze-afl-uav `
  --config configs/uav_benchmark_v1.yaml `
  --candidate artifacts/planning_benchmarks/afl_uav_candidates/openai-gpt41-v1 `
  --approve-source-hash <完整源码哈希> `
  --run-id openai-gpt41-frozen-v1
```

冻结器重新计算候选清单、源码和配置哈希，然后执行 AST 策略、CLI-v2 契约 smoke，并按固定顺序验收六张 Train 地图：

1. `train-000-sparse-c4b92a431b`
2. `train-001-dense-b533794b47`
3. `train-002-corridor-594730e0c7`
4. `train-003-clustered-fdcee6fa2f`
5. `train-004-rooms_maze-8a9244e3f3`
6. `train-005-mixed-5dee276124`

任一地图失败、计数器越界或源码发生变化都会拒绝冻结。源码修订后必须重新生成候选 ID 和源码哈希，再次人工批准。

v2 artifact 保存 Provider、请求模型、实际响应模型、响应 ID、token、重试、延迟、SDK 版本、候选/批准哈希和六图结果。只有真实 Provider、显式模型、完整成功的调用审计、哈希批准和六图全部通过时，`research_claim_eligible=true`。旧 v1 artifact 仍可加载，但不会自动升级研究资格。

## 第三步：四个 AFL artifact 一起跑 Validation

`--afl-artifact` 可重复使用，格式必须是 `ARM_ID=PATH`：

```powershell
python -m uav_operator_evolution.cli benchmark-planners `
  --config configs/uav_benchmark_v1.yaml `
  --split validation `
  --planners dijkstra astar theta_star rrt rrt_star prm ga pso de aco_acor afl_uav `
  --afl-artifact offline_v3=artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3 `
  --afl-artifact openai_gpt41=<OpenAI冻结artifact目录> `
  --afl-artifact deepseek_v4pro=<DeepSeek冻结artifact目录> `
  --afl-artifact gemini_25pro=<Gemini冻结artifact目录> `
  --time-limit 1 `
  --max-evaluations 2000 `
  --repetitions 5 `
  --run-id afl-uav-provider-matrix-validation-v1
```

完整矩阵应有 3,480 条唯一记录：

- 三个确定性算法：180
- 七个随机传统算法：2,100
- 离线 AFL-UAV v3：300
- 三个真实 Provider AFL-UAV：900

结果主键是 `planner + arm_id + map_id + repetition/seed`。所有最终路径仍由统一碰撞器、硬约束验证器和 `PathEvaluator` 可信复核。`benchmark_summary.json` 分开汇总每个 arm，并报告 AFL 相对 A*、Theta* 在双方均可行地图上的成对胜负；`benchmark_metadata.json` 汇总每个 artifact 的生成成功、修订数、调用/token/延迟和资格信息。

OpenAI GPT-4.1 始终是后续 Evolutionary AFL-UAV 主模型；不能根据本次 Validation 成绩改选主模型。
