# AFL-UAV 独立复现说明

本模块把论文 *An Agentic Framework with LLMs for Solving Complex
Optimization Problems* 的工作流迁移到静态二维无人机路径规划。审计基线为公开仓库
`ZHANG-NI/AFL` 的提交 `602c6be26f98204e514adef982577a9d5d5c215f`。
该提交根目录未发现 `LICENSE`、`COPYING` 或 `NOTICE`，因此本项目没有复制其源码，
而是依据论文与可观察工作流重新实现接口、数据模型、编排和 UAV 求解器。

## 复现映射

| 论文/AFL 概念 | AFL-UAV 实现 | UAV 语义 |
| --- | --- | --- |
| Problem Description Generation | Generation Agent | 将地图、约束和目标整理为严格的 `UAVProblemDescription` |
| Problem Description Judgment | Judgment Agent + 硬校验器 | 同时做语义审查和必需字段、约束、来源哈希检查 |
| Code Generation | Generation Agent | 按八个函数阶段逐步生成求解器 |
| Code Judgment | Judgment Agent + AST 合同检查 | 同时审查算法意图、函数签名、导入和危险语法 |
| Error Analysis | Error Analysis Agent | 解释运行时错误或可信外部验收失败 |
| Complete Revision | Revision Agent | 在有界次数内重写完整求解器 |
| Problem/Code Buffer | `SolverBuffer` | 按问题合同寻址，只缓存曾通过外部验收的求解器 |
| Solver execution | `GeneratedSolverRunner` | 独立 Python 进程、超时、输出上限、精简环境 |
| Answer verification | `PathEvaluator` | 不信任生成代码的自报结果，重新检查端点、边界、净空和成本 |

八个代码阶段由原 VRP 版本迁移为：`read_problem`、`geometry`、`cost`、
`initial`、`destroy`、`repair`、`validation` 和 `main`。离线 Mock 求解器使用确定性
网格 A*、视线简化和有界 destroy/repair + 模拟退火，输出完整 JSON 结果。

## 三子任务、四角色闭环

1. 描述子任务：Generation Agent 生成问题描述；Judgment Agent 与硬校验器审查；
   不通过时由 Revision Agent 在固定预算内修订。
2. 代码子任务：Generation Agent 逐阶段生成；Judgment Agent 与 AST 合同审查；
   不通过时由 Revision Agent 修订该阶段。
3. 解答子任务：运行生成求解器，再由可信 `PathEvaluator` 独立验收；运行或验收失败时，
   Error Analysis Agent 诊断，Revision Agent 修订完整程序。

所有循环都有显式上限，不沿用上游实现中的无界 `while` 重试。角色调用、Prompt 版本、
Provider 记录、代码哈希、执行报告和外部验收结果都会写入运行产物。

## 运行

无网络、无 API key 的确定性复现：

```powershell
python -m uav_operator_evolution.cli afl-uav-demo `
  --provider mock `
  --config configs/agent_smoke.yaml `
  --run-id afl-uav-smoke
```

使用真实结构化模型只生成代码，默认不执行：

```powershell
python -m uav_operator_evolution.cli afl-uav-demo `
  --provider openai `
  --model <model> `
  --config configs/agent_smoke.yaml
```

只有显式加入 `--execute-untrusted-code` 才会执行真实模型生成的 Python。当前 AST 白名单、
隔离解释器、精简环境和超时只能降低意外风险，不是操作系统级沙箱；不应在含敏感数据的
主机上执行未经人工审查的生成代码。

## 产物与验收口径

每次运行写出：

- `problem_description.json`：带来源哈希的问题定义；
- `afl_uav_instance.json`：实际地图、目标权重和求解参数；
- `generated_uav_solver.py`：完整自包含求解器；
- `solver_output.json`：生成求解器的原始输出；
- `afl_uav_result.json`：角色事件、预算、执行与可信验收总结果。

成功必须同时满足：进程正常结束、输出满足 JSON 合同、路径航点有限且不超上限、端点正确，
以及项目既有 `PathEvaluator` 判定路径可行。缓存命中只减少代码生成调用；求解器仍会在当前
地图上重新执行并重新验收。

## 当前范围

- 已复现 Agent 拓扑、阶段化代码生成、判断/修订闭环、错误反馈和验证后缓存。
- 已将问题完整迁移为静态二维 UAV 路径规划，并复用项目已有圆形/矩形障碍与风险区模型。
- 尚未复现实验论文中的全部基准表、特定远程模型输出和成本统计。
- 尚未覆盖三维运动学、动力学约束、移动障碍、ROS/PX4/AirSim 或真实飞控。
- Mock 模式证明编排与工程合同可运行，不等价于证明真实 LLM 能稳定生成高质量新算法。
