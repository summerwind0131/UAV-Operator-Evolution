# uav2d Hidden Test-v2

`uav2d-hidden-test-v2` 是 Evolutionary AFL-UAV v1 冻结后的一次性隐藏终测集。终测已在明确授权后完成；它不能再用于继续调参数、修改算子、选择指标或重新评价 v1。

## 最终状态

- 状态：`completed_audited`
- 打开时间：2026-08-17 10:20:45 +08:00
- 执行完成时间：2026-08-17 11:29:18 +08:00
- 地图：120 张，六类各 20 张；`rooms_maze` 中 rooms/maze 各 10 张
- 地图尺寸：100×100；安全距离 2.0；连通性检查分辨率 2.0
- 与 `uav2d-v1` 的 content、终点、障碍布局、完整几何和地图种子重复数均为 0
- 预注册执行矩阵：6,960 条；实际记录：6,960 条；唯一记录：6,960 条
- 14 个实验臂全部完成；API 调用次数：0
- 冻结审计状态：`passed`

固定标识：

- Manifest 内容哈希：`ebfb307652363aae4537c0efae8891cbf08fd433b89018b9b9585529408237ac`
- 预注册 ID：`8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99`
- Addendum ID：`ab1e83123806d7924b5439e02fea1ff6f837af03684d5249eddf34676380b3b6`
- Seal ID：`cea0af043be0ba6d457df70870712ed76a61bd860c8f3bd14572d0c278e24602`
- Opening ID：`e569d34d03272dd3debd23a0ea4b5331d1bb2a1631915c9dbaa54ed34611b76b`
- Execution receipt ID：`d5ddb5658846940f3b95b7b894dc304f5e03d0e536e4d0d8d9235661c48e4a46`
- Audit receipt ID：`6789dc4901faab1cccdc8af38668fdb09a2357c403f50eb1492810631e2ab838`

## 预注册矩阵

| 实验组 | 记录数 |
| --- | ---: |
| Dijkstra、A*、Theta* | 360 |
| RRT、RRT*、PRM、GA、PSO、DE、ACO/ACOR | 4,200 |
| 原始冻结 AFL-UAV | 600 |
| Evolutionary AFL-UAV v1 | 600 |
| 去 rooms 策略、固定长度两个关键消融 | 1,200 |

主假设、次假设、预算、共享 seed、可信超时规则、统计方法、bootstrap 和多重比较校正均由原 `preregistration.json` 固定。执行后不得更换指标、删除失败记录或修改 v1。

## 打开与审计链

用户以固定授权语句明确打开终测。打开程序校验了预注册、addendum、preflight、manifest、seed schedule 和原 seal，并将锁归档为 `SEALED.preopening.json`。因此当前没有 `SEALED.json` 是授权打开的预期结果，不是数据缺失。

专用 runner 完成全部 14 臂并写入 execution receipt；冻结 auditor 随后核验 6,960 条 seed schedule identity、原始运行矩阵、路径矩阵、规范化结果和报告哈希。终测已经结束，打开与运行命令仅作为历史实现保留，不应再次执行。

## 结论与文件

正式结果和限制见 [`hidden_test_v2_final_report.md`](hidden_test_v2_final_report.md)。关键本地证据：

- `data/benchmarks/uav2d-hidden-test-v2/opening_receipt.json`
- `data/benchmarks/uav2d-hidden-test-v2/SEALED.preopening.json`
- `artifacts/planning_benchmarks/uav2d-hidden-test-v2-final/execution_receipt.json`
- `artifacts/planning_benchmarks/uav2d-hidden-test-v2-final/audit_receipt.json`
- `artifacts/planning_benchmarks/uav2d-hidden-test-v2-final/audit_report.md`

大体积原始结果保持在 Git 忽略目录中，由规范化发布包及 SHA-256 固化，不直接写入 Git 历史。
