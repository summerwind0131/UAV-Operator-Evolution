# uav2d Hidden Test-v2

`uav2d-hidden-test-v2` 是 Evolutionary AFL-UAV v1 冻结后的隐藏终测集。它只用于未来一次性最终评价，不能用于继续调参数、修改算子或选择指标。

## 当前状态

- 状态：`sealed_unrun`
- 地图：120 张，六类各 20 张
- `rooms_maze`：10 张 rooms、10 张 maze
- 地图尺寸：100×100
- 安全距离：2.0
- 连通性检查分辨率：2.0
- 起终点距离至少为地图对角线的 65%
- 与 `uav2d-v1` 的 content、终点、障碍布局、完整几何和地图种子重复数均为 0
- planner 执行次数：0
- API 调用次数：0
- 最终结果目录尚不存在

固定标识：

- Manifest 内容哈希：`ebfb307652363aae4537c0efae8891cbf08fd433b89018b9b9585529408237ac`
- 预注册 ID：`8d8dcbb568f25a5fd9da74afbe865c5576ce585e25fc37da36662ad1787c3c99`
- Seal ID：`cea0af043be0ba6d457df70870712ed76a61bd860c8f3bd14572d0c278e24602`

## 预注册矩阵

最终一次性运行已经固定为 6,960 条记录：

| 实验组 | 记录数 |
|---|---:|
| Dijkstra、A*、Theta* | 360 |
| RRT、RRT*、PRM、GA、PSO、DE、ACO/ACOR | 4,200 |
| 原始冻结 AFL-UAV | 600 |
| Evolutionary AFL-UAV v1 | 600 |
| 去 rooms 策略、固定长度两个关键消融 | 1,200 |

主假设、次假设、预算、共享 seed、可信超时规则、统计方法、bootstrap 和多重比较校正均保存在 `preregistration.json`。看到结果后不得更换指标、删除失败记录或修改 v1。

## 防误跑

数据根目录包含 `SEALED.json`。统一 `run_planner_benchmark()` 在发现该文件时，会在创建任何结果目录之前抛出 `PermissionError`。不要手工删除或修改该文件。

未来打开终测需要用户再次明确授权，并提供与本次预注册哈希匹配的 opening receipt。当前项目没有执行打开步骤。

只验证现有 seal、不运行 planner：

```powershell
$env:PYTHONPATH = "src"
python scripts/seal_hidden_test_v2.py --config configs/uav_hidden_test_v2.yaml
```

## 固定文件

- `configs/uav_hidden_test_v2.yaml`：生成和终测协议
- `data/benchmarks/uav2d-hidden-test-v2/manifest.json`：地图 manifest
- `data/benchmarks/uav2d-hidden-test-v2/seed_schedule.csv`：6,960 条预注册执行键
- `data/benchmarks/uav2d-hidden-test-v2/preregistration.json`：最终评价预注册
- `data/benchmarks/uav2d-hidden-test-v2/seal_receipt.json`：数据、代码、依赖和去重验收
- `data/benchmarks/uav2d-hidden-test-v2/SEALED.json`：运行锁
