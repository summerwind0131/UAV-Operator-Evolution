# UAV2D Hidden Test-v2 final report

状态：最终、已审计、不可重新调参  
执行日期：2026-08-17  
审计矩阵：6,960 条预注册记录，6,960 条唯一结果  
审计规则：超时计失败；先按可行率、再按可信可行成本排序

## 1. 主要结论

Evolutionary AFL-UAV v1 在配对成本比较中表现出明确优势，但其可信可行率为 `0.9967`，低于 frozen AFL-UAV 和去 rooms 策略的 `1.0000`，也低于固定长度消融的 `0.9983`。因此在预注册的可行性优先排名中，v1 位列第 4，不能只根据更低成本宣称总体胜出。

两个关键消融均未支持原设计：带 rooms 策略的 v1 对去 rooms 策略为 7W/0T/13L，对固定长度策略为 8W/0T/12L。这些负结果与正结果同等保留，不能据此修改 v1 后重跑同一终测。

## 2. 全部实验臂排名

| 排名 | 实验臂 | 可信可行率 | 可行结果成本中位数 |
| ---: | --- | ---: | ---: |
| 1 | `evo_no_rooms_strategy` | 1.0000 | 121.1629 |
| 2 | `frozen_afl_uav` | 1.0000 | 126.1536 |
| 3 | `evo_fixed_length` | 0.9983 | 123.7847 |
| 4 | `evolutionary_afl_uav_v1` | 0.9967 | 120.8939 |
| 5 | `rrt` | 0.9650 | 129.9417 |
| 6 | `prm` | 0.9050 | 129.6821 |
| 7 | `theta_star` | 0.8917 | 122.3878 |
| 8 | `astar` | 0.8167 | 121.6464 |
| 9 | `rrt_star` | 0.6033 | 118.8585 |
| 10 | `dijkstra` | 0.5333 | 117.6242 |
| 11 | `pso` | 0.5017 | 117.7633 |
| 12 | `de` | 0.5000 | 119.9257 |
| 13 | `aco_acor` | 0.5000 | 120.6634 |
| 14 | `ga` | 0.4950 | 117.5308 |

成本只在可信可行结果内解释，不能用低可行率方法的成本中位数绕过首要可行性指标。

## 3. 预注册配对比较

- H1，v1 vs frozen AFL-UAV：113W/7T/0L，half-tie rate `0.9708`。
- H2a，v1 vs A*：110W/7T/3L，half-tie rate `0.9458`。
- H2b，v1 vs Theta*：104W/7T/9L，half-tie rate `0.8958`。
- H3，v1 vs 去 rooms 策略：7W/0T/13L，half-tie rate `0.3500`。
- H4，v1 vs 固定长度策略：8W/0T/12L，half-tie rate `0.4000`。

## 4. 审计与内容身份

| Artifact | SHA-256 / identity |
| --- | --- |
| Opening receipt | `e569d34d03272dd3debd23a0ea4b5331d1bb2a1631915c9dbaa54ed34611b76b` |
| Execution receipt | `d5ddb5658846940f3b95b7b894dc304f5e03d0e536e4d0d8d9235661c48e4a46` |
| Audit receipt | `6789dc4901faab1cccdc8af38668fdb09a2357c403f50eb1492810631e2ab838` |
| Raw run matrix | `f76b5d21e281cab03097e10963c1522a3b46bc0056a4a85a534c0a32a54ef835` |
| Raw path matrix | `93a250c3a4884e8b65545014a1592f61dc15a9bc106bd42045cfae94c93413fc` |
| Normalized run matrix | `3d45dce0c88a494fb22616beda0eb2356ccbc64490f0ae60941faa0bceb8afa2` |
| Structured audit report (`audit_report.json`) | `93fb31bf94dee61d05cf26b8fc0532a1a2aacaad66bbcf66cc3f5b79bdafedc8` |
| Human-readable audit report (`audit_report.md`) | `7f842573982d92d412e83467459f0ab0d549099bcf1a6a34debdc53396ed40f5` |
| Audit summary | `6aaabca2123f241999e7ef8999c56a2a1a042eb976e19047ea0fa034921ad46a` |

## 5. 研究限制

- 结果只支持冻结的二维 UAV 环境、预算、14 个实验臂和统计协议。
- 轨迹诊断与算子机制是受控关联证据，不等于因果证明。
- v1 的成本优势不消除其可行率退化；总体结论必须保留这一权衡。
- Hidden Test-v2 已永久退出开发闭环。后续通用化和迁移研究必须使用新的 train/validation，以及独立封存的 test 数据。
