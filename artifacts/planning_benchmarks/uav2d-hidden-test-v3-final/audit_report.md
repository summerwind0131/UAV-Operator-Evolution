# UAV2D Hidden Test-v3 final audit

Status: **passed**  
Records: **6360**  
Paper method selected by preregistered rule: **evolutionary_afl_uav_v1**  
Reason: higher trusted feasible rate.

| Rank | Arm | Trusted feasible rate | Median trusted cost |
|---:|---|---:|---:|
| 1 | evolutionary_afl_uav_v1 | 1.0000 | 118.2734 |
| 2 | frozen_afl_uav | 1.0000 | 125.6071 |
| 3 | evolutionary_afl_uav_v2 | 0.9983 | 118.5303 |
| 4 | theta_star | 0.9833 | 123.7429 |
| 5 | rrt | 0.9717 | 132.8846 |
| 6 | astar | 0.9500 | 125.5255 |
| 7 | rrt_star | 0.9233 | 121.5953 |
| 8 | prm | 0.9000 | 129.2486 |
| 9 | dijkstra | 0.7000 | 119.6784 |
| 10 | ga | 0.6883 | 117.9920 |
| 11 | aco_acor | 0.5383 | 117.6304 |
| 12 | pso | 0.5283 | 116.0267 |
| 13 | de | 0.4967 | 116.5015 |

## Frozen comparisons

- H1_v2_vs_v1: 2W/33T/85L, half-tie rate 0.1542.
- H2_v2_vs_frozen_afl: 104W/16T/0L, half-tie rate 0.9333.
- H3_v2_vs_astar: 99W/15T/6L, half-tie rate 0.8875.
- H4_v2_vs_theta_star: 92W/16T/12L, half-tie rate 0.8333.
- E1_rooms_v2_vs_v1: 0W/2T/18L, half-tie rate 0.0500.
