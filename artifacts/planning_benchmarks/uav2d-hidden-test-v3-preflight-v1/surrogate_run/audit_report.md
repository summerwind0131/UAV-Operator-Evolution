# UAV2D Hidden Test-v3 final audit

Status: **passed**  
Records: **138**  
Paper method selected by preregistered rule: **evolutionary_afl_uav_v1**  
Reason: higher trusted feasible rate.

| Rank | Arm | Trusted feasible rate | Median trusted cost |
|---:|---|---:|---:|
| 1 | rrt | 1.0000 | 123.2429 |
| 2 | theta_star | 0.6667 | 114.5804 |
| 3 | astar | 0.6667 | 114.9365 |
| 4 | prm | 0.6667 | 118.8666 |
| 5 | aco_acor | 0.4167 | 111.7966 |
| 6 | dijkstra | 0.3333 | 110.4356 |
| 7 | pso | 0.2500 | 113.0598 |
| 8 | rrt_star | 0.1667 | 109.4909 |
| 9 | evolutionary_afl_uav_v1 | 0.1667 | 111.3803 |
| 10 | de | 0.1667 | 112.4282 |
| 11 | ga | 0.1667 | 112.4282 |
| 12 | evolutionary_afl_uav_v2 | 0.0000 | — |
| 13 | frozen_afl_uav | 0.0000 | — |

## Frozen comparisons

- H1_v2_vs_v1: 0W/5T/1L, half-tie rate 0.4167.
- H2_v2_vs_frozen_afl: 0W/6T/0L, half-tie rate 0.5000.
- H3_v2_vs_astar: 0W/2T/4L, half-tie rate 0.1667.
- H4_v2_vs_theta_star: 0W/2T/4L, half-tie rate 0.1667.
- E1_rooms_v2_vs_v1: 0W/1T/0L, half-tie rate 0.5000.
