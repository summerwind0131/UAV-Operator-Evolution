# UAV2D final-evaluation audit

Status: **passed**  
Records: **150**  
Timeouts are treated as failures. Ranking is feasibility first, then trusted cost.

| Rank | Arm | Feasible rate | Median feasible cost |
|---:|---|---:|---:|
| 1 | rrt | 1.0000 | 123.2429 |
| 2 | theta_star | 0.8333 | 117.7804 |
| 3 | astar | 0.6667 | 114.9365 |
| 4 | prm | 0.6667 | 118.8666 |
| 5 | rrt_star | 0.4167 | 111.4967 |
| 6 | dijkstra | 0.3333 | 110.4356 |
| 7 | aco_acor | 0.1667 | 112.4282 |
| 8 | de | 0.1667 | 112.4282 |
| 9 | ga | 0.1667 | 112.4282 |
| 10 | pso | 0.1667 | 112.4282 |
| 11 | evo_fixed_length | 0.0000 | — |
| 12 | evo_no_rooms_strategy | 0.0000 | — |
| 13 | evolutionary_afl_uav_v1 | 0.0000 | — |
| 14 | frozen_afl_uav | 0.0000 | — |

## Preregistered paired comparisons

- H1_evo_vs_frozen_afl: 0W/6T/0L, half-tie rate 0.5000.
- H2a_evo_vs_astar: 0W/2T/4L, half-tie rate 0.1667.
- H2b_evo_vs_theta_star: 0W/1T/5L, half-tie rate 0.0833.
- H3_rooms_evo_vs_no_rooms_strategy: 0W/1T/0L, half-tie rate 0.5000.
- H4_rooms_evo_vs_fixed_length: 0W/1T/0L, half-tie rate 0.5000.
