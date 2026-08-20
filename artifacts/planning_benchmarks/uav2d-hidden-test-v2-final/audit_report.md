# UAV2D final-evaluation audit

Status: **passed**  
Records: **6960**  
Timeouts are treated as failures. Ranking is feasibility first, then trusted cost.

| Rank | Arm | Feasible rate | Median feasible cost |
|---:|---|---:|---:|
| 1 | evo_no_rooms_strategy | 1.0000 | 121.1629 |
| 2 | frozen_afl_uav | 1.0000 | 126.1536 |
| 3 | evo_fixed_length | 0.9983 | 123.7847 |
| 4 | evolutionary_afl_uav_v1 | 0.9967 | 120.8939 |
| 5 | rrt | 0.9650 | 129.9417 |
| 6 | prm | 0.9050 | 129.6821 |
| 7 | theta_star | 0.8917 | 122.3878 |
| 8 | astar | 0.8167 | 121.6464 |
| 9 | rrt_star | 0.6033 | 118.8585 |
| 10 | dijkstra | 0.5333 | 117.6242 |
| 11 | pso | 0.5017 | 117.7633 |
| 12 | de | 0.5000 | 119.9257 |
| 13 | aco_acor | 0.5000 | 120.6634 |
| 14 | ga | 0.4950 | 117.5308 |

## Preregistered paired comparisons

- H1_evo_vs_frozen_afl: 113W/7T/0L, half-tie rate 0.9708.
- H2a_evo_vs_astar: 110W/7T/3L, half-tie rate 0.9458.
- H2b_evo_vs_theta_star: 104W/7T/9L, half-tie rate 0.8958.
- H3_rooms_evo_vs_no_rooms_strategy: 7W/0T/13L, half-tie rate 0.3500.
- H4_rooms_evo_vs_fixed_length: 8W/0T/12L, half-tie rate 0.4000.
