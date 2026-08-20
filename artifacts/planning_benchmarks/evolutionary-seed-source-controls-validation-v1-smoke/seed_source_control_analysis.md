# Evolutionary AFL-UAV seed-source control analysis

Conclusion: `afl_seed_superiority_not_established`

| Arm | Feasible | Median cost | Median seed cost | Median time (s) |
|---|---:|---:|---:|---:|
| evolutionary_astar_seed | 5/6 (83.333%) | 117.055565 | 118.492671 | 0.915005 |
| evolutionary_theta_seed | 5/6 (83.333%) | 116.675755 | 117.780389 | 0.918978 |
| evolutionary_afl_uav_v1 | 6/6 (100.000%) | 117.924183 | 119.322632 | 0.905759 |
| evolutionary_handcrafted_destroy_repair_seed | 6/6 (100.000%) | 118.302568 | 118.814275 | 0.895332 |

| Comparator | AFL wins | Ties | AFL losses | Holm p | Median cost difference (control - AFL) | 95% cluster bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| evolutionary_astar_seed | 4 | 1 | 1 | 0.75 | 0.000111 | [-0.406360, 20.434561] |
| evolutionary_theta_seed | 3 | 1 | 2 | 1 | 0.000000 | [-0.170400, 15.070734] |
| evolutionary_handcrafted_destroy_repair_seed | 5 | 1 | 0 | 0.1875 | 0.378386 | [0.000055, 11.360429] |
