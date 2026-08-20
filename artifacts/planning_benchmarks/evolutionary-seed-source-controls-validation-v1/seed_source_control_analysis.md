# Evolutionary AFL-UAV seed-source control analysis

Conclusion: `afl_seed_clearly_superior_under_preregistered_validation_rule`

Protocol feasible means `status == success` and trusted geometry feasible; every timeout is a failure.

| Arm | Protocol feasible | Timeouts | Median cost | Median seed cost | Median time (s) |
|---|---:|---:|---:|---:|---:|
| evolutionary_astar_seed | 246/300 (82.000%) | 54 | 115.686273 | 118.403255 | 0.911443 |
| evolutionary_theta_seed | 263/300 (87.667%) | 37 | 116.488848 | 118.416285 | 0.914919 |
| evolutionary_afl_uav_v1 | 300/300 (100.000%) | 0 | 117.500906 | 120.810004 | 0.914035 |
| evolutionary_handcrafted_destroy_repair_seed | 246/300 (82.000%) | 54 | 115.699097 | 118.323651 | 0.911704 |

| Comparator | AFL wins | Ties | AFL losses | Reliability wins | Cost W/T/L on both feasible | Holm p | Median cost difference (control - AFL) | 95% cluster bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| evolutionary_astar_seed | 176 | 33 | 91 | 54 | 122/33/91 | 6.54859e-07 | 0.000000 | [0.000000, 0.033359] |
| evolutionary_theta_seed | 161 | 33 | 106 | 37 | 124/33/106 | 0.000915997 | 0.000000 | [0.000000, 0.008924] |
| evolutionary_handcrafted_destroy_repair_seed | 166 | 35 | 99 | 54 | 112/35/99 | 9.22325e-05 | 0.000000 | [-0.000256, 0.021960] |
