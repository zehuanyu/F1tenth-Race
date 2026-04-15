# Race Planner Parameter Study Summary

## Experimental Setup

The `race_planner` controller was evaluated on the Spielberg map in the F1TENTH simulator. Ten parameter groups were tested. For each group, the simulator was fully restarted, the controller was launched with the target parameter file, and the first completed lap time was recorded using the in-simulator lap timer. This design reduced run-to-run contamination from prior simulator state and provided a consistent one-lap comparison across all conditions.

## Results Table

| Rank | Group | Configuration Focus | Lap 1 Time (s) | Relative to Baseline |
|---|---|---|---:|---:|
| 1 | `g03_aggressive_speed` | Higher speed and acceleration limits | 61.21 | -5.17 |
| 2 | `g07_short_horizon` | Shorter planning horizon | 65.33 | -1.05 |
| 3 | `g08_smoother_steering` | Higher steering smoothing | 65.47 | -0.91 |
| 4 | `g01_baseline_fast` | Baseline fast setup | 66.38 | 0.00 |
| 5 | `g09_sharper_steering` | Lower steering smoothing | 66.49 | +0.11 |
| 6 | `g05_low_clearance` | Lower clearance margins | 66.57 | +0.19 |
| 7 | `g06_long_horizon` | Longer planning horizon | 67.87 | +1.49 |
| 8 | `g10_ml_disabled` | ML tuner disabled | 68.01 | +1.63 |
| 9 | `g04_high_clearance` | Higher clearance margins | 68.34 | +1.96 |
| 10 | `g02_conservative_speed` | Lower speed limits | 69.47 | +3.09 |

## Parameter Groups

| Group | Main Changes |
|---|---|
| `g01_baseline_fast` | Baseline fast Spielberg configuration |
| `g02_conservative_speed` | `max_speed=7.0`, `straight_speed=6.6`, `fast_curve_speed=5.6`, `medium_speed=4.3`, `corner_speed=2.5` |
| `g03_aggressive_speed` | `max_speed=8.5`, `straight_speed=8.0`, `fast_curve_speed=6.8`, `medium_speed=5.2`, `corner_speed=3.0`, `lat_accel_limit=10.2`, `accel_limit=6.4`, `decel_limit=10.3`, `brake_decel=8.9` |
| `g04_high_clearance` | `preferred_wall_clearance=0.9`, `safety_margin=0.22`, `clearance_weight=3.1`, `center_weight=1.7` |
| `g05_low_clearance` | `preferred_wall_clearance=0.64`, `safety_margin=0.14`, `clearance_weight=2.4`, `progress_weight=2.5` |
| `g06_long_horizon` | `min_horizon=2.8`, `max_horizon=6.6`, `lookahead_speed_gain=0.36` |
| `g07_short_horizon` | `min_horizon=2.0`, `max_horizon=4.9`, `lookahead_speed_gain=0.22` |
| `g08_smoother_steering` | `steering_smoothing=0.58`, `corner_steering_smoothing=0.38` |
| `g09_sharper_steering` | `steering_smoothing=0.38`, `corner_steering_smoothing=0.20` |
| `g10_ml_disabled` | `enable_ml_tuner=False` |

## Results Paragraph

Across the ten one-lap trials, the best performance was obtained with the aggressive speed configuration (`g03_aggressive_speed`), which achieved a lap time of `61.21 s`, improving on the baseline by `5.17 s`. Shorter horizon planning (`g07_short_horizon`) and increased steering smoothing (`g08_smoother_steering`) also improved lap time relative to the baseline, though by a smaller margin. In contrast, conservative speed limits (`g02_conservative_speed`) produced the slowest result at `69.47 s`, indicating that reduced speed authority had a stronger negative impact than the potential stability benefit. Disabling the ML tuner (`g10_ml_disabled`) also degraded performance relative to the baseline, suggesting that the adaptive tuning component contributed positively in this track setting. Overall, the study indicates that lap time on Spielberg was most sensitive to speed-policy parameters, while horizon and steering-smoothing parameters produced moderate but still measurable effects.

## Discussion Notes

- If your paper emphasizes raw performance, `g03_aggressive_speed` is the strongest candidate.
- If your paper emphasizes a balance between performance and likely control smoothness, `g07_short_horizon` and `g08_smoother_steering` are good middle-ground results.
- If you want stronger scientific confidence, the next step should be repeating each group for at least 3 to 5 laps and reporting mean and standard deviation rather than a single-lap value.
