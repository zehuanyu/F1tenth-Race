# Race Planner Parameter Study Summary

## Experimental Setup

The `race_planner` controller was evaluated on the Spielberg map in the F1TENTH simulator. Ten parameter groups were tested. For each group, the simulator was fully restarted, the controller was launched with the target parameter file, and the first completed lap time was recorded using the in-simulator lap timer. This design reduced run-to-run contamination from prior simulator state and provided a consistent one-lap comparison across all conditions.

The baseline group was the repo's fast Spielberg configuration. Every other group changed a small parameter cluster around one main design idea:

- speed policy
- clearance / obstacle margin policy
- planning horizon
- steering smoothness
- ML adaptation

This makes the experiment easier to explain in a research paper because each group has a clear tuning purpose rather than many unrelated changes at once.

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

## Detailed Group-by-Group Analysis

### `g01_baseline_fast` - Baseline Fast

**Purpose**

This group served as the reference configuration for the entire study. It represents the default fast Spielberg setup and provides the comparison point for all delta values reported later.

**Key parameters**

- `max_speed = 7.8`
- `straight_speed = 7.35`
- `fast_curve_speed = 6.15`
- `medium_speed = 4.75`
- `corner_speed = 2.75`
- `lat_accel_limit = 8.8`
- `accel_limit = 5.2`
- `decel_limit = 9.2`
- `brake_decel = 7.8`
- `min_horizon = 2.35`
- `max_horizon = 5.80`
- `lookahead_speed_gain = 0.29`

**Expected behavior**

The baseline was expected to balance speed and stability without pushing either to an extreme. In other words, it should be competitive but not necessarily the absolute fastest.

**Measured result**

- Lap 1 time: `66.38 s`

**Interpretation**

This result provides the reference level for the study. Any group below `66.38 s` improved on the default fast setup, while any group above it reduced performance under the same map and simulator conditions.

### `g02_conservative_speed` - Conservative Speed

**Purpose**

This group tested whether lowering the speed profile would produce a more stable and ultimately more efficient lap, especially if the planner was over-driving some corners in the baseline setup.

**Changed parameters**

- `max_speed = 7.0`
- `straight_speed = 6.6`
- `fast_curve_speed = 5.6`
- `medium_speed = 4.3`
- `corner_speed = 2.5`

**Expected behavior**

The expected effect was smoother motion, reduced corner-entry risk, and fewer large corrections. However, this also reduces the vehicle's ability to exploit long straights and medium-speed corners.

**Measured result**

- Lap 1 time: `69.47 s`
- Relative to baseline: `+3.09 s`

**Interpretation**

This was the slowest group in the study. The result suggests that the baseline was not excessively aggressive on Spielberg and that reducing the speed limits removed useful performance more than it improved stability.

### `g03_aggressive_speed` - Aggressive Speed

**Purpose**

This group tested whether the planner could benefit from a more aggressive speed policy with higher dynamic limits.

**Changed parameters**

- `max_speed = 8.5`
- `straight_speed = 8.0`
- `fast_curve_speed = 6.8`
- `medium_speed = 5.2`
- `corner_speed = 3.0`
- `lat_accel_limit = 10.2`
- `accel_limit = 6.4`
- `decel_limit = 10.3`
- `brake_decel = 8.9`

**Expected behavior**

This setup should allow the planner to carry more speed through straights and medium corners, while also giving the speed controller more authority to accelerate and brake decisively.

**Measured result**

- Lap 1 time: `61.21 s`
- Relative to baseline: `-5.17 s`

**Interpretation**

This was the best configuration in the study and improved significantly on the baseline. The result indicates that, for this map and controller, the baseline speed policy was somewhat conservative and that the planner could safely exploit additional speed headroom.

### `g04_high_clearance` - High Clearance

**Purpose**

This group tested whether maintaining larger distances from walls and obstacles would improve robustness enough to offset the longer path length and lower cornering aggressiveness.

**Changed parameters**

- `preferred_wall_clearance = 0.90`
- `safety_margin = 0.22`
- `clearance_weight = 3.10`
- `center_weight = 1.70`

**Expected behavior**

The car should stay farther from obstacles, prefer safer lines, and reduce the risk of clipping walls. This would likely smooth the path but may also make the driven line less direct.

**Measured result**

- Lap 1 time: `68.34 s`
- Relative to baseline: `+1.96 s`

**Interpretation**

The larger safety margins hurt lap time. This suggests that on Spielberg, the baseline planner was already maintaining sufficient clearance and that forcing even larger margins mainly increased path conservatism rather than improving usable speed.

### `g05_low_clearance` - Low Clearance

**Purpose**

This group tested whether reducing obstacle margins would allow the planner to take tighter lines and gain time in corners.

**Changed parameters**

- `preferred_wall_clearance = 0.64`
- `safety_margin = 0.14`
- `clearance_weight = 2.40`
- `progress_weight = 2.50`

**Expected behavior**

The planner should be more willing to accept tighter paths near walls and emphasize forward progress slightly more strongly than the baseline.

**Measured result**

- Lap 1 time: `66.57 s`
- Relative to baseline: `+0.19 s`

**Interpretation**

This group was very close to the baseline but did not outperform it. That implies tighter clearance alone was not enough to produce a meaningful gain, and the baseline may already have been near a good compromise between safety and path efficiency.

### `g06_long_horizon` - Long Horizon

**Purpose**

This group tested whether planning farther ahead would help the controller make smoother, more strategic path choices before the car reaches a difficult section.

**Changed parameters**

- `min_horizon = 2.8`
- `max_horizon = 6.6`
- `lookahead_speed_gain = 0.36`

**Expected behavior**

The planner should react earlier to upcoming structure in the scan and potentially create smoother lines. However, longer horizon planning can also make local reactions less agile in tight sections.

**Measured result**

- Lap 1 time: `67.87 s`
- Relative to baseline: `+1.49 s`

**Interpretation**

The longer horizon slowed the lap. On Spielberg, the controller appears to benefit more from responsive local planning than from a farther lookahead that may over-smooth or delay useful local corrections.

### `g07_short_horizon` - Short Horizon

**Purpose**

This group tested the opposite hypothesis: that shorter-horizon planning would make the controller more reactive and better aligned with local track geometry.

**Changed parameters**

- `min_horizon = 2.0`
- `max_horizon = 4.9`
- `lookahead_speed_gain = 0.22`

**Expected behavior**

The planner should react more quickly to local geometry, potentially improving cornering on a technical track by reducing over-commitment to distant structure.

**Measured result**

- Lap 1 time: `65.33 s`
- Relative to baseline: `-1.05 s`

**Interpretation**

This was the second-best result overall. It suggests that the planner benefited from being more local and reactive, and that Spielberg rewards quick, near-field path adaptation more than long-range smoothing.

### `g08_smoother_steering` - Smoother Steering

**Purpose**

This group tested whether stronger steering smoothing would reduce oscillation and help the vehicle maintain a cleaner line.

**Changed parameters**

- `steering_smoothing = 0.58`
- `corner_steering_smoothing = 0.38`

**Expected behavior**

The car should make less abrupt steering changes, which may reduce path jitter, improve stability, and preserve speed through curves if the baseline steering is slightly noisy.

**Measured result**

- Lap 1 time: `65.47 s`
- Relative to baseline: `-0.91 s`

**Interpretation**

This group improved on the baseline and ranked third overall. The result suggests that smoother steering was beneficial on Spielberg and that the baseline controller may have been making slightly more abrupt corrections than necessary.

### `g09_sharper_steering` - Sharper Steering

**Purpose**

This group tested whether faster steering response and less smoothing would help the car respond more aggressively to local geometry.

**Changed parameters**

- `steering_smoothing = 0.38`
- `corner_steering_smoothing = 0.20`

**Expected behavior**

The controller should respond more quickly to path changes, which could help in tight turns, but may also introduce extra oscillation or line instability.

**Measured result**

- Lap 1 time: `66.49 s`
- Relative to baseline: `+0.11 s`

**Interpretation**

This group was nearly identical to baseline but slightly slower. That suggests the baseline already had enough steering responsiveness, and pushing it toward sharper corrections did not create a practical benefit on this map.

### `g10_ml_disabled` - ML Disabled

**Purpose**

This group evaluated whether the ML-assisted tuning layer was helping performance or whether the fixed deterministic planner alone was sufficient.

**Changed parameters**

- `enable_ml_tuner = False`

**Expected behavior**

If the ML tuner was useful, disabling it should reduce adaptation quality and increase lap time. If it was not useful, the result should remain close to the baseline.

**Measured result**

- Lap 1 time: `68.01 s`
- Relative to baseline: `+1.63 s`

**Interpretation**

Disabling the ML tuner made the controller slower. This indicates that the adaptive tuning layer provided a meaningful performance benefit on Spielberg, even though the core planner remained deterministic and safety-driven.

## Cross-Group Interpretation

Several higher-level conclusions can be drawn from the ten groups:

### 1. Speed policy had the largest effect

The largest performance swing came from the speed-related groups. The aggressive speed configuration improved the lap by more than five seconds, while the conservative speed group was the slowest in the study. This suggests that speed selection and dynamic-limit tuning are the most sensitive performance levers for this controller on Spielberg.

### 2. Local responsiveness mattered more than long-horizon smoothing

The short-horizon group outperformed the baseline, while the long-horizon group underperformed it. This pattern suggests that for Spielberg, a more reactive local planner is advantageous.

### 3. Moderate steering smoothing was helpful

Smoother steering outperformed both the baseline and the sharper-steering setup. This implies that removing small oscillations and overly abrupt steering corrections helped the vehicle maintain a cleaner, faster line.

### 4. Extra safety margin increased lap time

The high-clearance group was substantially slower than baseline, while the low-clearance group was only marginally different from baseline. This suggests that overly conservative clearance policies add path inefficiency without providing a measurable speed benefit in this evaluation.

### 5. The ML tuner improved performance

The ML-disabled group was slower than baseline, which supports the idea that adaptive tuning improved the local planner's balance between speed, clearance, and steering behavior.

## Results Paragraph

Across the ten one-lap trials, the best performance was obtained with the aggressive speed configuration (`g03_aggressive_speed`), which achieved a lap time of `61.21 s`, improving on the baseline by `5.17 s`. Shorter horizon planning (`g07_short_horizon`) and increased steering smoothing (`g08_smoother_steering`) also improved lap time relative to the baseline, though by a smaller margin. In contrast, conservative speed limits (`g02_conservative_speed`) produced the slowest result at `69.47 s`, indicating that reduced speed authority had a stronger negative impact than the potential stability benefit. Disabling the ML tuner (`g10_ml_disabled`) also degraded performance relative to the baseline, suggesting that the adaptive tuning component contributed positively in this track setting. Overall, the study indicates that lap time on Spielberg was most sensitive to speed-policy parameters, while horizon and steering-smoothing parameters produced moderate but still measurable effects.

## Discussion Notes

- If your paper emphasizes raw performance, `g03_aggressive_speed` is the strongest candidate.
- If your paper emphasizes a balance between performance and likely control smoothness, `g07_short_horizon` and `g08_smoother_steering` are good middle-ground results.
- If your paper emphasizes robustness and safety, `g04_high_clearance` is easier to justify conceptually, although it was slower.
- If you want stronger scientific confidence, the next step should be repeating each group for at least 3 to 5 laps and reporting mean and standard deviation rather than a single-lap value.
