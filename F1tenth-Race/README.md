# F1tenth-Race

`race_planner` is a LiDAR-only local racing planner for the F1TENTH simulator. This repository contains the full ROS 2 package, tuning configs, ML helper scripts, pretrained Spielberg models, and the testing artifacts used during simulator evaluation.

## Overview

The planner combines several ideas into one local racing stack:

- Lab 3 style wall following: estimate wall angle from LiDAR beams and project side distance forward to avoid late corrections.
- Lab 4 style follow-the-gap: mask unsafe obstacles with a safety bubble and keep the best free-space target as a fallback direction.
- Lab 6 style local planning: sample short trajectories, reject unsafe candidates, and score the survivors.
- Predictive speed control: choose speed from curvature, clearance, braking distance, and steering smoothness.
- Racing-line setup: bias toward the outside before turn-in, then allow the local gap target to pull back toward the apex.
- ML-assisted tuning: a small NumPy model adjusts speed scale, scoring weights, steering smoothing, and lateral bias while the deterministic planner remains responsible for safety.

## Repository Layout

```text
.
├── config/
│   ├── spielberg_fast.yaml
│   ├── spielberg_fast_plus_1.yaml
│   ├── spielberg_max_stable.yaml
│   └── param_study/
├── models/
├── race_planner/
├── scripts/
└── results/
```

- `config/`: tuned parameter sets for Spielberg plus the 10-group parameter study.
- `models/`: pretrained ML tuner weights.
- `race_planner/`: main ROS 2 node implementation.
- `scripts/`: training and log-conversion utilities.
- `results/`: exported CSV/JSON/DOCX summaries from simulator testing.

## Running After Build

Build in your ROS 2 workspace, then run:

```bash
ros2 run race_planner race_planner_node.py
```

Run with a specific config:

```bash
ros2 run race_planner race_planner_node.py --ros-args \
  --params-file /path/to/spielberg_fast.yaml
```

Useful ML parameters:

```bash
ros2 run race_planner race_planner_node.py --ros-args \
  -p enable_ml_tuner:=true \
  -p ml_model_path:=''
```

If `ml_model_path` is empty, the built-in conservative model is used.

Run with a trained external model:

```bash
ros2 run race_planner race_planner_node.py --ros-args \
  -p enable_ml_tuner:=true \
  -p ml_model_path:=/path/to/ml_model.json
```

## Collecting Training Data

```bash
ros2 run race_planner race_planner_node.py --ros-args \
  -p enable_ml_tuner:=true \
  -p enable_training_log:=true \
  -p training_log_path:=/sim_ws/logs/race_training.jsonl \
  -p training_log_stride:=4
```

Train a model:

```bash
python3 scripts/train_ml_tuner.py \
  logs/race_training.jsonl \
  -o models/ml_model.json
```

Convert logs to CSV:

```bash
python3 scripts/jsonl_to_csv.py \
  logs/race_training.jsonl \
  -o logs/race_training.csv
```

For a smaller summary CSV:

```bash
python3 scripts/jsonl_to_csv.py \
  logs/race_training.jsonl \
  -o logs/race_training_summary.csv \
  --summary
```

## Simulator Testing Workflow

This planner was tested in the F1TENTH simulator on the Spielberg map inside Docker/noVNC. The simulator setup used:

- a clean Docker restart before every test run
- `race_planner` launched as the only active controller
- an in-simulator lap timer to record completed lap times
- RViz configured to follow the ego vehicle automatically

For the testing environment used during development, each run followed this pattern:

1. Fully stop and remove the existing simulator containers and Docker network.
2. Rebuild `f1tenth_gym_ros` and `race_planner`.
3. Relaunch the simulator.
4. Launch `race_planner` with a specific parameter file.
5. Record the first completed lap time from the simulator lap timer.

## Parameter Study

A 10-group one-lap parameter study was run to evaluate how tuning affected lap time on Spielberg. Each group used a full simulator restart before measurement.

### Ranked Results

| Rank | Group | Configuration Focus | Lap 1 Time (s) | Delta vs Baseline (s) |
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

### Main Takeaway

The strongest one-lap result came from the aggressive speed configuration, which improved lap time by `5.17 s` relative to the baseline. Shorter planning horizon and smoother steering also improved performance, though by smaller margins. Conservative speed settings and larger clearance margins were slower, and disabling the ML tuner also hurt performance on Spielberg.

## Testing Artifacts

The repository includes the exported study outputs:

- `results/parameter_study_results.csv`
- `results/parameter_study_results.json`
- `results/parameter_study_report.docx`
- `results/parameter_study_paper_summary.md`

The parameter files used in the study are stored in:

- `config/param_study/`

These files make it easy to rerun the same 10-group comparison in another simulator workspace.

## Suggested Next Step

For a research-paper quality evaluation, the next improvement would be to repeat each parameter group for 3 to 5 laps and report mean, standard deviation, and failure count instead of a single-lap measurement.
