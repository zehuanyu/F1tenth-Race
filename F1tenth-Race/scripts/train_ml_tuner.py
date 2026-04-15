#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


OUTPUT_NAMES = [
    'speed_scale',
    'risk',
    'clearance_weight_scale',
    'center_weight_scale',
    'raceline_weight_scale',
    'steering_smoothing',
    'steer_step_scale',
    'lateral_bias',
]


def clip(value, low, high):
    return float(np.clip(value, low, high))


def logit(value):
    value = clip(value, 1e-4, 1.0 - 1e-4)
    return float(np.log(value / (1.0 - value)))


def load_records(paths):
    records = []
    for path in paths:
        with Path(path).expanduser().open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'features' in record and 'feature_names' in record:
                    records.append(record)
    return records


def make_targets(record):
    values = record.get('feature_values', {})
    candidate = record.get('candidate', {})
    track = record.get('track', {})
    wall = record.get('wall', {})

    front = float(values.get('front_clearance', 0.0))
    front_min = float(values.get('front_min_clearance', 0.0))
    gap_width = float(values.get('gap_width', 0.0))
    signed_gap = float(values.get('signed_gap_angle', 0.0))
    abs_gap = float(values.get('abs_gap_angle', 0.0))
    curvature = float(values.get('track_curvature', 0.0))
    side_min = float(values.get('side_min_clearance', 0.0))
    side_balance = float(values.get('side_balance', 0.0))
    wall_error = float(values.get('wall_error', 0.0))
    abs_wall_error = float(values.get('abs_wall_error', 0.0))
    prev_speed = float(values.get('prev_speed', 0.0))
    abs_prev_steer = float(values.get('abs_prev_steer', 0.0))
    center_conf = float(values.get('center_confidence', 0.0))

    min_clearance = float(candidate.get('min_clearance', 1.0))
    safe = bool(candidate.get('safe', True))
    side_clearance = min(
        float(track.get('side_left_clearance', 1.5)),
        float(track.get('side_right_clearance', 1.5)),
        float(wall.get('side_clearance', 1.5)),
    )

    clearance_risk = clip((0.62 - min_clearance) / 0.62, 0.0, 1.0)
    side_risk = clip((0.72 - side_clearance) / 0.72, 0.0, 1.0)
    front_risk = 1.0 - clip(front_min / 0.45, 0.0, 1.0)

    risk = (
        0.06
        + 0.45 * abs_gap
        + 0.40 * curvature
        + 0.42 * clearance_risk
        + 0.30 * side_risk
        + 0.20 * front_risk
        + 0.16 * abs_wall_error
        + 0.12 * abs_prev_steer
    )
    if not safe:
        risk = max(risk, 0.92)
    risk = clip(risk, 0.02, 0.98)

    speed_scale = (
        1.10
        + 0.12 * front
        + 0.08 * gap_width
        - 0.34 * risk
        - 0.18 * abs_gap
        - 0.14 * curvature
        - 0.08 * abs_wall_error
    )
    speed_scale = clip(speed_scale, 0.76, 1.17)

    clearance_weight = clip(1.0 + 0.48 * risk + 0.22 * (1.0 - side_min) + 0.12 * abs_wall_error, 0.85, 1.55)
    center_weight = clip(1.0 + 0.34 * risk + 0.12 * center_conf + 0.10 * abs_prev_steer, 0.85, 1.45)
    raceline_weight = clip(1.08 + 0.18 * gap_width - 0.34 * risk - 0.14 * abs_wall_error, 0.62, 1.28)
    steering_smoothing = clip(0.34 + 0.24 * prev_speed + 0.20 * risk - 0.10 * curvature - 0.08 * abs_gap, 0.24, 0.72)
    steer_step = clip(1.0 + 0.18 * risk + 0.16 * abs_gap + 0.08 * curvature - 0.08 * prev_speed, 0.78, 1.22)
    lateral_bias = clip(0.08 * side_balance + 0.05 * signed_gap - 0.04 * wall_error, -0.16, 0.16)

    return {
        'speed_scale': speed_scale,
        'risk': risk,
        'clearance_weight_scale': clearance_weight,
        'center_weight_scale': center_weight,
        'raceline_weight_scale': raceline_weight,
        'steering_smoothing': steering_smoothing,
        'steer_step_scale': steer_step,
        'lateral_bias': lateral_bias,
    }


def fit_ridge(x, y, ridge):
    ones = np.ones((x.shape[0], 1), dtype=float)
    design = np.hstack((ones, x))
    penalty = np.eye(design.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(coef[0]), coef[1:].astype(float)


def train(records, ridge):
    if not records:
        raise ValueError('No usable records found.')

    feature_names = list(records[0]['feature_names'])
    rows = []
    labels = {name: [] for name in OUTPUT_NAMES}

    for record in records:
        if list(record.get('feature_names', [])) != feature_names:
            continue
        features = np.asarray(record.get('features', []), dtype=float)
        if features.size != len(feature_names) or not np.all(np.isfinite(features)):
            continue
        targets = make_targets(record)
        rows.append(features)
        risk = targets['risk']
        labels['risk'].append(logit(risk))
        labels['speed_scale'].append(targets['speed_scale'] - 1.0)
        labels['clearance_weight_scale'].append(targets['clearance_weight_scale'] - 1.0 - 0.35 * risk)
        labels['center_weight_scale'].append(targets['center_weight_scale'] - 1.0 - 0.22 * risk)
        labels['raceline_weight_scale'].append(targets['raceline_weight_scale'] - 1.0 + 0.18 * risk)
        labels['steering_smoothing'].append(targets['steering_smoothing'] - 0.07 * risk)
        labels['steer_step_scale'].append(targets['steer_step_scale'] - 1.0 - 0.10 * risk)
        labels['lateral_bias'].append(targets['lateral_bias'])

    x = np.vstack(rows)
    model = {
        'metadata': {
            'trainer': 'scripts/train_ml_tuner.py',
            'records': int(x.shape[0]),
            'ridge': float(ridge),
            'feature_names': feature_names,
            'label_policy': 'heuristic_safety_speed_targets_v1',
        }
    }
    for name in OUTPUT_NAMES:
        y = np.asarray(labels[name], dtype=float)
        bias, weights = fit_ridge(x, y, ridge)
        model[name] = {
            'bias': bias,
            'weights': weights.tolist(),
        }
    return model


def main():
    parser = argparse.ArgumentParser(description='Train race_planner ML tuner weights from JSONL logs.')
    parser.add_argument('logs', nargs='+', help='One or more race_training.jsonl files.')
    parser.add_argument('-o', '--output', default='src/race_fo/race_planner/models/ml_model.json')
    parser.add_argument('--ridge', type=float, default=0.08)
    args = parser.parse_args()

    records = load_records(args.logs)
    model = train(records, args.ridge)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(model, handle, indent=2)
        handle.write('\n')
    print(f'wrote {output} from {model["metadata"]["records"]} records')


if __name__ == '__main__':
    main()
