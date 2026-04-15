#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


BASE_COLUMNS = [
    'stamp',
    'mode',
    'drive.speed',
    'drive.steering_angle',
    'tuning.risk',
    'tuning.speed_scale',
    'tuning.clearance_weight_scale',
    'tuning.center_weight_scale',
    'tuning.raceline_weight_scale',
    'tuning.steering_smoothing',
    'tuning.steer_step_scale',
    'tuning.lateral_bias',
    'candidate.curvature',
    'candidate.min_clearance',
    'candidate.mean_clearance',
    'candidate.score',
    'candidate.safe',
    'candidate.center_error',
    'track.confidence',
    'track.curvature',
    'track.width',
    'track.front_clearance',
    'track.front_left_clearance',
    'track.front_right_clearance',
    'track.side_left_clearance',
    'track.side_right_clearance',
    'gap.angle',
    'gap.distance',
    'gap.width',
    'gap.exists',
    'wall.error',
    'wall.confidence',
    'wall.side_clearance',
    'wall.exists',
    'raceline.active',
    'raceline.s_bend_active',
    'raceline.turn_sign',
    'raceline.weight',
    'odom.x',
    'odom.y',
    'odom.yaw',
    'odom.speed',
    'params.max_speed',
    'params.straight_speed',
    'params.fast_curve_speed',
    'params.medium_speed',
    'params.corner_speed',
]


SUMMARY_COLUMNS = [
    'stamp',
    'drive.speed',
    'drive.steering_angle',
    'tuning.risk',
    'tuning.speed_scale',
    'track.front_clearance',
    'track.front_left_clearance',
    'track.front_right_clearance',
    'track.side_left_clearance',
    'track.side_right_clearance',
    'gap.angle',
    'gap.distance',
    'gap.width',
    'candidate.min_clearance',
    'candidate.safe',
    'track.curvature',
    'odom.x',
    'odom.y',
    'odom.yaw',
    'odom.speed',
]


def nested_get(record, dotted_name):
    value = record
    for part in dotted_name.split('.'):
        if not isinstance(value, dict) or part not in value:
            return ''
        value = value[part]
    return value


def flatten_value(value):
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ''
    return value


def load_records(path):
    with Path(path).expanduser().open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def convert(input_path, output_path, summary=False):
    rows = list(load_records(input_path))
    if not rows:
        raise ValueError(f'No usable records in {input_path}')

    feature_names = list(rows[0].get('feature_names', []))
    feature_columns = [] if summary else [f'feature.{name}' for name in feature_names]
    columns = (SUMMARY_COLUMNS if summary else BASE_COLUMNS) + feature_columns

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in rows:
            row = {}
            for column in (SUMMARY_COLUMNS if summary else BASE_COLUMNS):
                row[column] = flatten_value(nested_get(record, column))

            if not summary:
                feature_values = record.get('feature_values', {})
                features = record.get('features', [])
                for idx, name in enumerate(feature_names):
                    if name in feature_values:
                        row[f'feature.{name}'] = flatten_value(feature_values[name])
                    elif idx < len(features):
                        row[f'feature.{name}'] = flatten_value(features[idx])
                    else:
                        row[f'feature.{name}'] = ''

            writer.writerow(row)

    return len(rows), output


def main():
    parser = argparse.ArgumentParser(description='Convert race_planner JSONL training logs to CSV.')
    parser.add_argument('input', help='Input race_training.jsonl file.')
    parser.add_argument('-o', '--output', help='Output CSV path. Defaults to input path with .csv suffix.')
    parser.add_argument('--summary', action='store_true', help='Only export the most useful columns for quick viewing.')
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else input_path.with_suffix('.csv')
    count, output = convert(input_path, output_path, summary=args.summary)
    print(f'wrote {output} from {count} records')


if __name__ == '__main__':
    main()
