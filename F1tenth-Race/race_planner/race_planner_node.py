#!/usr/bin/env python3

import json
from dataclasses import dataclass
from math import atan2, cos, isfinite, sin, sqrt, tan
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


@dataclass
class TrackModel:
    xs: np.ndarray
    ys: np.ndarray
    confidence: float
    curvature: float
    width: float
    front_clearance: float
    front_left_clearance: float
    front_right_clearance: float
    side_left_clearance: float
    side_right_clearance: float


@dataclass
class GapTarget:
    angle: float
    distance: float
    width: float
    exists: bool


@dataclass
class WallAssist:
    correction: float
    target_y: float
    error: float
    confidence: float
    left_future: float
    right_future: float
    side_clearance: float
    exists: bool


@dataclass
class RaceLineBias:
    correction: float
    target_y: float
    weight: float
    turn_sign: float
    active: bool
    s_bend_active: bool


@dataclass
class MLTuning:
    speed_scale: float
    clearance_weight_scale: float
    center_weight_scale: float
    raceline_weight_scale: float
    steering_smoothing: float
    steer_step_scale: float
    lateral_bias: float
    risk: float
    enabled: bool


@dataclass
class Candidate:
    curvature: float
    xs: np.ndarray
    ys: np.ndarray
    headings: np.ndarray
    min_clearance: float
    mean_clearance: float
    score: float
    safe: bool
    center_error: float
    progress: float


class RacePlanner(Node):
    """Predictive local trajectory planner for LiDAR-only F1TENTH driving."""

    def __init__(self):
        super().__init__('race_planner_node')

        self.scan_topic = str(self.declare_parameter('scan_topic', '/scan').value)
        self.drive_topic = str(self.declare_parameter('drive_topic', '/drive').value)
        self.odom_topic = str(self.declare_parameter('odom_topic', '/ego_racecar/odom').value)

        self.forward_fov_deg = float(self.declare_parameter('forward_fov_deg', 220.0).value)
        self.max_lidar_range = float(self.declare_parameter('max_lidar_range', 10.0).value)
        self.smoothing_window = int(self.declare_parameter('smoothing_window', 5).value)

        self.wheelbase = float(self.declare_parameter('wheelbase', 0.33).value)
        self.max_steer = float(self.declare_parameter('max_steer', 0.42).value)
        self.max_steer_step = float(self.declare_parameter('max_steer_step', 0.075).value)
        self.corner_steer_step = float(self.declare_parameter('corner_steer_step', 0.11).value)
        self.steering_smoothing = float(self.declare_parameter('steering_smoothing', 0.48).value)
        self.corner_steering_smoothing = float(self.declare_parameter('corner_steering_smoothing', 0.28).value)
        self.pure_pursuit_gain = float(self.declare_parameter('pure_pursuit_gain', 0.88).value)
        self.feedforward_gain = float(self.declare_parameter('feedforward_gain', 0.28).value)

        self.car_width = float(self.declare_parameter('car_width', 0.32).value)
        self.car_length = float(self.declare_parameter('car_length', 0.58).value)
        self.safety_margin = float(self.declare_parameter('safety_margin', 0.18).value)
        self.preferred_wall_clearance = float(self.declare_parameter('preferred_wall_clearance', 0.78).value)
        self.hard_wall_clearance = float(self.declare_parameter('hard_wall_clearance', 0.42).value)
        self.wall_theta_deg = float(self.declare_parameter('wall_theta_deg', 50.0).value)
        self.wall_lookahead_base = float(self.declare_parameter('wall_lookahead_base', 0.72).value)
        self.wall_lookahead_speed_gain = float(self.declare_parameter('wall_lookahead_speed_gain', 0.10).value)
        self.wall_assist_kp = float(self.declare_parameter('wall_assist_kp', 0.085).value)
        self.wall_assist_kd = float(self.declare_parameter('wall_assist_kd', 0.018).value)
        self.max_wall_correction = float(self.declare_parameter('max_wall_correction', 0.115).value)
        self.wall_target_clip = float(self.declare_parameter('wall_target_clip', 0.42).value)
        self.wall_assist_weight = float(self.declare_parameter('wall_assist_weight', 1.05).value)
        self.raceline_gain = float(self.declare_parameter('raceline_gain', 0.13).value)
        self.raceline_score_weight = float(self.declare_parameter('raceline_score_weight', 1.25).value)
        self.raceline_target_y = float(self.declare_parameter('raceline_target_y', 0.44).value)
        self.raceline_setup_distance = float(self.declare_parameter('raceline_setup_distance', 1.70).value)
        self.raceline_min_turn_angle = float(self.declare_parameter('raceline_min_turn_angle', 0.15).value)
        self.raceline_front_clearance = float(self.declare_parameter('raceline_front_clearance', 1.35).value)
        self.raceline_s_bend_hold_time = float(self.declare_parameter('raceline_s_bend_hold_time', 0.85).value)
        self.raceline_turn_memory_time = float(self.declare_parameter('raceline_turn_memory_time', 0.80).value)
        self.raceline_countersteer_angle = float(self.declare_parameter('raceline_countersteer_angle', 0.13).value)
        self.raceline_apex_pull_gain = float(self.declare_parameter('raceline_apex_pull_gain', 0.09).value)
        self.inner_corner_penalty_weight = float(self.declare_parameter('inner_corner_penalty_weight', 1.60).value)

        self.min_horizon = float(self.declare_parameter('min_horizon', 2.10).value)
        self.max_horizon = float(self.declare_parameter('max_horizon', 5.40).value)
        self.horizon_speed_gain = float(self.declare_parameter('horizon_speed_gain', 0.44).value)
        self.path_ds = float(self.declare_parameter('path_ds', 0.12).value)
        self.trajectory_count = int(self.declare_parameter('trajectory_count', 39).value)
        self.max_candidate_curvature = float(self.declare_parameter('max_candidate_curvature', 1.18).value)
        self.min_lookahead = float(self.declare_parameter('min_lookahead', 1.05).value)
        self.max_lookahead = float(self.declare_parameter('max_lookahead', 2.85).value)
        self.lookahead_speed_gain = float(self.declare_parameter('lookahead_speed_gain', 0.25).value)

        self.center_samples = int(self.declare_parameter('center_samples', 10).value)
        self.center_slice_width = float(self.declare_parameter('center_slice_width', 0.32).value)
        self.min_track_width = float(self.declare_parameter('min_track_width', 1.05).value)
        self.max_track_width = float(self.declare_parameter('max_track_width', 4.60).value)
        self.min_center_confidence = float(self.declare_parameter('min_center_confidence', 0.34).value)

        self.max_speed = float(self.declare_parameter('max_speed', 6.80).value)
        self.straight_speed = float(self.declare_parameter('straight_speed', 6.55).value)
        self.fast_curve_speed = float(self.declare_parameter('fast_curve_speed', 5.45).value)
        self.medium_speed = float(self.declare_parameter('medium_speed', 4.10).value)
        self.corner_speed = float(self.declare_parameter('corner_speed', 2.55).value)
        self.emergency_speed = float(self.declare_parameter('emergency_speed', 0.55).value)
        self.lat_accel_limit = float(self.declare_parameter('lat_accel_limit', 7.20).value)
        self.accel_limit = float(self.declare_parameter('accel_limit', 3.80).value)
        self.decel_limit = float(self.declare_parameter('decel_limit', 8.40).value)
        self.brake_decel = float(self.declare_parameter('brake_decel', 6.60).value)
        self.brake_buffer = float(self.declare_parameter('brake_buffer', 0.50).value)

        self.enable_ml_tuner = bool(self.declare_parameter('enable_ml_tuner', True).value)
        self.ml_model_path = str(self.declare_parameter('ml_model_path', '').value)
        self.ml_speed_scale_min = float(self.declare_parameter('ml_speed_scale_min', 0.78).value)
        self.ml_speed_scale_max = float(self.declare_parameter('ml_speed_scale_max', 1.16).value)
        self.ml_lateral_bias_limit = float(self.declare_parameter('ml_lateral_bias_limit', 0.16).value)
        self.ml_smoothing_alpha = float(self.declare_parameter('ml_smoothing_alpha', 0.30).value)
        self.ml_risk_slowdown = float(self.declare_parameter('ml_risk_slowdown', 0.22).value)
        self.enable_training_log = bool(self.declare_parameter('enable_training_log', False).value)
        self.training_log_path = str(
            self.declare_parameter('training_log_path', '/sim_ws/logs/race_training.jsonl').value
        )
        self.training_log_stride = int(self.declare_parameter('training_log_stride', 4).value)
        self.enable_map_guard = bool(self.declare_parameter('enable_map_guard', False).value)
        self.map_guard_name = str(self.declare_parameter('map_guard_name', '').value).lower()
        self.map_guard_resolution = float(self.declare_parameter('map_guard_resolution', 0.08).value)
        self.enable_global_waypoints = bool(self.declare_parameter('enable_global_waypoints', False).value)
        self.waypoint_lookahead = float(self.declare_parameter('waypoint_lookahead', 1.55).value)
        self.waypoint_speed = float(self.declare_parameter('waypoint_speed', 2.35).value)
        self.waypoint_corner_speed = float(self.declare_parameter('waypoint_corner_speed', 1.25).value)

        self.progress_weight = float(self.declare_parameter('progress_weight', 2.30).value)
        self.clearance_weight = float(self.declare_parameter('clearance_weight', 2.80).value)
        self.center_weight = float(self.declare_parameter('center_weight', 1.55).value)
        self.gap_weight = float(self.declare_parameter('gap_weight', 0.46).value)
        self.curvature_weight = float(self.declare_parameter('curvature_weight', 0.56).value)
        self.smooth_weight = float(self.declare_parameter('smooth_weight', 0.82).value)

        self.bubble_radius = float(self.declare_parameter('bubble_radius', 0.43).value)
        self.min_gap_range = float(self.declare_parameter('min_gap_range', 0.78).value)
        self.gap_window = int(self.declare_parameter('gap_window', 25).value)

        self.prev_steer = 0.0
        self.prev_speed = 0.0
        self.prev_curvature = 0.0
        self.prev_wall_error = 0.0
        self.prev_raceline_correction = 0.0
        self.turn_memory_sign = 0.0
        self.turn_memory_until = 0.0
        self.s_bend_until = 0.0
        self.prev_time = None
        self.best_miss_count = 0
        self.odom_pose = None
        self.odom_speed = 0.0
        self.last_ml_features = np.zeros(0, dtype=float)
        self.last_ml_feature_values = {}
        self.training_log_count = 0
        self.training_log_file = None
        self.levine_guard_segments = self._build_levine_guard_segments()
        self.levine_waypoints = self._build_levine_waypoints()
        self.ml_feature_names = [
            'front_clearance',
            'front_min_clearance',
            'gap_distance',
            'gap_width',
            'signed_gap_angle',
            'abs_gap_angle',
            'track_curvature',
            'side_min_clearance',
            'side_balance',
            'wall_error',
            'abs_wall_error',
            'prev_speed',
            'abs_prev_steer',
            'center_confidence',
        ]
        self.ml_weights = self._default_ml_weights()
        self.prev_ml_tuning = self._neutral_ml_tuning(enabled=self.enable_ml_tuner)

        self._sanitize_parameters()
        self._load_ml_model()
        self._open_training_log()

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        self.get_logger().info(
            'RacePlanner started: '
            f'vmax={self.max_speed:.1f}m/s, paths={self.trajectory_count}, '
            f'horizon={self.min_horizon:.1f}-{self.max_horizon:.1f}m, '
            f'ml_tuner={self.enable_ml_tuner}'
        )

    def _sanitize_parameters(self):
        self.forward_fov_deg = float(np.clip(self.forward_fov_deg, 120.0, 260.0))
        self.max_lidar_range = max(1.0, self.max_lidar_range)
        self.smoothing_window = self._odd_window(self.smoothing_window)
        self.gap_window = self._odd_window(self.gap_window)
        self.wheelbase = max(0.1, self.wheelbase)
        self.max_steer = max(0.08, self.max_steer)
        self.max_steer_step = max(0.02, self.max_steer_step)
        self.corner_steer_step = max(self.max_steer_step, self.corner_steer_step)
        self.steering_smoothing = float(np.clip(self.steering_smoothing, 0.0, 0.95))
        self.corner_steering_smoothing = float(np.clip(self.corner_steering_smoothing, 0.0, 0.95))
        self.car_width = max(0.15, self.car_width)
        self.car_length = max(0.25, self.car_length)
        self.safety_margin = max(0.04, self.safety_margin)
        self.preferred_wall_clearance = max(0.45, self.preferred_wall_clearance)
        self.hard_wall_clearance = max(0.20, self.hard_wall_clearance)
        self.wall_theta_deg = float(np.clip(self.wall_theta_deg, 25.0, 70.0))
        self.wall_lookahead_base = max(0.2, self.wall_lookahead_base)
        self.wall_lookahead_speed_gain = max(0.0, self.wall_lookahead_speed_gain)
        self.wall_assist_kp = max(0.0, self.wall_assist_kp)
        self.wall_assist_kd = max(0.0, self.wall_assist_kd)
        self.max_wall_correction = max(0.0, self.max_wall_correction)
        self.wall_target_clip = max(0.05, self.wall_target_clip)
        self.wall_assist_weight = max(0.0, self.wall_assist_weight)
        self.raceline_gain = max(0.0, self.raceline_gain)
        self.raceline_score_weight = max(0.0, self.raceline_score_weight)
        self.raceline_target_y = max(0.05, self.raceline_target_y)
        self.raceline_setup_distance = max(0.50, self.raceline_setup_distance)
        self.raceline_min_turn_angle = max(0.03, self.raceline_min_turn_angle)
        self.raceline_front_clearance = max(0.6, self.raceline_front_clearance)
        self.raceline_s_bend_hold_time = max(0.0, self.raceline_s_bend_hold_time)
        self.raceline_turn_memory_time = max(0.0, self.raceline_turn_memory_time)
        self.raceline_countersteer_angle = max(0.03, self.raceline_countersteer_angle)
        self.raceline_apex_pull_gain = max(0.0, self.raceline_apex_pull_gain)
        self.inner_corner_penalty_weight = max(0.0, self.inner_corner_penalty_weight)
        self.min_horizon = max(1.0, self.min_horizon)
        self.max_horizon = max(self.min_horizon, self.max_horizon)
        self.path_ds = float(np.clip(self.path_ds, 0.05, 0.25))
        self.trajectory_count = max(11, int(self.trajectory_count))
        if self.trajectory_count % 2 == 0:
            self.trajectory_count += 1
        steer_curvature = abs(tan(self.max_steer)) / self.wheelbase
        self.max_candidate_curvature = float(np.clip(self.max_candidate_curvature, 0.20, steer_curvature))
        self.min_lookahead = max(0.45, self.min_lookahead)
        self.max_lookahead = max(self.min_lookahead, self.max_lookahead)
        self.center_samples = max(4, int(self.center_samples))
        self.center_slice_width = max(0.08, self.center_slice_width)
        self.min_track_width = max(0.60, self.min_track_width)
        self.max_track_width = max(self.min_track_width, self.max_track_width)
        self.min_center_confidence = float(np.clip(self.min_center_confidence, 0.0, 1.0))
        self.max_speed = max(0.5, self.max_speed)
        self.straight_speed = float(np.clip(self.straight_speed, self.emergency_speed, self.max_speed))
        self.fast_curve_speed = float(np.clip(self.fast_curve_speed, self.emergency_speed, self.max_speed))
        self.medium_speed = float(np.clip(self.medium_speed, self.emergency_speed, self.max_speed))
        self.corner_speed = float(np.clip(self.corner_speed, self.emergency_speed, self.max_speed))
        self.lat_accel_limit = max(0.5, self.lat_accel_limit)
        self.accel_limit = max(0.1, self.accel_limit)
        self.decel_limit = max(0.1, self.decel_limit)
        self.brake_decel = max(0.5, self.brake_decel)
        self.ml_speed_scale_min = float(np.clip(self.ml_speed_scale_min, 0.45, 1.05))
        self.ml_speed_scale_max = float(np.clip(self.ml_speed_scale_max, self.ml_speed_scale_min, 1.35))
        self.ml_lateral_bias_limit = float(np.clip(self.ml_lateral_bias_limit, 0.0, 0.35))
        self.ml_smoothing_alpha = float(np.clip(self.ml_smoothing_alpha, 0.02, 1.0))
        self.ml_risk_slowdown = float(np.clip(self.ml_risk_slowdown, 0.0, 0.50))
        self.training_log_stride = max(1, int(self.training_log_stride))
        self.map_guard_resolution = float(np.clip(self.map_guard_resolution, 0.03, 0.25))
        self.waypoint_lookahead = float(np.clip(self.waypoint_lookahead, 0.60, 3.50))
        self.waypoint_speed = float(np.clip(self.waypoint_speed, self.emergency_speed, self.max_speed))
        self.waypoint_corner_speed = float(np.clip(self.waypoint_corner_speed, self.emergency_speed, self.waypoint_speed))
        self.bubble_radius = max(0.05, self.bubble_radius)
        self.min_gap_range = max(0.05, self.min_gap_range)

    @staticmethod
    def _odd_window(window):
        window = max(1, int(window))
        return window if window % 2 == 1 else window + 1

    @staticmethod
    def _smooth(values, window):
        if window <= 1 or values.size < window:
            return values.copy()
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(values, kernel, mode='same')

    @staticmethod
    def _sigmoid(value):
        value = float(np.clip(value, -20.0, 20.0))
        return 1.0 / (1.0 + np.exp(-value))

    @staticmethod
    def _neutral_ml_tuning(enabled=False):
        return MLTuning(
            speed_scale=1.0,
            clearance_weight_scale=1.0,
            center_weight_scale=1.0,
            raceline_weight_scale=1.0,
            steering_smoothing=0.48,
            steer_step_scale=1.0,
            lateral_bias=0.0,
            risk=0.0,
            enabled=enabled,
        )

    def _default_ml_weights(self):
        count = len(self.ml_feature_names)

        def vector(values):
            result = np.zeros(count, dtype=float)
            for name, weight in values.items():
                if name in self.ml_feature_names:
                    result[self.ml_feature_names.index(name)] = float(weight)
            return result

        return {
            'speed_scale': {
                'bias': 0.02,
                'weights': vector({
                    'front_clearance': 0.10,
                    'front_min_clearance': 0.08,
                    'gap_distance': 0.08,
                    'gap_width': 0.05,
                    'abs_gap_angle': -0.22,
                    'track_curvature': -0.24,
                    'side_min_clearance': 0.10,
                    'abs_wall_error': -0.05,
                    'abs_prev_steer': -0.08,
                }),
            },
            'risk': {
                'bias': -1.80,
                'weights': vector({
                    'front_min_clearance': -1.30,
                    'side_min_clearance': -1.10,
                    'abs_gap_angle': 1.80,
                    'track_curvature': 2.20,
                    'abs_wall_error': 1.10,
                    'abs_prev_steer': 0.80,
                }),
            },
            'clearance_weight_scale': {
                'bias': 0.00,
                'weights': vector({
                    'abs_gap_angle': 0.18,
                    'track_curvature': 0.24,
                    'abs_wall_error': 0.20,
                    'side_min_clearance': -0.10,
                }),
            },
            'center_weight_scale': {
                'bias': 0.00,
                'weights': vector({
                    'track_curvature': 0.18,
                    'abs_wall_error': 0.18,
                    'abs_prev_steer': 0.08,
                    'center_confidence': 0.16,
                }),
            },
            'raceline_weight_scale': {
                'bias': 0.00,
                'weights': vector({
                    'front_clearance': 0.08,
                    'abs_gap_angle': 0.16,
                    'gap_width': 0.06,
                    'track_curvature': -0.08,
                    'abs_wall_error': -0.18,
                }),
            },
            'steering_smoothing': {
                'bias': self.steering_smoothing,
                'weights': vector({
                    'prev_speed': 0.14,
                    'abs_gap_angle': -0.11,
                    'track_curvature': -0.10,
                    'abs_prev_steer': 0.08,
                    'center_confidence': 0.04,
                }),
            },
            'steer_step_scale': {
                'bias': 0.00,
                'weights': vector({
                    'abs_gap_angle': 0.20,
                    'track_curvature': 0.14,
                    'prev_speed': -0.08,
                    'abs_wall_error': 0.08,
                }),
            },
            'lateral_bias': {
                'bias': 0.00,
                'weights': vector({
                    'signed_gap_angle': 0.06,
                    'side_balance': 0.10,
                    'wall_error': -0.04,
                }),
            },
        }

    def _load_ml_model(self):
        if not self.ml_model_path:
            return

        path = Path(self.ml_model_path).expanduser()
        if not path.exists():
            self.get_logger().warn(f'ML model file not found, using built-in tuner: {path}')
            return

        try:
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            for key, spec in data.items():
                if key not in self.ml_weights or not isinstance(spec, dict):
                    continue
                if 'bias' in spec:
                    self.ml_weights[key]['bias'] = float(spec['bias'])
                if 'weights' in spec:
                    weights = np.asarray(spec['weights'], dtype=float)
                    if weights.size == len(self.ml_feature_names):
                        self.ml_weights[key]['weights'] = weights
                    else:
                        self.get_logger().warn(
                            f'Ignoring ML weights for {key}: expected {len(self.ml_feature_names)}, got {weights.size}'
                        )
            self.get_logger().info(f'Loaded ML tuner model: {path}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to load ML tuner model {path}: {exc}')

    def _open_training_log(self):
        if not self.enable_training_log:
            return
        try:
            path = Path(self.training_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.training_log_file = path.open('a', encoding='utf-8', buffering=1)
            self.get_logger().info(f'Training log enabled: {path}')
        except Exception as exc:
            self.training_log_file = None
            self.enable_training_log = False
            self.get_logger().warn(f'Failed to open training log {self.training_log_path}: {exc}')

    def close_training_log(self):
        if self.training_log_file is not None:
            self.training_log_file.flush()
            self.training_log_file.close()
            self.training_log_file = None

    @staticmethod
    def _tuning_to_dict(tuning):
        return {
            'speed_scale': float(tuning.speed_scale),
            'clearance_weight_scale': float(tuning.clearance_weight_scale),
            'center_weight_scale': float(tuning.center_weight_scale),
            'raceline_weight_scale': float(tuning.raceline_weight_scale),
            'steering_smoothing': float(tuning.steering_smoothing),
            'steer_step_scale': float(tuning.steer_step_scale),
            'lateral_bias': float(tuning.lateral_bias),
            'risk': float(tuning.risk),
            'enabled': bool(tuning.enabled),
        }

    def _build_levine_guard_segments(self):
        segments = [
            ((-16.30, 13.70), (-14.42, 13.70)),  # upper-left exit
            ((-16.30, -6.88), (-14.42, -6.88)),  # lower-left exit
            ((12.55, 0.20), (12.55, 2.48)),      # right-side exit/dead end
        ]

        points = []
        for (x0, y0), (x1, y1) in segments:
            length = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            count = max(2, int(np.ceil(length / max(self.map_guard_resolution, 1e-3))) + 1)
            for t in np.linspace(0.0, 1.0, count):
                points.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return np.asarray(points, dtype=float)

    def _build_levine_waypoints(self):
        corners = np.asarray([
            (-8.0, -3.0),
            (8.0, -3.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (8.0, 6.50),
            (-12.0, 6.50),
            (-16.0, 6.0),
            (-16.0, -3.0),
            (-8.0, -3.0),
        ], dtype=float)
        points = []
        spacing = 0.22
        for start, end in zip(corners[:-1], corners[1:]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            count = max(2, int(np.ceil(length / spacing)) + 1)
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                points.append(start + t * delta)
        return np.asarray(points, dtype=float)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.odom_pose = (float(pose.position.x), float(pose.position.y), float(yaw))
        twist = msg.twist.twist
        self.odom_speed = float(sqrt(twist.linear.x * twist.linear.x + twist.linear.y * twist.linear.y))

    def add_map_guard_obstacles(self, obs_x, obs_y):
        if (
            not self.enable_map_guard
            or self.map_guard_name != 'levine'
            or self.odom_pose is None
            or self.levine_guard_segments.size == 0
        ):
            return obs_x, obs_y

        px, py, yaw = self.odom_pose
        guard = self.levine_guard_segments
        dx = guard[:, 0] - px
        dy = guard[:, 1] - py
        c = cos(yaw)
        s = sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        dist = np.sqrt(local_x * local_x + local_y * local_y)
        keep = np.logical_and(local_x > -0.55, dist < self.max_lidar_range)
        keep = np.logical_and(keep, np.abs(local_y) < 5.5)
        if not np.any(keep):
            return obs_x, obs_y

        if obs_x.size == 0:
            return local_x[keep].astype(float), local_y[keep].astype(float)
        return (
            np.concatenate((obs_x, local_x[keep].astype(float))),
            np.concatenate((obs_y, local_y[keep].astype(float))),
        )

    def compute_global_waypoint_drive(self, ranges, angles):
        if (
            not self.enable_global_waypoints
            or self.map_guard_name != 'levine'
            or self.odom_pose is None
            or self.levine_waypoints.size == 0
        ):
            return None

        px, py, yaw = self.odom_pose
        points = self.levine_waypoints
        dists = np.hypot(points[:, 0] - px, points[:, 1] - py)
        nearest = int(np.argmin(dists))
        step = max(4, int(round(self.waypoint_lookahead / 0.22)))
        target = points[(nearest + step) % points.shape[0]]

        dx = float(target[0] - px)
        dy = float(target[1] - py)
        c = cos(yaw)
        s = sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        if local_x < 0.25:
            target = points[(nearest + 2 * step) % points.shape[0]]
            dx = float(target[0] - px)
            dy = float(target[1] - py)
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy

        lookahead_sq = max(local_x * local_x + local_y * local_y, 1e-3)
        steer = atan2(2.0 * self.wheelbase * local_y, lookahead_sq)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        front = self.range_at(ranges, angles, 0.0, window=17)
        front_left = self.range_at(ranges, angles, np.deg2rad(28.0), window=11)
        front_right = self.range_at(ranges, angles, -np.deg2rad(28.0), window=11)
        min_front = min(front, front_left, front_right)
        min_scan = float(np.nanmin(ranges)) if ranges.size else self.max_lidar_range

        turn_ratio = float(np.clip(abs(steer) / max(self.max_steer, 1e-3), 0.0, 1.0))
        target_speed = self.waypoint_speed - turn_ratio * (self.waypoint_speed - self.waypoint_corner_speed)
        if min_front < 1.25:
            target_speed = min(target_speed, self.waypoint_corner_speed)
        if min_scan < 0.35:
            target_speed = self.emergency_speed

        now = self._now_seconds()
        if self.prev_time is None:
            dt = 0.05
        else:
            dt = float(np.clip(now - self.prev_time, 0.005, 0.20))
        self.prev_time = now

        if target_speed > self.prev_speed:
            speed = min(target_speed, self.prev_speed + self.accel_limit * dt)
        else:
            speed = max(target_speed, self.prev_speed - self.decel_limit * dt)
        self.prev_speed = float(np.clip(speed, self.emergency_speed, self.max_speed))
        self.prev_steer = steer
        return steer, self.prev_speed

    def _predict_linear(self, name, features):
        spec = self.ml_weights[name]
        return float(spec['bias'] + np.dot(spec['weights'], features))

    def extract_ml_features(self, track, gap_target, wall_assist):
        front_min = min(track.front_clearance, track.front_left_clearance, track.front_right_clearance)
        side_min = min(track.side_left_clearance, track.side_right_clearance)
        side_balance = track.side_left_clearance - track.side_right_clearance
        wall_error = wall_assist.error if wall_assist.exists else 0.0
        gap_angle = gap_target.angle if gap_target.exists else 0.0
        gap_distance = gap_target.distance if gap_target.exists else track.front_clearance
        gap_width = gap_target.width if gap_target.exists else track.width

        feature_values = {
            'front_clearance': np.clip(track.front_clearance / max(self.max_lidar_range, 1e-3), 0.0, 1.0),
            'front_min_clearance': np.clip(front_min / max(self.max_lidar_range, 1e-3), 0.0, 1.0),
            'gap_distance': np.clip(gap_distance / max(self.max_lidar_range, 1e-3), 0.0, 1.0),
            'gap_width': np.clip(gap_width / 3.2, 0.0, 1.0),
            'signed_gap_angle': np.clip(gap_angle / max(self.max_steer, 1e-3), -1.0, 1.0),
            'abs_gap_angle': np.clip(abs(gap_angle) / max(self.max_steer, 1e-3), 0.0, 1.0),
            'track_curvature': np.clip(track.curvature / 0.75, 0.0, 1.0),
            'side_min_clearance': np.clip(side_min / 1.8, 0.0, 1.0),
            'side_balance': np.clip(side_balance / 2.0, -1.0, 1.0),
            'wall_error': np.clip(wall_error / 1.6, -1.0, 1.0),
            'abs_wall_error': np.clip(abs(wall_error) / 1.6, 0.0, 1.0),
            'prev_speed': np.clip(self.prev_speed / max(self.max_speed, 1e-3), 0.0, 1.0),
            'abs_prev_steer': np.clip(abs(self.prev_steer) / max(self.max_steer, 1e-3), 0.0, 1.0),
            'center_confidence': np.clip(track.confidence, 0.0, 1.0),
        }
        features = np.asarray([feature_values[name] for name in self.ml_feature_names], dtype=float)
        self.last_ml_feature_values = {name: float(feature_values[name]) for name in self.ml_feature_names}
        self.last_ml_features = features
        return features

    def compute_ml_tuning(self, track, gap_target, wall_assist):
        features = self.extract_ml_features(track, gap_target, wall_assist)
        if not self.enable_ml_tuner:
            self.prev_ml_tuning = self._neutral_ml_tuning(enabled=False)
            return self.prev_ml_tuning

        risk = self._sigmoid(self._predict_linear('risk', features))
        speed_scale = 1.0 + self._predict_linear('speed_scale', features)
        speed_scale *= (1.0 - self.ml_risk_slowdown * risk)
        speed_scale = float(np.clip(speed_scale, self.ml_speed_scale_min, self.ml_speed_scale_max))

        clearance_weight_scale = 1.0 + self._predict_linear('clearance_weight_scale', features) + 0.35 * risk
        center_weight_scale = 1.0 + self._predict_linear('center_weight_scale', features) + 0.22 * risk
        raceline_weight_scale = 1.0 + self._predict_linear('raceline_weight_scale', features) - 0.18 * risk
        steer_step_scale = 1.0 + self._predict_linear('steer_step_scale', features) + 0.10 * risk
        steering_smoothing = self._predict_linear('steering_smoothing', features) + 0.07 * risk
        lateral_bias = self._predict_linear('lateral_bias', features)

        new_tuning = MLTuning(
            speed_scale=speed_scale,
            clearance_weight_scale=float(np.clip(clearance_weight_scale, 0.85, 1.55)),
            center_weight_scale=float(np.clip(center_weight_scale, 0.85, 1.45)),
            raceline_weight_scale=float(np.clip(raceline_weight_scale, 0.62, 1.28)),
            steering_smoothing=float(np.clip(steering_smoothing, 0.24, 0.72)),
            steer_step_scale=float(np.clip(steer_step_scale, 0.78, 1.22)),
            lateral_bias=float(np.clip(lateral_bias, -self.ml_lateral_bias_limit, self.ml_lateral_bias_limit)),
            risk=float(np.clip(risk, 0.0, 1.0)),
            enabled=True,
        )

        a = self.ml_smoothing_alpha
        old = self.prev_ml_tuning
        smoothed = MLTuning(
            speed_scale=(1.0 - a) * old.speed_scale + a * new_tuning.speed_scale,
            clearance_weight_scale=(1.0 - a) * old.clearance_weight_scale + a * new_tuning.clearance_weight_scale,
            center_weight_scale=(1.0 - a) * old.center_weight_scale + a * new_tuning.center_weight_scale,
            raceline_weight_scale=(1.0 - a) * old.raceline_weight_scale + a * new_tuning.raceline_weight_scale,
            steering_smoothing=(1.0 - a) * old.steering_smoothing + a * new_tuning.steering_smoothing,
            steer_step_scale=(1.0 - a) * old.steer_step_scale + a * new_tuning.steer_step_scale,
            lateral_bias=(1.0 - a) * old.lateral_bias + a * new_tuning.lateral_bias,
            risk=(1.0 - a) * old.risk + a * new_tuning.risk,
            enabled=True,
        )
        self.prev_ml_tuning = smoothed
        return smoothed

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def preprocess_lidar(self, scan_msg):
        raw_ranges = np.asarray(scan_msg.ranges, dtype=float)
        angles = scan_msg.angle_min + np.arange(raw_ranges.size) * scan_msg.angle_increment

        ranges = np.where(np.isfinite(raw_ranges), raw_ranges, self.max_lidar_range)
        ranges = np.clip(ranges, 0.0, self.max_lidar_range)
        ranges = self._smooth(ranges, self.smoothing_window)

        half_fov = np.deg2rad(0.5 * self.forward_fov_deg)
        front_mask = np.logical_and(angles >= -half_fov, angles <= half_fov)
        return ranges, angles, ranges[front_mask], angles[front_mask]

    def range_at(self, ranges, angles, target_angle, window=9):
        if ranges.size == 0:
            return self.max_lidar_range
        idx = int(np.argmin(np.abs(angles - target_angle)))
        half = max(0, window // 2)
        lo = max(0, idx - half)
        hi = min(ranges.size, idx + half + 1)
        sample = ranges[lo:hi]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return self.max_lidar_range
        return float(np.median(sample))

    def obstacle_points(self, ranges, angles):
        valid = np.logical_and(np.isfinite(ranges), ranges > 0.04)
        valid = np.logical_and(valid, ranges < self.max_lidar_range * 0.985)
        if not np.any(valid):
            return np.array([], dtype=float), np.array([], dtype=float)
        rs = ranges[valid]
        th = angles[valid]
        xs = rs * np.cos(th)
        ys = rs * np.sin(th)
        keep = np.logical_and(xs > -0.45, np.abs(ys) < 5.5)
        return xs[keep], ys[keep]

    def compute_gap_target(self, front_ranges, front_angles, angle_increment):
        if front_ranges.size == 0:
            return GapTarget(0.0, 0.0, 0.0, False)

        free = front_ranges.copy()
        obstacle_candidates = np.where(free > 0.05, free, self.max_lidar_range)
        closest_idx = int(np.argmin(obstacle_candidates))
        closest_dist = float(obstacle_candidates[closest_idx])
        if isfinite(closest_dist) and closest_dist < self.max_lidar_range:
            bubble_angle = np.arctan2(self.bubble_radius, max(closest_dist, 0.05))
            bubble_size = max(3, int(np.ceil(bubble_angle / max(abs(angle_increment), 1e-6))))
            lo = max(0, closest_idx - bubble_size)
            hi = min(free.size, closest_idx + bubble_size + 1)
            free[lo:hi] = 0.0

        is_free = free >= self.min_gap_range
        best_start = 0
        best_end = -1
        cur_start = None
        for idx, ok in enumerate(is_free):
            if ok and cur_start is None:
                cur_start = idx
            elif not ok and cur_start is not None:
                if idx - cur_start > best_end - best_start + 1:
                    best_start = cur_start
                    best_end = idx - 1
                cur_start = None
        if cur_start is not None and free.size - cur_start > best_end - best_start + 1:
            best_start = cur_start
            best_end = free.size - 1
        if best_end < best_start:
            return GapTarget(0.0, 0.0, 0.0, False)

        gap_ranges = free[best_start:best_end + 1]
        gap_angles = front_angles[best_start:best_end + 1]
        smoothed = self._smooth(gap_ranges, self.gap_window)
        range_score = smoothed / max(float(np.max(smoothed)), 1e-6)
        edge_distance = np.minimum(
            np.arange(smoothed.size),
            np.arange(smoothed.size - 1, -1, -1),
        ).astype(float)
        edge_score = edge_distance / max(float(np.max(edge_distance)), 1.0)
        heading_score = 1.0 - np.abs(gap_angles) / max(float(np.max(np.abs(front_angles))), 1e-6)
        score = 0.50 * range_score + 0.32 * edge_score + 0.18 * heading_score
        best = int(np.argmax(score))

        half = self.gap_window // 2
        lo = max(0, best - half)
        hi = min(gap_angles.size, best + half + 1)
        weights = np.maximum(score[lo:hi], 1e-3)
        angle = float(np.average(gap_angles[lo:hi], weights=weights))
        distance = float(np.average(gap_ranges[lo:hi], weights=weights))
        width = max(0.0, (best_end - best_start + 1) * abs(angle_increment) * distance)
        return GapTarget(angle, distance, width, True)

    def projected_wall_distance(self, ranges, angles, side_sign):
        theta = np.deg2rad(self.wall_theta_deg)
        if side_sign > 0:
            b_angle = np.deg2rad(90.0)
            a_angle = np.deg2rad(90.0) - theta
        else:
            b_angle = -np.deg2rad(90.0)
            a_angle = -np.deg2rad(90.0) + theta

        a = self.range_at(ranges, angles, a_angle, window=7)
        b = self.range_at(ranges, angles, b_angle, window=7)
        valid = (
            isfinite(a)
            and isfinite(b)
            and 0.05 < a < self.max_lidar_range * 0.96
            and 0.05 < b < self.max_lidar_range * 0.96
        )
        if not valid:
            return b, b, 0.0, False

        denominator = max(a * sin(theta), 1e-4)
        alpha = atan2(a * cos(theta) - b, denominator)
        current = b * cos(alpha)
        lookahead = self.wall_lookahead_base + self.wall_lookahead_speed_gain * max(self.prev_speed, 0.0)
        future = current + lookahead * sin(alpha)
        future = float(np.clip(future, 0.0, self.max_lidar_range))
        return future, float(current), float(alpha), True

    def compute_wall_assist(self, ranges, angles, dt):
        left_future, left_current, _left_alpha, left_ok = self.projected_wall_distance(ranges, angles, 1.0)
        right_future, right_current, _right_alpha, right_ok = self.projected_wall_distance(ranges, angles, -1.0)

        side_clearance = min(left_current, right_current)
        if left_ok and right_ok:
            error = right_future - left_future
            confidence = 1.0
            target_y = float(np.clip(-0.5 * error, -self.wall_target_clip, self.wall_target_clip))
        elif left_ok:
            error = self.preferred_wall_clearance - left_future
            confidence = 0.45
            target_y = float(np.clip(-error, -self.wall_target_clip, self.wall_target_clip))
        elif right_ok:
            error = right_future - self.preferred_wall_clearance
            confidence = 0.45
            target_y = float(np.clip(-error, -self.wall_target_clip, self.wall_target_clip))
        else:
            self.prev_wall_error = 0.0
            return WallAssist(0.0, 0.0, 0.0, 0.0, left_future, right_future, side_clearance, False)

        derivative = (error - self.prev_wall_error) / max(dt, 1e-3)
        self.prev_wall_error = error
        correction = -self.wall_assist_kp * error - self.wall_assist_kd * derivative
        correction = float(np.clip(correction * confidence, -self.max_wall_correction, self.max_wall_correction))
        return WallAssist(correction, target_y, float(error), confidence, left_future, right_future, side_clearance, True)

    def compute_raceline_bias(self, gap_target, track):
        if not gap_target.exists:
            self.prev_raceline_correction *= 0.70
            return RaceLineBias(self.prev_raceline_correction, 0.0, 0.0, 0.0, False, False)

        now = self._now_seconds()
        front_imbalance = track.front_left_clearance - track.front_right_clearance
        turn_hint = gap_target.angle
        if abs(turn_hint) < np.deg2rad(5.0):
            turn_hint = 0.32 * np.clip(front_imbalance, -1.2, 1.2)

        turn_intensity = float(np.clip(abs(turn_hint) / max(self.max_steer, 1e-3), 0.0, 1.0))
        clear_enough = (
            track.front_clearance > self.raceline_front_clearance
            and min(track.front_left_clearance, track.front_right_clearance) > 0.82
        )
        if turn_intensity < self.raceline_min_turn_angle or not clear_enough:
            self.prev_raceline_correction *= 0.70
            return RaceLineBias(self.prev_raceline_correction, 0.0, 0.0, 0.0, False, now < self.s_bend_until)

        turn_sign = float(np.sign(turn_hint))
        countersteer = (
            self.turn_memory_sign != 0.0
            and turn_sign != self.turn_memory_sign
            and now < self.turn_memory_until
            and abs(turn_hint) > self.raceline_countersteer_angle
        )
        if countersteer:
            self.s_bend_until = now + self.raceline_s_bend_hold_time

        self.turn_memory_sign = turn_sign
        self.turn_memory_until = now + self.raceline_turn_memory_time

        s_bend_active = now < self.s_bend_until
        setup_scale = 0.18 if s_bend_active else 1.0
        front_pressure = 1.0 - float(np.clip((track.front_clearance - 1.0) / 3.8, 0.0, 1.0))
        setup_weight = float(np.clip(0.25 + 0.75 * front_pressure, 0.0, 1.0)) * setup_scale

        outside_direction = -turn_sign
        target_y = outside_direction * self.raceline_target_y * turn_intensity * setup_weight
        correction = outside_direction * self.raceline_gain * turn_intensity * setup_weight

        # Keep the outside setup through the corner mouth. The apex pull stays
        # deliberately small so the planner does not cut into the inner corner.
        apex_correction = self.raceline_apex_pull_gain * self.raceline_gain * gap_target.angle * front_pressure
        correction += apex_correction
        correction = 0.62 * self.prev_raceline_correction + 0.38 * correction
        correction = float(np.clip(correction, -self.raceline_gain, self.raceline_gain))
        self.prev_raceline_correction = correction

        weight = self.raceline_score_weight * setup_weight
        return RaceLineBias(
            correction=correction,
            target_y=float(target_y),
            weight=float(weight),
            turn_sign=turn_sign,
            active=True,
            s_bend_active=s_bend_active,
        )

    def estimate_track(self, ranges, angles, obs_x, obs_y):
        front = self.range_at(ranges, angles, 0.0)
        front_left = self.range_at(ranges, angles, np.deg2rad(32.0))
        front_right = self.range_at(ranges, angles, -np.deg2rad(32.0))
        side_left = self.range_at(ranges, angles, np.deg2rad(88.0))
        side_right = self.range_at(ranges, angles, -np.deg2rad(88.0))

        horizon = float(np.clip(
            self.min_horizon + self.horizon_speed_gain * max(self.prev_speed, 0.0),
            self.min_horizon,
            self.max_horizon,
        ))
        sample_xs = np.linspace(0.65, horizon, self.center_samples)
        centers = []
        widths = []

        if obs_x.size >= 16:
            for sample_x in sample_xs:
                window = np.abs(obs_x - sample_x) <= self.center_slice_width
                left_y = obs_y[np.logical_and(window, obs_y > 0.06)]
                right_y = obs_y[np.logical_and(window, obs_y < -0.06)]
                if left_y.size < 2 or right_y.size < 2:
                    continue

                left_wall = float(np.percentile(left_y, 18.0))
                right_wall = float(np.percentile(right_y, 82.0))
                width = left_wall - right_wall
                if width < self.min_track_width or width > self.max_track_width:
                    continue

                centers.append((sample_x, 0.5 * (left_wall + right_wall)))
                widths.append(width)

        confidence = float(np.clip(len(centers) / max(float(self.center_samples), 1.0), 0.0, 1.0))
        if len(centers) < 2:
            return TrackModel(
                np.array([], dtype=float),
                np.array([], dtype=float),
                confidence,
                0.0,
                0.0,
                front,
                front_left,
                front_right,
                side_left,
                side_right,
            )

        points = np.asarray(centers, dtype=float)
        if len(centers) >= 4:
            weights = 0.8 + points[:, 0] / max(horizon, 1e-3)
            coeff = np.polyfit(points[:, 0], points[:, 1], 2, w=weights)
            dense_x = np.linspace(points[0, 0], points[-1, 0], self.center_samples)
            dense_y = np.polyval(coeff, dense_x)
            slope = 2.0 * coeff[0] * dense_x[-1] + coeff[1]
            curvature = abs(2.0 * coeff[0]) / max((1.0 + slope * slope) ** 1.5, 1e-3)
        else:
            dense_x = points[:, 0]
            dense_y = points[:, 1]
            curvature = 0.0

        return TrackModel(
            dense_x.astype(float),
            dense_y.astype(float),
            confidence,
            float(curvature),
            float(np.median(widths)) if widths else 0.0,
            front,
            front_left,
            front_right,
            side_left,
            side_right,
        )

    def generate_path(self, curvature, horizon):
        s = np.arange(self.path_ds, horizon + self.path_ds, self.path_ds, dtype=float)
        if abs(curvature) < 1e-5:
            xs = s
            ys = np.zeros_like(s)
            headings = np.zeros_like(s)
        else:
            k = float(curvature)
            xs = np.sin(k * s) / k
            ys = (1.0 - np.cos(k * s)) / k
            headings = k * s
        return xs, ys, headings

    def clearance_for_path(self, path_x, path_y, obs_x, obs_y):
        if obs_x.size == 0:
            return self.max_lidar_range, self.max_lidar_range

        px = path_x[path_x > 0.20]
        py = path_y[path_x > 0.20]
        if px.size == 0:
            return self.max_lidar_range, self.max_lidar_range

        min_dist = self.max_lidar_range
        clearances = []
        chunk_size = 12
        for start in range(0, px.size, chunk_size):
            end = min(px.size, start + chunk_size)
            dx = px[start:end, None] - obs_x[None, :]
            dy = py[start:end, None] - obs_y[None, :]
            dists = np.sqrt(dx * dx + dy * dy)
            local_min = np.min(dists, axis=1)
            clearances.extend(local_min.tolist())
            min_dist = min(min_dist, float(np.min(local_min)))
            if min_dist < self.collision_clearance:
                break

        if not clearances:
            return self.max_lidar_range, self.max_lidar_range
        return float(min_dist), float(np.mean(np.clip(clearances, 0.0, self.max_lidar_range)))

    @property
    def collision_clearance(self):
        return 0.5 * self.car_width + self.safety_margin

    def center_error_for_path(self, path_x, path_y, track, center_bias=0.0):
        if track.xs.size < 2 or track.confidence < self.min_center_confidence:
            return 0.0
        order = np.argsort(path_x)
        sorted_x = path_x[order]
        sorted_y = path_y[order]
        usable = np.logical_and(track.xs >= sorted_x[0], track.xs <= sorted_x[-1])
        if not np.any(usable):
            return 0.0
        path_y_at_center_x = np.interp(track.xs[usable], sorted_x, sorted_y)
        error = path_y_at_center_x - (track.ys[usable] + center_bias)
        return float(np.sqrt(np.mean(error * error)))

    def score_candidate(
        self,
        curvature,
        path_x,
        path_y,
        headings,
        obs_x,
        obs_y,
        track,
        gap_target,
        wall_assist,
        raceline,
        tuning,
        horizon,
    ):
        min_clearance, mean_clearance = self.clearance_for_path(path_x, path_y, obs_x, obs_y)
        safe = min_clearance >= self.collision_clearance

        progress = float(np.clip(path_x[-1] / max(horizon, 1e-3), 0.0, 1.0))
        clearance_norm = float(np.clip(
            (min_clearance - self.collision_clearance)
            / max(self.preferred_wall_clearance - self.collision_clearance, 1e-3),
            -1.0,
            1.25,
        ))
        center_error = self.center_error_for_path(path_x, path_y, track, tuning.lateral_bias)
        center_score = 1.0 - float(np.clip(center_error / 0.72, 0.0, 1.0))

        target_angle = atan2(path_y[min(path_y.size - 1, path_y.size // 2)], max(path_x[min(path_x.size - 1, path_x.size // 2)], 1e-3))
        if gap_target.exists:
            gap_score = 1.0 - float(np.clip(abs(target_angle - gap_target.angle) / np.deg2rad(55.0), 0.0, 1.0))
        else:
            gap_score = 0.0

        if wall_assist.exists:
            wall_idx = int(np.argmin(np.abs(path_x - min(1.45, path_x[-1]))))
            wall_error = abs(float(path_y[wall_idx]) - wall_assist.target_y)
            wall_score = 1.0 - float(np.clip(wall_error / max(self.wall_target_clip, 1e-3), 0.0, 1.0))
        else:
            wall_score = 0.0

        if raceline.active and raceline.weight > 0.0:
            race_idx = int(np.argmin(np.abs(path_x - min(self.raceline_setup_distance, path_x[-1]))))
            race_error = abs(float(path_y[race_idx]) - raceline.target_y)
            raceline_score = 1.0 - float(np.clip(race_error / max(self.raceline_target_y, 1e-3), 0.0, 1.0))
            inside_cut = max(0.0, raceline.turn_sign * float(path_y[race_idx]) + 0.03)
            inner_corner_penalty = self.inner_corner_penalty_weight * float(
                np.clip(inside_cut / max(self.raceline_target_y, 1e-3), 0.0, 1.0)
            )
        else:
            raceline_score = 0.0
            inner_corner_penalty = 0.0

        curvature_penalty = abs(curvature) / max(self.max_candidate_curvature, 1e-3)
        smooth_penalty = abs(curvature - self.prev_curvature) / max(self.max_candidate_curvature, 1e-3)
        wall_penalty = 0.0
        if min(track.side_left_clearance, track.side_right_clearance) < self.hard_wall_clearance:
            wall_penalty = 2.5
        if wall_assist.exists and wall_assist.side_clearance < self.hard_wall_clearance:
            wall_penalty += 2.0

        score = (
            self.progress_weight * progress
            + (self.clearance_weight * tuning.clearance_weight_scale) * clearance_norm
            + (self.center_weight * tuning.center_weight_scale) * track.confidence * center_score
            + self.gap_weight * gap_score
            + self.wall_assist_weight * wall_assist.confidence * wall_score
            + (raceline.weight * tuning.raceline_weight_scale) * raceline_score
            - self.curvature_weight * curvature_penalty
            - self.smooth_weight * smooth_penalty
            - wall_penalty
            - inner_corner_penalty
        )
        if not safe:
            score -= 100.0 + 30.0 * (self.collision_clearance - min_clearance)

        return Candidate(
            curvature=float(curvature),
            xs=path_x,
            ys=path_y,
            headings=headings,
            min_clearance=float(min_clearance),
            mean_clearance=float(mean_clearance),
            score=float(score),
            safe=bool(safe),
            center_error=float(center_error),
            progress=progress,
        )

    def plan_trajectory(self, obs_x, obs_y, track, gap_target, wall_assist, raceline, tuning):
        horizon = float(np.clip(
            self.min_horizon + self.horizon_speed_gain * max(self.prev_speed, 0.0),
            self.min_horizon,
            self.max_horizon,
        ))
        curvatures = np.linspace(-self.max_candidate_curvature, self.max_candidate_curvature, self.trajectory_count)

        candidates = []
        for curvature in curvatures:
            path_x, path_y, headings = self.generate_path(float(curvature), horizon)
            candidates.append(
                self.score_candidate(
                    float(curvature),
                    path_x,
                    path_y,
                    headings,
                    obs_x,
                    obs_y,
                    track,
                    gap_target,
                    wall_assist,
                    raceline,
                    tuning,
                    horizon,
                )
            )

        best = max(candidates, key=lambda item: item.score)
        if best.safe:
            self.best_miss_count = 0
            return best

        safe_candidates = [item for item in candidates if item.min_clearance > self.hard_wall_clearance]
        if safe_candidates:
            self.best_miss_count += 1
            return max(safe_candidates, key=lambda item: item.score)

        self.best_miss_count += 1
        return best

    def compute_steering(self, candidate, gap_target, wall_assist, raceline, tuning):
        lookahead = float(np.clip(
            self.min_lookahead + self.lookahead_speed_gain * max(self.prev_speed, 0.0),
            self.min_lookahead,
            self.max_lookahead,
        ))
        distances = np.sqrt(candidate.xs * candidate.xs + candidate.ys * candidate.ys)
        target_idx = int(np.argmin(np.abs(distances - lookahead)))
        target_x = float(candidate.xs[target_idx])
        target_y = float(candidate.ys[target_idx])

        pursuit = atan2(2.0 * self.wheelbase * target_y, max(target_x * target_x + target_y * target_y, 1e-3))
        feedforward = atan2(self.wheelbase * candidate.curvature, 1.0)
        raw = self.pure_pursuit_gain * pursuit + self.feedforward_gain * feedforward

        if (not candidate.safe or candidate.min_clearance < self.preferred_wall_clearance * 0.80) and gap_target.exists:
            raw = 0.72 * raw + 0.28 * gap_target.angle
        if wall_assist.exists:
            raw += wall_assist.correction
        if raceline.active:
            raw += raceline.correction

        raw = float(np.clip(raw, -self.max_steer, self.max_steer))
        sharp = abs(raw) > 0.19 or abs(candidate.curvature) > 0.32
        base_smoothing = self.corner_steering_smoothing if sharp else self.steering_smoothing
        if tuning.enabled:
            smoothing = 0.65 * base_smoothing + 0.35 * tuning.steering_smoothing
        else:
            smoothing = base_smoothing
        base_step = self.corner_steer_step if sharp else self.max_steer_step
        step = base_step * (tuning.steer_step_scale if tuning.enabled else 1.0)

        smoothed = smoothing * self.prev_steer + (1.0 - smoothing) * raw
        delta = float(np.clip(smoothed - self.prev_steer, -step, step))
        steer = float(np.clip(self.prev_steer + delta, -self.max_steer, self.max_steer))
        self.prev_steer = steer
        self.prev_curvature = candidate.curvature
        return steer

    def compute_speed(self, steering, candidate, track, wall_assist, raceline, tuning):
        now = self._now_seconds()
        if self.prev_time is None:
            dt = 0.05
        else:
            dt = float(np.clip(now - self.prev_time, 0.005, 0.20))
        self.prev_time = now

        front_clearance = min(track.front_clearance, track.front_left_clearance, track.front_right_clearance)
        side_clearance = min(track.side_left_clearance, track.side_right_clearance)
        if wall_assist.exists:
            side_clearance = min(side_clearance, wall_assist.side_clearance)
        steering_curvature = abs(tan(steering)) / self.wheelbase
        curvature = max(abs(candidate.curvature), steering_curvature, track.curvature)

        emergency = (
            self.best_miss_count > 2
            or candidate.min_clearance < self.collision_clearance
            or front_clearance < 0.62
            or side_clearance < self.hard_wall_clearance
        )
        if emergency:
            target = self.emergency_speed
        else:
            curve_limit = sqrt(self.lat_accel_limit / max(curvature, 1e-3))
            clearance_ratio = np.clip(
                (candidate.min_clearance - self.collision_clearance)
                / max(self.preferred_wall_clearance - self.collision_clearance, 1e-3),
                0.0,
                1.0,
            )
            clearance_limit = self.corner_speed + clearance_ratio * (self.max_speed - self.corner_speed)
            front_limit = 1.0 + 1.8 * min(track.front_clearance, 4.0)
            target = min(self.max_speed, curve_limit, clearance_limit, front_limit)

            brake_distance = max(0.0, (self.prev_speed * self.prev_speed - self.corner_speed * self.corner_speed) / (2.0 * self.brake_decel))
            if front_clearance < brake_distance + self.brake_buffer:
                target = min(target, self.medium_speed)

            if curvature < 0.060 and abs(steering) < 0.075 and front_clearance > 3.2 and candidate.min_clearance > 0.62:
                target = max(target, self.straight_speed)
            elif curvature < 0.155 and abs(steering) < 0.18 and front_clearance > 2.3:
                target = max(target, self.fast_curve_speed)
            elif abs(steering) < 0.28:
                target = max(target, self.medium_speed)
            else:
                target = max(target, self.corner_speed)

            if wall_assist.exists and abs(wall_assist.error) > 1.05 and side_clearance < 0.85:
                target = min(target, self.medium_speed)
            if raceline.active and not raceline.s_bend_active and abs(candidate.curvature) > 0.22:
                target = min(target, max(self.corner_speed + 0.85, self.medium_speed))
            if candidate.min_clearance < 0.55 or side_clearance < 0.58:
                target = min(target, self.corner_speed)
            if tuning.enabled:
                target = float(np.clip(target * tuning.speed_scale, self.emergency_speed, self.max_speed))

        if target > self.prev_speed:
            speed = min(target, self.prev_speed + self.accel_limit * dt)
        else:
            speed = max(target, self.prev_speed - self.decel_limit * dt)

        speed = float(np.clip(speed, self.emergency_speed, self.max_speed))
        self.prev_speed = speed
        return speed

    def log_training_sample(
        self,
        scan_msg,
        track,
        gap_target,
        wall_assist,
        raceline,
        tuning,
        candidate,
        steering,
        speed,
    ):
        if not self.enable_training_log or self.training_log_file is None:
            return
        self.training_log_count += 1
        if self.training_log_count % self.training_log_stride != 0:
            return

        odom = {}
        if self.odom_pose is not None:
            odom = {
                'x': float(self.odom_pose[0]),
                'y': float(self.odom_pose[1]),
                'yaw': float(self.odom_pose[2]),
                'speed': float(self.odom_speed),
            }

        record = {
            'stamp': float(scan_msg.header.stamp.sec) + 1e-9 * float(scan_msg.header.stamp.nanosec),
            'mode': 'local_trajectory',
            'feature_names': self.ml_feature_names,
            'features': self.last_ml_features.astype(float).tolist(),
            'feature_values': self.last_ml_feature_values,
            'tuning': self._tuning_to_dict(tuning),
            'drive': {
                'steering_angle': float(steering),
                'speed': float(speed),
            },
            'candidate': {
                'curvature': float(candidate.curvature),
                'min_clearance': float(candidate.min_clearance),
                'mean_clearance': float(candidate.mean_clearance),
                'score': float(candidate.score),
                'safe': bool(candidate.safe),
                'center_error': float(candidate.center_error),
            },
            'track': {
                'confidence': float(track.confidence),
                'curvature': float(track.curvature),
                'width': float(track.width),
                'front_clearance': float(track.front_clearance),
                'front_left_clearance': float(track.front_left_clearance),
                'front_right_clearance': float(track.front_right_clearance),
                'side_left_clearance': float(track.side_left_clearance),
                'side_right_clearance': float(track.side_right_clearance),
            },
            'gap': {
                'angle': float(gap_target.angle),
                'distance': float(gap_target.distance),
                'width': float(gap_target.width),
                'exists': bool(gap_target.exists),
            },
            'wall': {
                'error': float(wall_assist.error),
                'confidence': float(wall_assist.confidence),
                'side_clearance': float(wall_assist.side_clearance),
                'exists': bool(wall_assist.exists),
            },
            'raceline': {
                'active': bool(raceline.active),
                's_bend_active': bool(raceline.s_bend_active),
                'turn_sign': float(raceline.turn_sign),
                'weight': float(raceline.weight),
            },
            'odom': odom,
            'params': {
                'max_speed': float(self.max_speed),
                'straight_speed': float(self.straight_speed),
                'fast_curve_speed': float(self.fast_curve_speed),
                'medium_speed': float(self.medium_speed),
                'corner_speed': float(self.corner_speed),
            },
        }
        self.training_log_file.write(json.dumps(record, separators=(',', ':')) + '\n')
        if self.training_log_count % max(1, self.training_log_stride * 50) == 0:
            self.training_log_file.flush()

    def publish_drive(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.steering_angle = float(np.clip(steering, -self.max_steer, self.max_steer))
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def scan_callback(self, scan_msg):
        ranges, angles, front_ranges, front_angles = self.preprocess_lidar(scan_msg)
        if ranges.size == 0 or front_ranges.size == 0:
            self.publish_drive(0.0, 0.0)
            return

        waypoint_drive = self.compute_global_waypoint_drive(ranges, angles)
        if waypoint_drive is not None:
            steering, speed = waypoint_drive
            self.publish_drive(steering, speed)
            return

        now = self._now_seconds()
        if self.prev_time is None:
            dt = 0.05
        else:
            dt = float(np.clip(now - self.prev_time, 0.005, 0.20))
        obs_x, obs_y = self.obstacle_points(ranges, angles)
        obs_x, obs_y = self.add_map_guard_obstacles(obs_x, obs_y)
        track = self.estimate_track(ranges, angles, obs_x, obs_y)
        gap_target = self.compute_gap_target(front_ranges, front_angles, scan_msg.angle_increment)
        wall_assist = self.compute_wall_assist(ranges, angles, dt)
        raceline = self.compute_raceline_bias(gap_target, track)
        tuning = self.compute_ml_tuning(track, gap_target, wall_assist)
        candidate = self.plan_trajectory(obs_x, obs_y, track, gap_target, wall_assist, raceline, tuning)
        steering = self.compute_steering(candidate, gap_target, wall_assist, raceline, tuning)
        speed = self.compute_speed(steering, candidate, track, wall_assist, raceline, tuning)
        self.publish_drive(steering, speed)
        self.log_training_sample(
            scan_msg,
            track,
            gap_target,
            wall_assist,
            raceline,
            tuning,
            candidate,
            steering,
            speed,
        )


def main(args=None):
    rclpy.init(args=args)
    print('RacePlanner Initialized')
    node = RacePlanner()
    rclpy.spin(node)
    node.close_training_log()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
