#!/usr/bin/env python3
"""
soccer_env.py — Isaac Lab Soccer Environment (Modern API)
==========================================================
Uses the isaaclab package directly (omni.isaac.lab is deprecated).
Compatible with Isaac Lab 2.x + Isaac Sim 4.2+

Privileged state observation — no camera, no ROS, no bridge.
Everything runs on GPU tensors via PhysX.

Observation space (12 values, all in robot-local frame):
  [0-1]  ball_rel_xy     — ball position relative to robot (m)
  [2]    ball_dist       — distance to ball (m)
  [3]    ball_angle      — angle to ball in robot frame (rad)
  [4-5]  goal_rel_xy     — goal position relative to robot (m)
  [6]    goal_dist       — distance to goal (m)
  [7]    goal_angle      — angle to goal in robot frame (rad)
  [8]    robot_vx        — robot linear velocity (m/s)
  [9]    robot_wz        — robot angular velocity (rad/s)
  [10-11] ball_to_goal   — ball→goal vector (m)

Action space (2 continuous):
  [0]  linear.x   clamped to [0.0, 1.0] m/s
  [1]  angular.z  clamped to [-0.6, 0.6] rad/s
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

# ── Modern Isaac Lab imports (isaaclab, not omni.isaac.lab) ──────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation, ArticulationCfg,
    RigidObject, RigidObjectCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat


# ── World constants ──────────────────────────────────────────────────────────
GOAL_X          =  5.0
GOAL_Y          =  0.0
GOAL_HALF_WIDTH =  0.9
GOAL_DEPTH      =  0.35

ARENA_X_MIN, ARENA_X_MAX = -3.0,  5.5
ARENA_Y_MIN, ARENA_Y_MAX = -4.5,  4.5

ROBOT_SPAWN_X_RANGE = (-1.5,  0.0)
ROBOT_SPAWN_Y_RANGE = (-1.5,  1.5)
# In soccer_env.py, change these constants
ROBOT_YAW_RANGE  = (-1.5,  1.5)   # was (-0.5, 0.5) — robot can start facing sideways
BALL_DIST_RANGE  = ( 1.0,  5.0)   # was (1.0, 3.0)  — ball can be farther away

# TurtleBot3 Waffle Pi kinematics
WHEEL_SEPARATION = 0.287   # m
WHEEL_RADIUS     = 0.033   # m


# ── Environment config ───────────────────────────────────────────────────────
@configclass
class SoccerEnvCfg(DirectRLEnvCfg):

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
    )

    # decimation: number of physics steps per RL step
    # 2 physics steps × 0.01s = 0.02s per RL step = 50 Hz control
    decimation: int = 2

    # Episode length in seconds
    episode_length_s: float = 10.0

    # Number of parallel environments
    num_envs:    int   = 1024
    env_spacing: float = 12.0

    # Observation and action space sizes (required in Isaac Lab 2.x)
    observation_space: int = 12
    action_space:      int = 2
    state_space:       int = 0   # no asymmetric critic

    # ── Ball ─────────────────────────────────────────────────────────
    ball_radius: float = 0.15
    ball_mass:   float = 0.1

    # ── Scene ────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=12.0,
        replicate_physics=True,
    )


# ── Environment ──────────────────────────────────────────────────────────────
class SoccerEnv(DirectRLEnv):
    cfg: SoccerEnvCfg

    def __init__(self, cfg: SoccerEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode=render_mode)

        self._goal_pos = torch.tensor(
            [GOAL_X, GOAL_Y, 0.0], device=self.device)

        self._prev_ball_to_goal = torch.zeros(
            self.num_envs, device=self.device)

        self._goal_reward = torch.zeros(
            self.num_envs, device=self.device)

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        # Ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(color=(0.25, 0.55, 0.25)),
        )

        # ── Robot ─────────────────────────────────────────────────────
        # Use a simple box as robot body if TurtleBot3 USD isn't available.
        # Dimensions match Waffle Pi: 0.265 × 0.265 × 0.089 m
        robot_spawn_cfg = sim_utils.CuboidCfg(
            size=(0.265, 0.265, 0.089),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.37),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),
            ),
        )

        self.robot = RigidObject(
            cfg=RigidObjectCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=robot_spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.05),
                ),
            )
        )

        # ── Ball ──────────────────────────────────────────────────────
        self.ball = RigidObject(
            cfg=RigidObjectCfg(
                prim_path="/World/envs/env_.*/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=self.cfg.ball_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=10.0,
                        max_angular_velocity=100.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=self.cfg.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.85, 0.0),
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(2.0, 0.0, self.cfg.ball_radius),
                ),
            )
        )

        # ── Goal posts (static) ───────────────────────────────────────
        for side, y in [('left', -GOAL_HALF_WIDTH), ('right', GOAL_HALF_WIDTH)]:
            sim_utils.spawn_cylinder(
                prim_path=f"/World/GoalPost_{side}",
                cfg=sim_utils.CylinderCfg(
                    radius=0.05,
                    height=1.0,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 1.0)),
                ),
                translation=(GOAL_X, y, 0.5),
            )

        # ── Light ─────────────────────────────────────────────────────
        sim_utils.spawn_light(
            prim_path="/World/Light",
            cfg=sim_utils.DistantLightCfg(intensity=3000.0),
        )

        # ── Register and clone ────────────────────────────────────────
        # In DirectRLEnv 2.x, assets registered in scene dict before clone
        self.scene.rigid_objects["robot"] = self.robot
        self.scene.rigid_objects["ball"]  = self.ball

        # Clone environments — copies /World/envs/env_0 to all env indices
        self.scene.clone_environments(copy_from_source=False)
        # Filter collisions so envs don't collide with each other
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ── Actions ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Convert [linear_x, angular_z] to forces/torques on the robot body.
        Since we're using a RigidObject (not Articulation), we apply
        velocity directly to the root rather than wheel joints.
        """
        lin_x = actions[:, 0].clamp(0.0,  1.0)
        ang_z = actions[:, 1].clamp(-0.6,  0.6)

        # Build velocity in robot local frame then rotate to world frame
        robot_quat = self.robot.data.root_quat_w
        _, _, yaw   = euler_xyz_from_quat(robot_quat)

        # World-frame linear velocity
        vx_world = lin_x * torch.cos(yaw)
        vy_world = lin_x * torch.sin(yaw)

        # Pack into (N, 6) [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0] = vx_world
        vel[:, 1] = vy_world
        vel[:, 5] = ang_z

        self.robot.write_root_velocity_to_sim(vel)

    def _apply_action(self):
        self.robot.write_data_to_sim()
        self.ball.write_data_to_sim()

    # ── Observations ─────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        robot_pos  = self.robot.data.root_pos_w[:, :2]     # (N, 2)
        robot_quat = self.robot.data.root_quat_w            # (N, 4)
        robot_vel  = self.robot.data.root_lin_vel_w[:, 0]  # (N,) vx
        robot_wvel = self.robot.data.root_ang_vel_w[:, 2]  # (N,) wz

        _, _, yaw   = euler_xyz_from_quat(robot_quat)       # (N,)

        ball_pos    = self.ball.data.root_pos_w[:, :2]      # (N, 2)

        # Env-local goal position
        env_origins = self.scene.env_origins[:, :2]          # (N, 2)
        goal_world  = env_origins + self._goal_pos[:2]       # (N, 2)

        # World-frame relative vectors
        ball_rel  = ball_pos  - robot_pos                    # (N, 2)
        goal_rel  = goal_world - robot_pos                   # (N, 2)
        b2g       = goal_world - ball_pos                    # (N, 2)

        cos_y = torch.cos(yaw).unsqueeze(1)
        sin_y = torch.sin(yaw).unsqueeze(1)

        def to_robot_frame(v: torch.Tensor) -> torch.Tensor:
            rx =  v[:, 0:1] * cos_y + v[:, 1:2] * sin_y
            ry = -v[:, 0:1] * sin_y + v[:, 1:2] * cos_y
            return torch.cat([rx, ry], dim=1)

        ball_local = to_robot_frame(ball_rel)
        goal_local = to_robot_frame(goal_rel)
        b2g_local  = to_robot_frame(b2g)

        ball_dist  = ball_local.norm(dim=1, keepdim=True)
        ball_angle = torch.atan2(ball_local[:, 1:2], ball_local[:, 0:1])
        goal_dist  = goal_local.norm(dim=1, keepdim=True)
        goal_angle = torch.atan2(goal_local[:, 1:2], goal_local[:, 0:1])

        obs = torch.cat([
            ball_local,                  # [0, 1]
            ball_dist,                   # [2]
            ball_angle,                  # [3]
            goal_local,                  # [4, 5]
            goal_dist,                   # [6]
            goal_angle,                  # [7]
            robot_vel.unsqueeze(1),      # [8]
            robot_wvel.unsqueeze(1),     # [9]
            b2g_local,                   # [10, 11]
        ], dim=1)   # (N, 12)

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        ball_pos    = self.ball.data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]
        goal_world  = env_origins + self._goal_pos[:2]

        ball_to_goal = (goal_world - ball_pos).norm(dim=1)
        progress     = self._prev_ball_to_goal - ball_to_goal
        self._prev_ball_to_goal = ball_to_goal.clone()

        reward  = torch.full(
            (self.num_envs,), -0.005, device=self.device)
        reward += 2.0 * progress

        # Add goal reward set by _get_dones this step
        reward += self._goal_reward
        self._goal_reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_pos    = self.ball.data.root_pos_w[:, :2]
        robot_pos   = self.robot.data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]
        goal_world  = env_origins + self._goal_pos[:2]

        # Ball in goal
        b2g          = ball_pos - goal_world
        ball_in_goal = (
            (b2g[:, 0] >= -GOAL_DEPTH) &
            (b2g[:, 0] <=  0.35) &
            (b2g[:, 1].abs() <= GOAL_HALF_WIDTH)
        )

        # Robot out of bounds
        robot_local = robot_pos - env_origins
        robot_oob   = (
            (robot_local[:, 0] < ARENA_X_MIN) |
            (robot_local[:, 0] > ARENA_X_MAX) |
            (robot_local[:, 1] < ARENA_Y_MIN) |
            (robot_local[:, 1] > ARENA_Y_MAX)
        )

        # Goal bonus applied before reset
        self._goal_reward = torch.zeros(self.num_envs, device=self.device)
        self._goal_reward[ball_in_goal] = 300.0

        terminated = ball_in_goal | robot_oob
        truncated  = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n           = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]  # (n, 3)

        # ── Robot ─────────────────────────────────────────────────────
        rx  = torch.FloatTensor(n).uniform_(*ROBOT_SPAWN_X_RANGE).to(self.device)
        ry  = torch.FloatTensor(n).uniform_(*ROBOT_SPAWN_Y_RANGE).to(self.device)
        yaw = torch.FloatTensor(n).uniform_(*ROBOT_YAW_RANGE).to(self.device)

        robot_pos       = env_origins.clone()
        robot_pos[:, 0] += rx
        robot_pos[:, 1] += ry
        robot_pos[:, 2]  = 0.05

        half_yaw    = yaw / 2.0
        robot_quat  = torch.zeros(n, 4, device=self.device)
        robot_quat[:, 0] = torch.cos(half_yaw)   # w
        robot_quat[:, 3] = torch.sin(half_yaw)   # z

        self.robot.write_root_pose_to_sim(
            torch.cat([robot_pos, robot_quat], dim=1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        # ── Ball ──────────────────────────────────────────────────────
        dist    = torch.FloatTensor(n).uniform_(*BALL_DIST_RANGE).to(self.device)
        offset  = torch.FloatTensor(n).uniform_(-0.4, 0.4).to(self.device)
        bdir    = yaw + offset

        bx = (rx + dist * torch.cos(bdir)).clamp(ARENA_X_MIN + 0.5, GOAL_X - 0.5)
        by = (ry + dist * torch.sin(bdir)).clamp(ARENA_Y_MIN + 0.5, ARENA_Y_MAX - 0.5)

        ball_pos       = env_origins.clone()
        ball_pos[:, 0] += bx
        ball_pos[:, 1] += by
        ball_pos[:, 2]  = self.cfg.ball_radius

        ball_quat       = torch.zeros(n, 4, device=self.device)
        ball_quat[:, 0] = 1.0

        self.ball.write_root_pose_to_sim(
            torch.cat([ball_pos, ball_quat], dim=1), env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        # Reset progress tracker
        goal_world = env_origins[:, :2] + self._goal_pos[:2]
        self._prev_ball_to_goal[env_ids] = (
            goal_world - ball_pos[:, :2]).norm(dim=1)
