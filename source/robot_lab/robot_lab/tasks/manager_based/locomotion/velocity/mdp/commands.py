# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import extract_yaw_from_quat, is_robot_on_terrain, rotate_vec2d_world_to_body

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.low_speed_threshold
        ).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    low_speed_threshold: float = 0.2
    """Linear xy commands below this norm are clipped to zero."""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class RailPlanarPositionCommand(CommandTerm):
    """Rail-centered planar position target for PI velocity bridging (Walking-v1)."""

    cfg: RailPlanarPositionCommandCfg

    def __init__(self, cfg: RailPlanarPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        env_origins_xy = self._env.scene.env_origins[:, :2]
        offset = torch.tensor(self.cfg.rail_offset, device=self.device)
        self.rail_center_w = env_origins_xy + offset
        self.rail_yaw_w = torch.full((self.num_envs,), self.cfg.rail_yaw, device=self.device)
        self.rail_axis_w = torch.stack(
            [torch.cos(self.rail_yaw_w), torch.sin(self.rail_yaw_w)], dim=-1
        )
        self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_yaw_w = self.rail_yaw_w.clone()

    def __str__(self) -> str:
        return (
            "RailPlanarPositionCommand:\n"
            f"\tRail offset : {self.cfg.rail_offset}\n"
            f"\tRail yaw    : {self.cfg.rail_yaw:.3f} rad\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_w

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        self._sample_targets_around_rail(env_ids_t)
        self.target_yaw_w[env_ids_t] = self.rail_yaw_w[env_ids_t]
        self._reset_velocity_bridge_integrals(env_ids_t)

    def _sample_targets_around_rail(self, env_ids_t: torch.Tensor) -> None:
        n = env_ids_t.shape[0]
        axis = self.rail_axis_w[env_ids_t]
        perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=1)

        if self.cfg.relax_spawn_min_rail_distance:
            r_lo, r_hi = self.cfg.spawn_distance_range_relaxed
            r = torch.empty(n, device=self.device).uniform_(r_lo, r_hi)
            gamma = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        else:
            r = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_distance_range)
            gamma_abs = torch.empty(n, device=self.device).uniform_(*self.cfg.spawn_angle_from_rail_axis_rad)
            gamma_sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
            gamma = gamma_sign * gamma_abs

        offset = r.unsqueeze(1) * (torch.cos(gamma).unsqueeze(1) * axis + torch.sin(gamma).unsqueeze(1) * perp)
        self.target_pos_w[env_ids_t] = self.rail_center_w[env_ids_t] + offset

    def _reset_velocity_bridge_integrals(self, env_ids_t: torch.Tensor) -> None:
        try:
            bridge = self._env.command_manager.get_term(self.cfg.velocity_bridge_command_name)
        except KeyError:
            return
        if hasattr(bridge, "reset_integrals"):
            bridge.reset_integrals(env_ids_t)

    def _update_command(self):
        robot = self._env.scene[self.cfg.asset_name]
        robot_pos_w = robot.data.root_pos_w[:, :2]
        rel_w = self.target_pos_w - robot_pos_w
        d_xy = torch.linalg.norm(rel_w, dim=-1)

        self.target_yaw_w = torch.atan2(rel_w[:, 1], rel_w[:, 0])

        reached = d_xy < self.cfg.reach_distance_m
        if bool(reached.any()):
            self._resample_command(torch.where(reached)[0])

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis

    def _debug_vis_callback(self, event):
        del event


@configclass
class RailPlanarPositionCommandCfg(CommandTermCfg):
    """Configuration for :class:`RailPlanarPositionCommand`."""

    class_type: type = RailPlanarPositionCommand

    asset_name: str = "robot"
    rail_offset: tuple[float, float] = (0.0, 0.0)
    rail_yaw: float = math.pi / 2
    spawn_distance_range: tuple[float, float] = (1.0, 3.0)
    spawn_angle_from_rail_axis_rad: tuple[float, float] = (math.pi / 4, 3.0 * math.pi / 4)
    relax_spawn_min_rail_distance: bool = False
    spawn_distance_range_relaxed: tuple[float, float] = (0.0, 3.0)
    reach_distance_m: float = 0.15
    velocity_bridge_command_name: str = "base_velocity"


class PiVelocityBridgeCommand(CommandTerm):
    """PI controller: rail position targets -> body-frame SE(2) velocity command."""

    cfg: PiVelocityBridgeCommandCfg

    def __init__(self, cfg: PiVelocityBridgeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.integral_lin_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.integral_yaw = torch.zeros(self.num_envs, device=self.device)
        self._dt = env.step_dt

    def __str__(self) -> str:
        return (
            "PiVelocityBridgeCommand:\n"
            f"\tposition_command : {self.cfg.position_command_name}\n"
            f"\tkp_lin / ki_lin  : {self.cfg.kp_lin} / {self.cfg.ki_lin}\n"
            f"\tkp_yaw / ki_yaw  : {self.cfg.kp_yaw} / {self.cfg.ki_yaw}\n"
            f"\tv_max / w_max    : {self.cfg.v_max} / {self.cfg.w_max}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def reset_integrals(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        env_ids_t = (
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids.to(device=self.device, dtype=torch.long)
        )
        self.integral_lin_b[env_ids_t] = 0.0
        self.integral_yaw[env_ids_t] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        del env_ids

    def _update_command(self):
        pos_term = self._env.command_manager.get_term(self.cfg.position_command_name)
        target_pos_w = pos_term.target_pos_w
        target_yaw_w = pos_term.target_yaw_w

        robot = self._env.scene[self.cfg.asset_name]
        robot_pos_w = robot.data.root_pos_w[:, :2]
        robot_quat_w = robot.data.root_quat_w
        robot_yaw = extract_yaw_from_quat(robot_quat_w)

        rel_w = target_pos_w - robot_pos_w
        rel_b = rotate_vec2d_world_to_body(rel_w, robot_yaw)
        d_xy = torch.linalg.norm(rel_w, dim=-1)

        approach_yaw = torch.atan2(rel_w[:, 1], rel_w[:, 0])
        yaw_goal = torch.where(d_xy <= self.cfg.yaw_activate_dist, target_yaw_w, approach_yaw)
        err_yaw = wrap_to_pi(yaw_goal - robot_yaw)

        p_lin = self.cfg.kp_lin * rel_b
        i_lin = self.cfg.ki_lin * self.integral_lin_b
        v_cmd = p_lin + i_lin
        v_norm = torch.linalg.norm(v_cmd, dim=-1, keepdim=True).clamp(min=1e-8)
        v_saturated = v_norm.squeeze(-1) > self.cfg.v_max
        v_scale = torch.clamp(self.cfg.v_max / v_norm, max=1.0)
        v_out = v_cmd * v_scale

        windup_lin = ~(v_saturated.unsqueeze(-1) & (torch.sign(v_cmd) * torch.sign(rel_b) > 0))
        self.integral_lin_b += windup_lin.float() * rel_b * self._dt

        w_raw = self.cfg.kp_yaw * err_yaw + self.cfg.ki_yaw * self.integral_yaw
        w_out = torch.clamp(w_raw, -self.cfg.w_max, self.cfg.w_max)
        w_saturated = torch.abs(w_raw) > self.cfg.w_max
        windup_yaw = ~(w_saturated & (torch.sign(w_raw) * torch.sign(err_yaw) > 0))
        self.integral_yaw += windup_yaw.float() * err_yaw * self._dt

        self.vel_command_b[:, 0] = v_out[:, 0]
        self.vel_command_b[:, 1] = v_out[:, 1]
        self.vel_command_b[:, 2] = w_out

        low_speed = d_xy < self.cfg.low_speed_threshold
        if bool(low_speed.any()):
            self.vel_command_b[low_speed, :2] = 0.0

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis

    def _debug_vis_callback(self, event):
        del event


@configclass
class PiVelocityBridgeCommandCfg(CommandTermCfg):
    """Configuration for :class:`PiVelocityBridgeCommand`."""

    class_type: type = PiVelocityBridgeCommand

    asset_name: str = "robot"
    position_command_name: str = "target_position"
    kp_lin: float = 0.5
    ki_lin: float = 0.05
    kp_yaw: float = 0.5
    ki_yaw: float = 0.02
    v_max: float = 0.5
    w_max: float = 0.5
    yaw_activate_dist: float = 0.01
    low_speed_threshold: float = 0.005
