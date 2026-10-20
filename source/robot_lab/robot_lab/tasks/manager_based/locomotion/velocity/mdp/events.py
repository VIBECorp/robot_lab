# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def assign_unique_subterrain_cell_per_env(env: ManagerBasedEnv, env_ids):
    """Assign each env to a unique terrain cell when the terrain grid is large enough."""
    del env_ids

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return

    terrain_origins = terrain.terrain_origins
    num_rows, num_cols = int(terrain_origins.shape[0]), int(terrain_origins.shape[1])
    num_cells = num_rows * num_cols
    num_envs = env.num_envs
    if num_envs > num_cells:
        return

    device = terrain_origins.device
    idx = torch.arange(num_envs, device=device, dtype=torch.long)
    rows = idx // num_cols
    cols = idx % num_cols
    terrain.env_origins[:num_envs] = terrain_origins[rows, cols]
    terrain.terrain_levels = rows.clone()
    terrain.terrain_types = cols.clone()


def align_env_root_xforms_with_terrain_origins(env: ManagerBasedEnv, env_ids):
    """Move ``/World/envs/env_*`` Xforms to terrain origins before simulation starts."""
    del env_ids

    terrain = env.scene.terrain
    if terrain is None or not getattr(terrain.cfg, "use_terrain_origins", False):
        return

    origins = terrain.env_origins
    if origins is None:
        return

    from isaaclab.sim.utils.transforms import standardize_xform_ops

    stage = env.scene.stage
    prim_paths = env.scene.env_prim_paths
    num_envs = min(len(prim_paths), origins.shape[0])
    for env_idx in range(num_envs):
        prim = stage.GetPrimAtPath(prim_paths[env_idx])
        if not prim.IsValid() or not prim.IsActive():
            continue
        x, y, z = float(origins[env_idx, 0]), float(origins[env_idx, 1]), float(origins[env_idx, 2])
        standardize_xform_ops(prim, translation=(x, y, z))


def ensure_rail_physics_colliders(
    env: ManagerBasedEnv,
    env_ids,
    rail_child_name: str = "Rail",
):
    """Add PhysX collision APIs to rail Mesh/Cube prims before simulation starts.

    ``UsdFileCfg.collision_props`` can only update existing collision APIs. The
    rail USD may contain plain Mesh/Cube prims, so we explicitly define the
    collision schema here to make robot-rail contact active.
    """
    del env_ids

    from pxr import Usd, UsdPhysics

    from isaaclab.sim import schemas
    from isaaclab.sim.schemas import schemas_cfg

    stage = env.scene.stage
    kinematic_cfg = schemas_cfg.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    collision_cfg = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    convex_cfg = schemas_cfg.ConvexHullPropertiesCfg()

    def subtree_has_rigid_body(root) -> bool:
        for prim in Usd.PrimRange(root):
            if UsdPhysics.RigidBodyAPI(prim):
                return True
        return False

    for env_path in env.scene.env_prim_paths:
        rail_path = f"{env_path}/{rail_child_name}"
        rail_prim = stage.GetPrimAtPath(rail_path)
        if not rail_prim.IsValid():
            continue

        if not subtree_has_rigid_body(rail_prim):
            schemas.define_rigid_body_properties(rail_path, kinematic_cfg, stage=stage)

        for prim in Usd.PrimRange(rail_prim):
            if prim.IsInstance():
                continue
            prim_type = prim.GetTypeName()
            prim_path = str(prim.GetPath())
            if prim_type == "Mesh":
                schemas.define_collision_properties(prim_path, collision_cfg, stage=stage)
                schemas.define_mesh_collision_properties(prim_path, convex_cfg, stage=stage)
            elif prim_type == "Cube":
                schemas.define_collision_properties(prim_path, collision_cfg, stage=stage)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization,
            or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Separate pit and non-pit environments
    # Check which environments are assigned to pit terrain (not random reset)
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # Reset pit environments to default state (no random perturbations)
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # Reset non-pit environments with random perturbations
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)


def reset_root_state_around_rail(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    rail_offset: tuple[float, float] = (0.0, 0.0),
    spawn_distance_range: tuple[float, float] = (1.0, 3.0),
    spawn_angle_from_rail_axis_rad: tuple[float, float] = (math.pi / 4, 3.0 * math.pi / 4),
    yaw_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    command_name: str = "target_position",
    initial_height: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    relax_spawn_min_rail_distance: bool | None = None,
    spawn_distance_range_relaxed: tuple[float, float] | None = None,
):
    """Reset robot root around the rail center using rail-axis coordinates."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)

    rail_offset_t = torch.tensor(rail_offset, device=env.device)
    rail_xy = env.scene.env_origins[env_ids, :2] + rail_offset_t

    cmd_term = env.command_manager.get_term(command_name)
    cfg = cmd_term.cfg
    if relax_spawn_min_rail_distance is None:
        relax_spawn_min_rail_distance = bool(getattr(cfg, "relax_spawn_min_rail_distance", False))
    if spawn_distance_range_relaxed is None:
        spawn_distance_range_relaxed = tuple(getattr(cfg, "spawn_distance_range_relaxed", (0.0, 3.0)))

    rail_yaw = float(cfg.rail_yaw)
    axis = torch.tensor([math.cos(rail_yaw), math.sin(rail_yaw)], device=env.device, dtype=torch.float32)
    axis = axis.unsqueeze(0).expand(n, 2)
    perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=1)

    if relax_spawn_min_rail_distance:
        r_lo, r_hi = spawn_distance_range_relaxed
        r = torch.empty(n, device=env.device).uniform_(r_lo, r_hi)
        gamma = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    else:
        r = torch.empty(n, device=env.device).uniform_(*spawn_distance_range)
        gamma_abs = torch.empty(n, device=env.device).uniform_(*spawn_angle_from_rail_axis_rad)
        gamma_sign = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        gamma = gamma_sign * gamma_abs

    offset = r.unsqueeze(1) * (torch.cos(gamma).unsqueeze(1) * axis + torch.sin(gamma).unsqueeze(1) * perp)
    root_states = asset.data.default_root_state[env_ids].clone()

    positions = torch.zeros(n, 3, device=env.device)
    positions[:, 0] = rail_xy[:, 0] + offset[:, 0]
    positions[:, 1] = rail_xy[:, 1] + offset[:, 1]
    positions[:, 2] = env.scene.env_origins[env_ids, 2] + root_states[:, 2] + initial_height

    y_lo, y_hi = yaw_range
    yaw = torch.rand(n, device=env.device) * (y_hi - y_lo) + y_lo
    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device),
        torch.zeros(n, device=env.device),
        yaw,
    )
    velocities = torch.zeros(n, 6, device=env.device)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
