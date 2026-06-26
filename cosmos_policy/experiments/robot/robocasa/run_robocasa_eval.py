# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_robocasa_eval.py

Evaluates a trained policy in a RoboCasa simulation benchmark task suite.

Parallel inference:
    (Only applicable when doing best-of-N search with N GPUs)
    To enable parallel inference across 8 GPUs, use:
        --use_parallel_inference True
        --num_queries_best_of_n 8
        --available_gpus "0,1,2,3,4,5,6,7"

Usage examples:
    # *** Main checkpoint: 67.1% avg success rate ***
    #   Replace `task_suite_name` with one of {libero_spatial, libero_object, libero_goal, libero_10}
    #   Replace `seed` with one of {195, 196, 197}
    #   Replace `run_id_note` with a unique identifier for the run
    uv run -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --num_wrist_images 1 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 32 \
        --num_open_loop_steps 16 \
        --task_name TurnOffMicrowave \
        --num_trials_per_task 50 \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
        --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False

"""

import ast
import multiprocessing as mp
import json
import os
import pickle
import secrets
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import draccus
import h5py
import numpy as np
import robosuite
import torch
import wandb
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

from cosmos_policy.experiments.robot.cosmos_utils import (
    ACTION_DIM,
    WorkerPoolManager,
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: curr secondary img, 5: action, 6: future proprio, 7: future wrist img, 8: future primary img, 9: future secondary img, 10: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 4
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 6, 9

# Path to fixed controller configs
CONTROLLER_CONFIGS_PATH: str = "cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"

# Define max steps for each RoboCasa task (based on horizons in dataset_registry)
TASK_MAX_STEPS = {
    # Pick and place tasks
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door tasks
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer tasks
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove tasks
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink tasks
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee tasks
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave tasks
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robocasa"                                              # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Inference config name
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include third-person ("primary") image in input
    num_third_person_images: int = 2                                     # Number of third-person images to include in input (RoboCasa: 1 for left, 1 for right)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (RoboCasa: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis (RoboCasa: True because environment returns images upside down)
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    dataset_stats_path: str = ""                                         # Path to dataset statistics file for action unnormalization and proprio normalization
    t5_text_embeddings_path: str = ""                                    # Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 32                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "last")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    search_depth_value_aggregation_scheme: str = "use_last_value"        # How to aggregate value predictions across search depth (options: use_last_value, average)
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    compute_value_prediction_gap: bool = True                            # Whether to compute the value prediction gap metric: compares value predicted on world-model future states vs. on the actual reached state at the next inference step (serial inference, search_depth==1 only)

    use_simulator_for_planning: bool = False                             # Oracle/diagnostic planner: instead of predicting the future state with the world model, roll out each candidate action chunk in the simulator (saving/restoring sim state) and evaluate the value function on the REAL reached state. Removes world-model error from planning. Requires serial inference and search_depth==1.
    measure_value_agreement: bool = False                                # Diagnostic: for each candidate chunk, evaluate the value function on BOTH the world-model future state and the simulator-reached future state, then record which seed each method ranks highest and whether they agree. Selection/execution still use the world-model value (normal policy behavior). Requires serial inference, search_depth==1, ar_future_prediction and ar_value_prediction; incompatible with use_simulator_for_planning.

    use_planning_gating: bool = False                                    # Adaptive gating: at each requery, first run a single cheap rollout with the base model (action + predicted future state + value of that state). Only if the predicted improvement `V(s'_pred) - V(s_curr) < planning_gating_threshold` do we engage the full best-of-N planning (world model + value, planning checkpoint); otherwise just execute the base model's default chunk. Requires serial inference and search_depth==1.
    planning_gating_threshold: float = 0.0                               # Threshold on the predicted advantage `V(s'_pred) - V(s_curr)`. If the advantage is below this, planning is triggered (the default chunk is not improving the state enough). Higher threshold => plan more often.
    # Denoising-step counts for the NON-planning (gating default) path. When None, fall back to the
    # corresponding planning value below, so existing scripts are unchanged. Set these lower to make the
    # cheap "skip planning" path actually cheap (it generates the default chunk + V(s_curr) probe).
    num_denoising_steps_action_nonplanning: Optional[int] = None         # Action denoising steps for the gating default chunk. This is a joint action+future+value pass, so it also governs V(s'_pred). Falls back to num_denoising_steps_action.
    num_denoising_steps_value_nonplanning: Optional[int] = None          # Value denoising steps for the V(s_curr) gating probe. Falls back to num_denoising_steps_value. (V(s'_pred) is produced in the joint action pass above, not here.)

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # RoboCasa-specific parameters
    #################################################################################################################
    task_name: str = "PnPCounterToCab"                                   # Task name (must be in SINGLE_STAGE_TASK_DATASETS or MULTI_STAGE_TASK_DATASETS)
    num_trials_per_task: int = 50                                        # Number of rollouts per task
    env_img_res: int = 224                                               # Resolution for rendering environment images
    robots: str = "PandaMobile"                                          # Robot type for RoboCasa (PandaMobile is alias for PandaOmron)
    controllers: str = "OSC_POSE"                                        # Controller type (OSC_POSE = Operational Space Control with 6-DOF end-effector pose)
    obj_instance_split: str = "B"                                        # Object instance split - "B" = held-out test objects
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"       # Layout and style IDs - 5 test scenes
    randomize_cameras: bool = False                                      # Whether to randomize camera positions

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 195                                                      # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Fixed-initial-condition repeat mode (study policy stochasticity in isolation)
    #################################################################################################################
    repeat_fixed_init: bool = False                                      # If True, run the SAME initial condition (object, placement, scene) repeatedly and vary ONLY the policy action-generation seed across repeats. Lets you measure how much the policy's own sampling stochasticity changes the rollout, holding the environment fixed. Requires serial inference.
    num_fixed_init_repeats: int = 10                                     # (Only applicable if repeat_fixed_init) Number of times to repeat the fixed initial condition
    fixed_init_scene_idx: int = 0                                        # (Only applicable if repeat_fixed_init) Index into `layout_and_style_ids` selecting which test scene to use for the fixed initial condition
    policy_seed: Optional[int] = None                                    # (Only applicable if repeat_fixed_init) Base seed for the policy action generation. Repeat `r` uses generation seed `policy_seed + r` (so each repeat samples differently); the environment seed stays fixed. When None, defaults to `seed`.

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save policy rollouts for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving

    # fmt: on


def validate_config(cfg: PolicyEvalConfig) -> None:
    """Validate that the evaluation configuration is valid."""
    # Check that the task name is valid
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if cfg.task_name not in all_tasks:
        raise ValueError(
            f"Task name '{cfg.task_name}' not found in RoboCasa suite. Available tasks: {list(all_tasks.keys())}"
        )

    # Check that num_third_person_images is 2 (1 for left, 1 for right)
    assert cfg.num_third_person_images == 2, (
        f"Expecting `num_third_person_images` to be 2 (1 for left agentview, 1 for right agentview), "
        f"but got `num_third_person_images={cfg.num_third_person_images}`"
    )

    # Check that the dataset stats path is provided if action unnormalization or proprio normalization is enabled
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and cfg.dataset_stats_path == "":
        raise ValueError(
            "Must provide `dataset_stats_path` when `unnormalize_actions=True` or `normalize_proprio=True`"
        )

    # Check parallel inference configuration
    if cfg.use_parallel_inference and cfg.num_queries_best_of_n <= 1:
        raise ValueError("Parallel inference is enabled but `num_queries_best_of_n <= 1`!")

    # Validate the fixed-initial-condition repeat mode.
    if cfg.repeat_fixed_init:
        if cfg.use_parallel_inference:
            raise ValueError(
                "`repeat_fixed_init=True` is not supported with `use_parallel_inference=True`. "
                "The per-repeat policy seed override is applied in the serial inference path; parallel "
                "workers load their own config copy at startup and would not pick it up. Run serially."
            )
        if cfg.num_fixed_init_repeats < 1:
            raise ValueError(
                f"`repeat_fixed_init=True` requires `num_fixed_init_repeats >= 1` (got {cfg.num_fixed_init_repeats})."
            )
        all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids) if cfg.layout_and_style_ids else []
        if all_layout_style_ids and not (0 <= cfg.fixed_init_scene_idx < len(all_layout_style_ids)):
            raise ValueError(
                f"`fixed_init_scene_idx={cfg.fixed_init_scene_idx}` is out of range for "
                f"`layout_and_style_ids` with {len(all_layout_style_ids)} scenes."
            )

    # Warn if the value prediction gap metric would be horizon-misaligned.
    # The world model predicts the state `chunk_size` steps ahead, so the real reached state only matches
    # that horizon (and is produced by the predicted chunk) when `num_open_loop_steps == chunk_size`.
    # When they differ, the metric is gated off in run_episode to avoid reporting a misleading gap.
    if (
        cfg.compute_value_prediction_gap
        and not cfg.use_parallel_inference
        and cfg.search_depth == 1
        and cfg.num_open_loop_steps != cfg.chunk_size
    ):
        print(
            f"WARNING: `compute_value_prediction_gap=True` but `num_open_loop_steps "
            f"({cfg.num_open_loop_steps}) != chunk_size ({cfg.chunk_size})`. The world model predicts the "
            f"state `chunk_size` steps ahead, so the real reached state at the next requery is "
            f"horizon-misaligned. The value prediction gap metric will be DISABLED for this run. "
            f"Set `num_open_loop_steps == chunk_size` to enable it."
        )

    # Validate the simulator-based (oracle) planner configuration.
    if cfg.use_simulator_for_planning:
        if cfg.use_parallel_inference:
            raise ValueError(
                "`use_simulator_for_planning=True` is not supported with `use_parallel_inference=True`. "
                "The simulator rollout reuses the single eval environment, so planning must be serial."
            )
        if cfg.search_depth != 1:
            raise ValueError(
                "`use_simulator_for_planning=True` only supports `search_depth == 1` "
                f"(got search_depth={cfg.search_depth})."
            )
        if cfg.ar_qvalue_prediction:
            raise ValueError(
                "`use_simulator_for_planning=True` is incompatible with `ar_qvalue_prediction=True` "
                "(Q-value conditions on (state, action), not on a future state)."
            )
        if cfg.num_open_loop_steps != cfg.chunk_size:
            print(
                f"WARNING: `use_simulator_for_planning=True` with `num_open_loop_steps "
                f"({cfg.num_open_loop_steps}) != chunk_size ({cfg.chunk_size})`. The value function is "
                f"trained to evaluate the state `chunk_size` steps ahead, so the simulator rollout uses the "
                f"full chunk; consider setting `num_open_loop_steps == chunk_size` for horizon alignment."
            )

    # Validate the value-agreement diagnostic configuration.
    if cfg.measure_value_agreement:
        if cfg.use_simulator_for_planning:
            raise ValueError(
                "`measure_value_agreement=True` is incompatible with `use_simulator_for_planning=True`. "
                "The agreement metric compares the world-model value against the simulator value, so the "
                "world-model planning path must be active."
            )
        if cfg.use_parallel_inference:
            raise ValueError(
                "`measure_value_agreement=True` is not supported with `use_parallel_inference=True` "
                "(the simulator rollout reuses the single eval environment, so planning must be serial)."
            )
        if cfg.search_depth != 1:
            raise ValueError(
                f"`measure_value_agreement=True` only supports `search_depth == 1` (got {cfg.search_depth})."
            )
        if not (cfg.ar_future_prediction and cfg.ar_value_prediction):
            raise ValueError(
                "`measure_value_agreement=True` requires `ar_future_prediction=True` and "
                "`ar_value_prediction=True` so a world-model value is computed to compare against."
            )

    # Validate the adaptive planning-gating configuration.
    if cfg.use_planning_gating:
        if cfg.use_parallel_inference:
            raise ValueError(
                "`use_planning_gating=True` is not supported with `use_parallel_inference=True` "
                "(the gating decision runs serially before deciding whether to fan out)."
            )
        if cfg.search_depth != 1:
            raise ValueError(
                f"`use_planning_gating=True` only supports `search_depth == 1` (got {cfg.search_depth})."
            )
        if cfg.measure_value_agreement:
            raise ValueError(
                "`use_planning_gating=True` is incompatible with `measure_value_agreement=True` "
                "(the agreement metric expects every seed to run the full WM+sim value path)."
            )
        if cfg.num_queries_best_of_n <= 1:
            print(
                "WARNING: `use_planning_gating=True` with `num_queries_best_of_n <= 1`. Gating only "
                "decides whether to run best-of-N; with N<=1 there is nothing to gate."
            )


def prepare_observation(obs, flip_images: bool = False):
    """Prepare observations from environment for policy input.

    Returns:
        dict: Observation dictionary with keys:
            - "primary_image": Left third-person image (primary camera)
            - "secondary_image": Right third-person image
            - "wrist_image": Eye-in-hand wrist camera image
            - "proprio": Proprioceptive state (eef pose + gripper state)
    """
    # Extract images based on available cameras
    primary_img = None
    secondary_img = None
    wrist_img = None
    # RoboCasa has multiple camera views: left third-person, right third-person, wrist camera
    # We call the left third-person image the primary image and the right third-person image the secondary image
    if "robot0_agentview_left_image" in obs:
        img = obs["robot0_agentview_left_image"]
        if flip_images:
            img = np.flipud(img)
        primary_img = img
    if "robot0_agentview_right_image" in obs:
        img = obs["robot0_agentview_right_image"]
        if flip_images:
            img = np.flipud(img)
        secondary_img = img
    if "robot0_eye_in_hand_image" in obs:
        img = obs["robot0_eye_in_hand_image"]
        if flip_images:
            img = np.flipud(img)
        wrist_img = img
    # Extract proprioceptive state
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    # Prepare observations dict
    observation = {
        "primary_image": primary_img,
        "secondary_image": secondary_img,
        "wrist_image": wrist_img,
        "proprio": proprio,
    }
    return observation


def create_robocasa_env(cfg: PolicyEvalConfig, seed=None, episode_idx=None):
    """Create a RoboCasa environment.

    Args:
        cfg: Configuration object
        seed: Random seed for environment
        episode_idx: Episode index for deterministic scene selection (if None, uses all scenes)
    """
    # Parse layout_and_style_ids
    if cfg.layout_and_style_ids:
        all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids)
        # Deterministically select one scene based on episode index
        # Episodes 0-9 use scene 0, episodes 10-19 use scene 1, etc.
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_layout_style_ids)
            layout_and_style_ids = (all_layout_style_ids[scene_index],)
        else:
            # If no episode index provided, use all scenes (random selection by env)
            layout_and_style_ids = all_layout_style_ids
    else:
        layout_and_style_ids = None

    with open(CONTROLLER_CONFIGS_PATH, "rb") as pickle_file:
        controller_configs = pickle.load(pickle_file)
    # Create environment
    # We use the same args used in the official RoboCasa evals (robocasa/utils/eval_utils.py)
    env_kwargs = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


def snapshot_env_state(env):
    """Capture the full simulator + bookkeeping state needed to restore the env exactly after a
    planning rollout.

    `env.sim.get_state()` only captures the MuJoCo physics state (qpos/qvel/time); `env.step` also
    mutates robosuite-level counters (`timestep`, `cur_time`, `done`) and the observable cache. We
    snapshot all of them so that simulating candidate action chunks does not perturb the real episode.
    """
    return {
        "sim_state": env.sim.get_state().flatten(),
        "timestep": env.timestep,
        "cur_time": env.cur_time,
        "done": env.done,
    }


def restore_env_state(env, snapshot):
    """Restore the env to a previously captured snapshot (see `snapshot_env_state`).

    After resetting the MuJoCo state we call `sim.forward()` to re-synchronize derived quantities,
    restore the robosuite counters, clear the observable cache, and re-anchor the controller goals to
    the restored end-effector pose so subsequent stepping behaves as if the rollout never happened.
    """
    env.sim.set_state_from_flattened(snapshot["sim_state"])
    env.sim.forward()
    env.timestep = snapshot["timestep"]
    env.cur_time = snapshot["cur_time"]
    env.done = snapshot["done"]
    env._obs_cache = {}
    # Re-anchor each robot's controller goal to the restored state (OSC controllers hold an internal
    # goal pose that would otherwise reflect the end of the simulated rollout).
    for robot in env.robots:
        composite_controller = getattr(robot, "composite_controller", None)
        if composite_controller is not None:
            composite_controller.update_state()
            composite_controller.reset()


def simulate_action_chunk_in_env(env, action_chunk, cfg):
    """Roll a candidate action chunk out in the simulator and return the observation at the reached
    state, restoring the env to its pre-rollout state afterwards.

    Used by the simulator-based (oracle) planner (`cfg.use_simulator_for_planning`): instead of
    predicting the future state with the world model, we step the real simulator so the value
    function can be evaluated on the true reached state. MuJoCo transitions are deterministic, so this
    does not affect the actual episode.
    """
    snapshot = snapshot_env_state(env)
    obs = None
    try:
        for action in action_chunk:
            action = np.asarray(action)
            # RoboCasa: policy emits 7-dim manipulation actions, but env expects 12-dim (7 + 5 mobile
            # base). Append [0, 0, 0, 0, -1] for the (unused) mobile base, matching the real-step path.
            if action.shape[-1] == 7 and env.action_dim == 12:
                action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
            obs, _, _, _ = env.step(action)
    finally:
        restore_env_state(env, snapshot)
    return prepare_observation(obs, cfg.flip_images)


def _average_ranks(values):
    """Return average ranks (ties share the mean of their rank positions). Used for Spearman."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_corr(a, b):
    """Spearman rank correlation between two equal-length sequences (no SciPy dependency).

    Returns NaN if there are fewer than 2 points or either side is constant.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_value_of_current_state(cfg, model, action_dict, seed, num_denoising_steps_value=None):
    """Evaluate the value function on the CURRENT state, V(s_curr).

    The value function reads the future-state latent frames, so to score the current state we copy the
    current-state frames into the future-state slot (same trick as the value-gap metric) and run the
    value head. Used by the planning-gating decision; uses the base model (consistent with the base
    model's joint-predicted V(s'_pred) it is compared against).

    num_denoising_steps_value: override for the number of value denoising steps; falls back to
    cfg.num_denoising_steps_value when None.
    """
    if num_denoising_steps_value is None:
        num_denoising_steps_value = cfg.num_denoising_steps_value
    curr_state_latent = action_dict["generated_latent"].clone()
    curr_state_latent[:, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1] = action_dict[
        "generated_latent"
    ][:, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1]
    value_return_dict = get_value_prediction(
        cfg,
        model=model,
        data_batch=action_dict["data_batch"],
        future_state_samples_list=[curr_state_latent],
        seed=seed,
        randomize_seed=cfg.randomize_seed,
        num_denoising_steps_value=num_denoising_steps_value,
        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
    )
    return value_return_dict["value_prediction"]


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    episode_idx: int,
    log_file=None,
    policy_seed: Optional[int] = None,
):
    """Run a single evaluation episode.

    policy_seed: when provided, overrides the seed used for ALL policy action/value generation in this
    episode (every `cfg.seed`-derived generation seed below). The environment seed is unaffected. This
    is how `repeat_fixed_init` mode varies only the policy's sampling stochasticity while holding the
    initial conditions fixed. Implemented by locally shadowing `cfg` with a seed-overridden copy so the
    rest of this function is unchanged.
    """
    if policy_seed is not None and policy_seed != cfg.seed:
        cfg = replace(cfg, seed=policy_seed)
    # Wait for objects to stabilize
    NUM_STEPS_WAIT = 10
    for _ in range(NUM_STEPS_WAIT):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)
    # Get max steps for this task
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    # Important variables
    success = False
    episode_length = 0
    action_queue = deque()
    # Containers for episode replay images for saving videos
    replay_primary_images = []  # Left third-person camera
    replay_secondary_images = []  # Right third-person camera
    replay_wrist_images = []  # Wrist camera
    # Containers for episode data collection
    if cfg.data_collection:
        collected_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "success": [],
        }
        primary_images_list = []
        secondary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []
    # Best-of-N search variables
    future_image_predictions_list = []
    planning_stats = []
    # Value prediction gap metric variables
    # `pending_value_metric` holds the previous inference step's value prediction (conditioned on the
    # world-model-predicted future state) plus the latents needed to re-evaluate the value on the actual
    # reached state once we observe it at the next inference step.
    pending_value_metric = None
    value_gap_stats = []
    # Value-function agreement metric (world model vs. simulator), one record per inference step
    value_agreement_stats = []
    # Adaptive planning-gating metric, one record per requery when use_planning_gating is on
    gating_stats = []
    # Main episode loop
    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        # Store replay images for video saving
        replay_primary_images.append(observation["primary_image"])
        replay_secondary_images.append(observation["secondary_image"])
        replay_wrist_images.append(observation["wrist_image"])
        # Collect data if enabled
        if cfg.data_collection:
            primary_images_list.append(observation["primary_image"])
            secondary_images_list.append(observation["secondary_image"])
            wrist_images_list.append(observation["wrist_image"])
            proprio_list.append(observation["proprio"])
        # Query policy for new action chunk
        if len(action_queue) == 0:
            # Latent encoding of the actual reached state at this inference step (current-state frames).
            # Captured from the first serial query and used to resolve the previous step's value gap metric.
            real_current_state_latent = None
            # Use parallel inference if enabled
            if cfg.use_parallel_inference and worker_pool and worker_pool.initialized:
                # Query model in parallel
                start_time = time.time()
                query_results = query_model_parallel(
                    cfg,
                    observation,
                    task_description,
                    worker_pool,
                    timeout=cfg.parallel_timeout,
                )
                total_query_time = time.time() - start_time

                log_message(
                    f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                )

            else:
                # Serial execution
                query_results = []
                # Disable planning during the episode first steps
                num_queries_best_of_n = cfg.num_queries_best_of_n

                # ===== Adaptive planning gating =====
                # Decide whether to engage best-of-N planning at all. Run a single cheap rollout with the
                # base model (the WAVM: jointly predicts the action chunk, the future state, and the value
                # of that future state). Compare the predicted improvement V(s'_pred) - V(s_curr) against a
                # threshold; only plan when the default chunk is not improving the state enough. The
                # best-of-N path below uses the planning checkpoint; this default path uses the base model.
                do_planning = True
                if cfg.use_planning_gating:
                    # Non-planning denoising-step counts (fall back to the planning values when unset).
                    nonplanning_action_steps = (
                        cfg.num_denoising_steps_action_nonplanning
                        if cfg.num_denoising_steps_action_nonplanning is not None
                        else cfg.num_denoising_steps_action
                    )
                    nonplanning_value_steps = (
                        cfg.num_denoising_steps_value_nonplanning
                        if cfg.num_denoising_steps_value_nonplanning is not None
                        else cfg.num_denoising_steps_value
                    )
                    gating_action_dict = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        observation,
                        task_description,
                        seed=cfg.seed,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_action=nonplanning_action_steps,
                        generate_future_state_and_value_in_parallel=True,  # base WAVM: joint action+future+value
                    )
                    value_pred_future = gating_action_dict["value_prediction"]  # V(s'_pred)
                    value_curr = compute_value_of_current_state(
                        cfg, model, gating_action_dict, seed=cfg.seed, num_denoising_steps_value=nonplanning_value_steps
                    )  # V(s_curr)
                    gating_advantage = value_pred_future - value_curr
                    do_planning = gating_advantage < cfg.planning_gating_threshold
                    gating_stats.append({
                        "timestep": t,
                        "value_curr": float(value_curr),
                        "value_pred_future": float(value_pred_future),
                        "advantage": float(gating_advantage),
                        "did_plan": bool(do_planning),
                    })
                    log_message(
                        f"t={t}: [Gating] V(s_curr)={value_curr:.4f}, V(s'_pred)={value_pred_future:.4f}, "
                        f"advantage={gating_advantage:.4f} (threshold={cfg.planning_gating_threshold}) -> "
                        f"{'PLAN (best-of-N)' if do_planning else 'SKIP (use base chunk)'}",
                        log_file,
                    )
                    if not do_planning:
                        # Execute the base model's default chunk; skip best-of-N this requery.
                        query_results = [{
                            "actions": gating_action_dict["actions"],
                            "future_image_predictions": gating_action_dict["future_image_predictions"],
                            "value_prediction": gating_action_dict["value_prediction"],
                            "all_value_predictions": gating_action_dict["value_prediction"],
                            "data_batch": gating_action_dict["data_batch"],
                            "value_cond_latents": [gating_action_dict["generated_latent"]],
                            "actions_by_depth": [gating_action_dict["actions"]],
                            "value_predictions_by_depth": [gating_action_dict["value_prediction"]],
                            "future_image_predictions_by_depth": [gating_action_dict["future_image_predictions"]],
                        }]

                # Run best-of-N only when gating is off or the gate decided to plan.
                for query_idx in range(num_queries_best_of_n if do_planning else 0):
                    actions_by_depth = []  # Action chunks across all depths of the search
                    future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                    value_predictions_by_depth = []  # Value predictions across all depths of the search
                    return_dict = {}
                    # Query model to get action
                    start_time = time.time()
                    if cfg.use_planning_gating and query_idx == 0:
                        # Reuse the gating rollout's action chunk as candidate 0 (deterministic, same
                        # seed as this query would use), saving one base action call. Its future state
                        # and value are still (re)computed below through the same planning path as the
                        # other candidates, so the selection values stay comparable.
                        action_return_dict = gating_action_dict
                    else:
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cfg.ar_future_prediction
                                or cfg.ar_value_prediction
                                or cfg.ar_qvalue_prediction
                                or cfg.use_simulator_for_planning
                            ),
                        )
                    query_time = time.time() - start_time
                    log_message(
                        f"Query {query_idx + 1}/{num_queries_best_of_n}: Action query time = {query_time:.3f} sec",
                        log_file,
                    )
                    return_dict["actions"] = action_return_dict["actions"]
                    actions_by_depth.append(return_dict["actions"])
                    # Keep the data batch (holds this step's proprio + latent indices) so the value gap
                    # metric can re-evaluate the value with the exact same current-state context as the
                    # original (world-model) value prediction.
                    return_dict["data_batch"] = action_return_dict["data_batch"]

                    # Capture the actual reached state (current-state latent frames). All queries encode the
                    # same observation, so we only need it from the first query.
                    if query_idx == 0:
                        real_current_state_latent = action_return_dict["generated_latent"].clone()

                    if cfg.use_simulator_for_planning:
                        # ===== Simulator-based (oracle) planning =====
                        # Instead of predicting the future state with the world model, roll the
                        # candidate chunk out in the simulator to get the TRUE reached state, encode
                        # it, and splice it into the value-conditioning latent. This removes
                        # world-model error from planning; the value function is then evaluated on
                        # the real future state. (Diagnostic/upper-bound planner: it queries the sim.)
                        start_time = time.time()
                        sim_future_obs = simulate_action_chunk_in_env(env, action_return_dict["actions"], cfg)
                        # Encode the reached observation into latent frames (reuses the policy's
                        # obs->latent encoding; we only need the current-state frames of the result).
                        sim_future_action_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            sim_future_obs,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=False,
                        )
                        # Build the value-conditioning latent: take this candidate's latent (correct
                        # current-state + action frames) and overwrite its future-state frames with
                        # the encoded reached state.
                        sim_future_state_latent = action_return_dict["generated_latent"].clone()
                        sim_future_state_latent[
                            :, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1
                        ] = sim_future_action_dict["generated_latent"][
                            :, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1
                        ]
                        future_state_samples_list = [sim_future_state_latent]
                        # For visualization, the "predicted" future images are the real reached obs.
                        return_dict["future_image_predictions"] = {
                            "future_wrist_image": sim_future_obs["wrist_image"],
                            "future_image": sim_future_obs["primary_image"],
                            "future_image2": sim_future_obs["secondary_image"],
                        }
                        future_image_predictions_by_depth.append(return_dict["future_image_predictions"])
                        # Evaluate the value function on the simulated (real) future state.
                        value_return_dict = get_value_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=future_state_samples_list,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        return_dict["all_value_predictions"] = value_return_dict["all_value_predictions"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        # Save value-conditioning latents for parity with the world-model path.
                        return_dict["value_cond_latents"] = future_state_samples_list
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Simulator rollout + value "
                            f"query time = {query_time:.3f} sec, value = {return_dict['value_prediction']:.4f}",
                            log_file,
                        )

                    elif cfg.ar_future_prediction:
                        # Autoregressively query model to get future state prediction
                        start_time = time.time()
                        future_state_return_dict = get_future_state_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            generated_latent_with_action=action_return_dict["generated_latent"],
                            orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                            future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                            future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                "future_wrist_image_latent_idx"
                            ],
                            future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                "future_wrist_image2_latent_idx"
                            ],
                            future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                            future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                            use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                            num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                            future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Future state prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                        future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                    else:
                        return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                    if cfg.use_simulator_for_planning:
                        # Value was already computed in the simulator branch above.
                        pass
                    elif cfg.ar_value_prediction:
                        # Autoregressively query model to get value prediction
                        start_time = time.time()
                        value_return_dict = get_value_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        return_dict["all_value_predictions"] = value_return_dict["all_value_predictions"]
                        value_std = torch.tensor(value_return_dict["all_value_predictions"]).std()
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        # Save the value-conditioning latents (containing the world-model-predicted future state)
                        # so we can later re-evaluate the value on the actual reached state (value gap metric).
                        return_dict["value_cond_latents"] = future_state_return_dict["future_state_samples_list"]
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}, std: {value_std:.4f}",
                            log_file,
                        )
                    elif cfg.ar_qvalue_prediction:
                        # Autoregressively query model to get Q-value prediction
                        start_time = time.time()
                        value_return_dict = get_qvalue_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            action_sample=action_return_dict["generated_latent"],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Value prediction query time = {query_time:.3f} sec",
                            log_file,
                        )
                        return_dict["value_prediction"] = value_return_dict["value_prediction"]
                        return_dict["all_value_predictions"] = value_return_dict["all_value_predictions"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        # Q-value prediction conditions on (state, action), not on a future state, so the
                        # value gap metric (which swaps the future state) does not apply here.
                        return_dict["value_cond_latents"] = None
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                            log_file,
                        )
                    else:
                        return_dict["value_prediction"] = action_return_dict["value_prediction"]
                        return_dict["all_value_predictions"] = action_return_dict["value_prediction"]
                        value_predictions_by_depth.append(return_dict["value_prediction"])
                        # Future state and value were generated jointly inside get_action; the generated
                        # latent already contains the world-model-predicted future state frames.
                        return_dict["value_cond_latents"] = [action_return_dict["generated_latent"]]

                    if cfg.measure_value_agreement:
                        # ===== Value-function agreement: WM future state vs. simulator future state =====
                        # The world-model value above (`value_prediction`) drives the actual best-of-N
                        # selection. Here we ALSO evaluate the value function on the TRUE simulator-reached
                        # state for this same candidate chunk, so we can later compare which seed each
                        # method ranks highest. This does not change which action is executed.
                        start_time = time.time()
                        return_dict["value_prediction_wm"] = return_dict["value_prediction"]
                        sim_future_obs = simulate_action_chunk_in_env(env, action_return_dict["actions"], cfg)
                        sim_future_action_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            sim_future_obs,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=False,
                        )
                        sim_future_state_latent = action_return_dict["generated_latent"].clone()
                        sim_future_state_latent[
                            :, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1
                        ] = sim_future_action_dict["generated_latent"][
                            :, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1
                        ]
                        sim_value_return_dict = get_value_prediction(
                            cfg,
                            model=planning_model if planning_model is not None else model,
                            data_batch=action_return_dict["data_batch"],
                            future_state_samples_list=[sim_future_state_latent],
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_value=cfg.num_denoising_steps_value,
                            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                        )
                        return_dict["value_prediction_sim"] = sim_value_return_dict["value_prediction"]
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries_best_of_n}: [Agreement] WM value = "
                            f"{return_dict['value_prediction_wm']:.4f}, sim value = "
                            f"{return_dict['value_prediction_sim']:.4f} (sim eval time = {query_time:.3f} sec)",
                            log_file,
                        )

                    if cfg.search_depth > 1:
                        assert not cfg.ar_qvalue_prediction, "Search depth > 1 not supported for Q-value prediction!"
                        for depth in range(2, cfg.search_depth + 1):
                            for future_state_latent in future_state_return_dict["future_state_samples_list"]:
                                next_generated_latent_with_future_state = future_state_latent.clone()
                                # Rearrange latent frames such that predicted future state replaces current state in the sequence
                                rearranged_next_latent_with_future_state = (
                                    next_generated_latent_with_future_state.clone()
                                )
                                rearranged_next_latent_with_future_state[
                                    :, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1
                                ] = next_generated_latent_with_future_state[
                                    :, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1
                                ]
                                ################################
                                # Predict next action
                                ################################
                                data_batch = action_return_dict["data_batch"]
                                data_batch["num_conditional_frames"] = (
                                    model.config.min_num_conditional_frames
                                )  # Reset to the original value
                                data_batch["mask_current_state_action_for_value_prediction"] = (
                                    False  # Don't use input masking for action prediction
                                )
                                if cfg.randomize_seed:
                                    seed = secrets.randbits(32) % 256
                                else:
                                    seed = cfg.seed + query_idx
                                batch_size = 1
                                next_generated_latent_with_action, next_orig_clean_latent_frames = (
                                    model.generate_samples_from_batch(
                                        data_batch,
                                        n_sample=batch_size,
                                        num_steps=cfg.num_denoising_steps_action,
                                        seed=seed,
                                        is_negative_prompt=False,
                                        use_variance_scale=cfg.use_variance_scale,
                                        skip_vae_encoding=True,
                                        previous_generated_latent=rearranged_next_latent_with_future_state,  # Use future state sample since parts of value sample might be masked out
                                        return_orig_clean_latent_frames=True,
                                    )
                                )  # (B, C'=16, T', H'=28, W'=28)
                                # Extract the action chunk prediction from the generated samples
                                action_latent_idx = action_return_dict["latent_indices"]["action_latent_idx"]
                                action_indices = torch.full(
                                    (batch_size,),
                                    action_latent_idx,
                                    dtype=torch.int64,
                                    device=next_generated_latent_with_action.device,
                                )
                                next_actions = (
                                    extract_action_chunk_from_latent_sequence(
                                        next_generated_latent_with_action,
                                        (cfg.chunk_size, ACTION_DIM),
                                        action_indices=action_indices,
                                    )
                                    .to(torch.float32)
                                    .cpu()
                                    .numpy()
                                )
                                # Unnormalize actions
                                if cfg.unnormalize_actions:
                                    next_actions = unnormalize_actions(next_actions, dataset_stats)
                                # Squeeze and convert to list
                                next_actions = next_actions[0]
                                next_actions = [next_actions[i] for i in range(len(next_actions))]
                                actions_by_depth.append(next_actions)
                                ################################
                                # Predict next future state
                                ################################
                                future_state_return_dict = get_future_state_prediction(
                                    cfg,
                                    model=planning_model if planning_model is not None else model,
                                    data_batch=action_return_dict["data_batch"],
                                    generated_latent_with_action=next_generated_latent_with_action,
                                    orig_clean_latent_frames=next_orig_clean_latent_frames,
                                    future_proprio_latent_idx=action_return_dict["latent_indices"][
                                        "future_proprio_latent_idx"
                                    ],
                                    future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_wrist_image_latent_idx"
                                    ],
                                    future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_wrist_image2_latent_idx"
                                    ],
                                    future_image_latent_idx=action_return_dict["latent_indices"][
                                        "future_image_latent_idx"
                                    ],
                                    future_image2_latent_idx=action_return_dict["latent_indices"][
                                        "future_image2_latent_idx"
                                    ],
                                    seed=cfg.seed + query_idx,
                                    randomize_seed=cfg.randomize_seed,
                                    num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                    use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                    num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                    future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                                )
                                # Track per-depth prediction
                                future_image_predictions_by_depth.append(
                                    future_state_return_dict["future_image_predictions"]
                                )
                                ################################
                                # Predict next value
                                ################################
                                value_return_dict = get_value_prediction(
                                    cfg,
                                    model=planning_model if planning_model is not None else model,
                                    data_batch=action_return_dict["data_batch"],
                                    future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                    seed=cfg.seed + query_idx,
                                    randomize_seed=cfg.randomize_seed,
                                    num_denoising_steps_value=cfg.num_denoising_steps_value,
                                    use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                    num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                                )
                                return_dict["value_prediction"] = value_return_dict["value_prediction"]
                                value_predictions_by_depth.append(return_dict["value_prediction"])
                                log_message(
                                    f"Query {query_idx + 1}/{num_queries_best_of_n}: Value prediction: {return_dict['value_prediction']:.4f}",
                                    log_file,
                                )
                    # Add results to the return dict
                    return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                    return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                    return_dict["actions_by_depth"] = actions_by_depth
                    query_results.append(return_dict)

            # Replace value of each query with aggregate value over all value predictions
            # This is only applicable when search depth > 1 because otherwise the aggregation is handled in get_value_prediction()
            # For search depth == 1, the return dict contains the aggregated value prediction already
            if cfg.search_depth > 1:
                for query_idx, return_dict in enumerate(query_results):
                    if cfg.search_depth_value_aggregation_scheme == "average":
                        return_dict["value_prediction"] = np.mean(return_dict["value_predictions_by_depth"]).item()
                    elif cfg.search_depth_value_aggregation_scheme == "use_last_value":
                        return_dict["value_prediction"] = return_dict["value_predictions_by_depth"][-1]
                    else:
                        raise ValueError(
                            f"Invalid search depth value aggregation scheme: {cfg.search_depth_value_aggregation_scheme}"
                        )
            # Print all value predictions
            for query_idx, return_dict in enumerate(query_results):
                predicted_value = return_dict["value_prediction"]
                log_message(
                    f"Query {query_idx + 1}/{cfg.num_queries_best_of_n} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                    log_file,
                )
            # Only keep the first num_open_loop_steps timesteps of the action chunk
            for query_idx, return_dict in enumerate(query_results):
                return_dict["actions"] = return_dict["actions"][: cfg.num_open_loop_steps]
            # Get dict: seed number -> (action chunk, future state, value)
            seed_to_return_dict = {
                cfg.seed + query_idx: (
                    return_dict["actions"],
                    return_dict["future_image_predictions"],
                    return_dict["value_prediction"],
                    # return_dict["all_value_predictions"],
                )
                for query_idx, return_dict in enumerate(query_results)
            }
            # Get seed with highest value
            best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
            best_actions = best_return_dict[0]
            best_future_predictions = best_return_dict[1]
            best_value_predictions = best_return_dict[2]

            # Calculate planning advantage and collect info
            base_value = seed_to_return_dict[cfg.seed][2]
            advantage = best_value_predictions - base_value
            all_values = [rd["value_prediction"] for rd in query_results]

            planning_stats.append({
                "timestep": t,
                "base_value": float(base_value),
                "best_value": float(best_value_predictions),
                "advantage": float(advantage),
                "all_values": [float(v) for v in all_values],
            })

            # base_value_std = torch.tensor(seed_to_return_dict[cfg.seed][3]).std()

            ###############################################################################################
            # Value-function agreement metric (world model vs. simulator)
            # -----------------------------------------------------------
            # For each candidate seed we evaluated the value function on both the world-model-predicted
            # future state and the simulator-reached future state. Here we record which seed each method
            # ranks highest and whether they agree. `regret_sim` is the simulator value lost by executing
            # the world-model's choice instead of the simulator's choice (0 == they pick equally good
            # chunks under the simulator metric). Selection/execution above still use the WM value.
            ###############################################################################################
            if cfg.measure_value_agreement:
                wm_values = [rd["value_prediction_wm"] for rd in query_results]
                sim_values = [rd["value_prediction_sim"] for rd in query_results]
                best_idx_wm = int(np.argmax(wm_values))
                best_idx_sim = int(np.argmax(sim_values))
                argmax_agree = best_idx_wm == best_idx_sim
                spearman = spearman_corr(wm_values, sim_values)
                regret_sim = float(sim_values[best_idx_sim] - sim_values[best_idx_wm])
                value_agreement_stats.append({
                    "timestep": t,
                    "num_seeds": len(query_results),
                    "seeds": [cfg.seed + i for i in range(len(query_results))],
                    "wm_values": [float(v) for v in wm_values],
                    "sim_values": [float(v) for v in sim_values],
                    "best_seed_wm": int(cfg.seed + best_idx_wm),
                    "best_seed_sim": int(cfg.seed + best_idx_sim),
                    "argmax_agree": bool(argmax_agree),
                    "spearman": float(spearman),
                    "sim_value_at_wm_best": float(sim_values[best_idx_wm]),
                    "sim_value_at_sim_best": float(sim_values[best_idx_sim]),
                    "regret_sim": regret_sim,
                })
                log_message(
                    f"t={t}: [Agreement] WM picks seed {cfg.seed + best_idx_wm}, "
                    f"sim picks seed {cfg.seed + best_idx_sim} -> "
                    f"{'AGREE' if argmax_agree else 'DISAGREE'} | spearman={spearman:.3f} | "
                    f"sim regret={regret_sim:.4f}",
                    log_file,
                )

            # Use the best actions, future predictions, and value predictions found
            action_queue.extend(best_actions)
            future_image_predictions_list.append(best_future_predictions)
            log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)
            # log_message(f"t={t}: Base value = {base_value:.4f}, Advantage = {advantage:.4f}, Value std of the first seed: {base_value_std}", log_file)

            ###############################################################################################
            # Value prediction gap metric
            # ----------------------------
            # Goal: quantify how much the value prediction changes when it is conditioned on the
            # world-model-PREDICTED future state vs. on the ACTUAL reached state. A large gap implies the
            # world model produces poor future states (or the value function is highly variant); a small
            # gap implies the value function is consistent across predicted and actual states, so the
            # issue does not stem from the value function.
            #
            # Timing: the world model predicts the state `chunk_size` steps ahead (training pairs the
            # current state with the frame at `relative_step_idx + chunk_size`). For the real reached
            # state at the next requery to actually match that horizon AND to have been produced by the
            # predicted action chunk (not a replanned one), we require `num_open_loop_steps == chunk_size`.
            # Under that condition the next requery lands exactly `chunk_size` steps later and executes the
            # full predicted chunk, so `real_current_state_latent` is the true counterpart of the WM future
            # state. If `num_open_loop_steps < chunk_size`, the comparison is horizon-misaligned (see the
            # warning in validate_config), so the metric is gated off below.
            ###############################################################################################
            if (
                cfg.compute_value_prediction_gap
                and not cfg.use_parallel_inference
                and not cfg.use_simulator_for_planning  # no world model to compare against
                and not cfg.use_planning_gating  # steps may skip planning; gap metric not meaningful
                and cfg.search_depth == 1
                and cfg.num_open_loop_steps == cfg.chunk_size  # required for horizon alignment (see comment above)
            ):
                # Resolve the previous inference step's metric using the now-observed real state.
                if (
                    pending_value_metric is not None
                    and pending_value_metric["value_cond_latents"] is not None
                    and real_current_state_latent is not None
                ):
                    with torch.inference_mode():
                        # Build value-conditioning latents where the world-model-predicted future state
                        # frames are replaced by the actual reached state (= current-state frames at this
                        # inference step). Only the future-state content differs from the WM-conditioned
                        # latents, so the resulting value gap isolates world-model state quality.
                        real_state_cond_latents = []
                        for wm_latent in pending_value_metric["value_cond_latents"]:
                            real_state_latent = wm_latent.clone()
                            real_state_latent[
                                :, :, FUTURE_STATE_START_LATENT_IDX : FUTURE_STATE_END_LATENT_IDX + 1
                            ] = real_current_state_latent[
                                :, :, CURR_STATE_START_LATENT_IDX : CURR_STATE_END_LATENT_IDX + 1
                            ]
                            real_state_cond_latents.append(real_state_latent)
                    # Predict the value on the real reached state (same value-function code path and
                    # ensemble settings used to produce the world-model value, only the future state differs).
                    # Use the data batch from the prediction step so the current-state context matches.
                    real_value_return_dict = get_value_prediction(
                        cfg,
                        model=planning_model if planning_model is not None else model,
                        data_batch=pending_value_metric["data_batch"],
                        future_state_samples_list=real_state_cond_latents,
                        seed=cfg.seed,
                        randomize_seed=cfg.randomize_seed,
                        num_denoising_steps_value=cfg.num_denoising_steps_value,
                        use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                        num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                    )
                    value_pred_wm = pending_value_metric["value_pred_wm"]
                    value_pred_real = real_value_return_dict["value_prediction"]
                    value_pred_gap = value_pred_wm - value_pred_real
                    value_gap_stats.append({
                        "pred_timestep": pending_value_metric["timestep"],
                        "real_timestep": t,
                        "value_pred_wm": float(value_pred_wm),
                        "value_pred_real": float(value_pred_real),
                        "value_pred_gap": float(value_pred_gap),
                        "abs_value_pred_gap": float(abs(value_pred_gap)),
                    })
                    log_message(
                        f"t={t}: [Value gap] WM-state value (predicted at t={pending_value_metric['timestep']}) = "
                        f"{value_pred_wm:.4f}, real-state value = {value_pred_real:.4f}, "
                        f"gap (WM - real) = {value_pred_gap:.4f}",
                        log_file,
                    )

                # Stash this step's best prediction to be resolved at the next inference step.
                best_query_idx = best_seed - cfg.seed
                pending_value_metric = {
                    "timestep": t,
                    "value_pred_wm": best_value_predictions,
                    "value_cond_latents": query_results[best_query_idx].get("value_cond_latents", None),
                    "data_batch": query_results[best_query_idx].get("data_batch", None),
                }

        # Get next action from chunk
        action = action_queue.popleft()
        # RoboCasa: Policy was trained on 7-dim manipulation actions, but env expects 12-dim (7 + 5 mobile base)
        # Append [0, 0, 0, 0, -1] for mobile base since we're not using it
        if action.shape[-1] == 7 and env.action_dim == 12:
            mobile_base_action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
            action = np.concatenate([action, mobile_base_action])
        # Execute action
        print(f"t: {t}, action: {action}")
        obs, reward, done, info = env.step(action)
        episode_length += 1
        # Collect action data if enabled
        if cfg.data_collection:
            actions_list.append(action)
        # Check for success
        if env._check_success():
            success = True
            log_message(f"  Success detected at timestep {t}!", log_file)
            break

    # Log episode result
    log_message(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length: {episode_length})",
        log_file,
    )
    # Log value prediction gap summary for the episode
    if len(value_gap_stats) > 0:
        mean_gap = np.mean([s["value_pred_gap"] for s in value_gap_stats])
        mean_abs_gap = np.mean([s["abs_value_pred_gap"] for s in value_gap_stats])
        mean_value_wm = np.mean([s["value_pred_wm"] for s in value_gap_stats])
        mean_value_real = np.mean([s["value_pred_real"] for s in value_gap_stats])
        log_message(
            f"  Episode {episode_idx} value prediction gap (over {len(value_gap_stats)} inference steps): "
            f"mean WM-state value = {mean_value_wm:.4f}, mean real-state value = {mean_value_real:.4f}, "
            f"mean gap (WM - real) = {mean_gap:.4f}, mean |gap| = {mean_abs_gap:.4f}",
            log_file,
        )
    # Prepare collected data if enabled
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C) - left camera
            secondary_images=np.stack(secondary_images_list, axis=0),  # (T, H, W, C) - right camera
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
            planning_stats=json.dumps(planning_stats),
            value_gap_stats=json.dumps(value_gap_stats),
            value_agreement_stats=json.dumps(value_agreement_stats),
            gating_stats=json.dumps(gating_stats),
        )
        # Add future image predictions
        if len(future_image_predictions_list) > 0:
            # Primary camera predictions (left third-person)
            if (
                "future_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image"] is not None
            ):
                future_primary_images = [x["future_image"] for x in future_image_predictions_list]
                collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Secondary camera predictions (right third-person)
            if (
                "future_image2" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image2"] is not None
            ):
                future_secondary_images = [x["future_image2"] for x in future_image_predictions_list]
                collected_data["future_secondary_images"] = np.stack(future_secondary_images, axis=0)
            # Wrist camera predictions
            if (
                "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None
    return (
        success,
        episode_length,
        replay_primary_images,
        replay_secondary_images,
        replay_wrist_images,
        future_image_predictions_list,
        collected_data,
        planning_stats,
        value_gap_stats,
        value_agreement_stats,
        gating_stats,
    )


def run_task(
    cfg: PolicyEvalConfig,
    task_name: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    log_file=None,
):
    """Run evaluation for a single task."""
    log_message(f"\nEvaluating task: {task_name}", log_file)
    successes = []
    episode_lengths = []
    total_episodes = 0
    total_successes = 0
    all_value_gap_stats = []  # Value prediction gap metric, accumulated across all episodes
    all_value_agreement_stats = []  # Value-function agreement metric (WM vs sim), accumulated across all episodes
    all_gating_stats = []  # Adaptive planning-gating metric, accumulated across all episodes
    # In fixed-initial-condition repeat mode, the loop count is the number of repeats; otherwise it is
    # the usual number of trials per task.
    num_episodes = cfg.num_fixed_init_repeats if cfg.repeat_fixed_init else cfg.num_trials_per_task
    # Holders for the single env + post-reset snapshot reused across all repeats in fixed-init mode.
    fixed_init_env = None
    fixed_init_snapshot = None
    fixed_init_task_description = None
    for episode_idx in range(num_episodes):
        # Per-episode policy action-generation seed override (None => use cfg.seed). Only set in
        # fixed-init mode, where it is what we vary across repeats.
        policy_seed = None
        if cfg.repeat_fixed_init:
            # Fixed-initial-condition repeat mode: build the environment and its initial condition
            # exactly once, snapshot the post-reset state, and restore that identical state before each
            # repeat. Only the policy action-generation seed changes across repeats, so any difference in
            # the rollouts is due solely to the policy's own sampling stochasticity.
            if fixed_init_env is None:
                # Use a single fixed environment seed and a single fixed scene for ALL repeats.
                env_seed = cfg.seed if (cfg.deterministic or cfg.deterministic_reset) else None
                # `create_robocasa_env` selects the scene as `(episode_idx // 10) % num_scenes`; pass
                # `fixed_init_scene_idx * 10` so we deterministically land on the requested scene.
                env, env_kwargs = create_robocasa_env(
                    cfg, seed=env_seed, episode_idx=cfg.fixed_init_scene_idx * 10
                )
                if cfg.deterministic_reset:
                    reset_seed = (
                        cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
                    )
                    set_seed_everywhere(reset_seed)
                env.reset()
                fixed_init_env = env
                fixed_init_snapshot = snapshot_env_state(env)
                fixed_init_task_description = env.get_ep_meta()["lang"]
            else:
                # Restore the exact post-reset state captured on the first repeat (bit-identical initial
                # conditions; no re-reset, which would resample placements).
                env = fixed_init_env
                restore_env_state(env, fixed_init_snapshot)
            task_description = fixed_init_task_description
            policy_seed = (cfg.policy_seed if cfg.policy_seed is not None else cfg.seed) + episode_idx
            log_message(
                f"Starting repeat {episode_idx + 1}/{num_episodes} (fixed init, "
                f"scene {cfg.fixed_init_scene_idx}, policy seed {policy_seed})...",
                log_file,
            )
            log_message(f"\nTask description: {task_description}", log_file)
        else:
            log_message(f"Starting episode {episode_idx + 1}...", log_file)
            # Create environment with scene selection based on episode index
            # Episodes 0-9 use scene 0, 10-19 use scene 1, etc.
            if cfg.deterministic or cfg.deterministic_reset:
                # Deterministic seeding for reproducibility
                seed = cfg.seed * episode_idx * 256
            else:
                seed = None
            env, env_kwargs = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
            # Reset environment
            # NOTE: Every reset changes the scene/task! So only reset ONCE per episode.
            if cfg.deterministic_reset:
                reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
                set_seed_everywhere(reset_seed)
            env.reset()
            # Get task description
            # IMPORTANT: Get the task description AFTER resetting the environment. Resetting the environment changes the task!
            task_description = env.get_ep_meta()["lang"]
            log_message(f"\nTask description: {task_description}", log_file)
        # Run episode
        (
            success,
            length,
            replay_primary_images,
            replay_secondary_images,
            replay_wrist_images,
            future_image_predictions_list,
            collected_data,
            planning_stats,
            value_gap_stats,
            value_agreement_stats,
            gating_stats,
        ) = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            episode_idx,
            log_file,
            policy_seed=policy_seed,
        )
        successes.append(success)
        episode_lengths.append(length)
        all_value_gap_stats.extend(value_gap_stats)
        all_value_agreement_stats.extend(value_agreement_stats)
        all_gating_stats.extend(gating_stats)

        if planning_stats:
            avg_advantage = np.mean([stat["advantage"] for stat in planning_stats])
            log_message(f"Average planning advantage for episode {episode_idx + 1} advantage: {avg_advantage:.4f}", log_file)

        # Update counters
        total_episodes += 1
        if success:
            total_successes += 1
        # Save rollout video
        rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data", f"{task_name}--{DATE_TIME}")
        os.makedirs(rollout_data_dir, exist_ok=True)
        save_rollout_video(
            replay_primary_images,
            replay_secondary_images,
            replay_wrist_images,
            episode_idx,
            success=success,
            task_description=task_description,
            rollout_data_dir=rollout_data_dir,
            log_file=log_file,
        )
        # Save rollout video with future image predictions
        if len(future_image_predictions_list) > 0:
            # Extract future predictions from the list
            future_primary_image_predictions = None
            future_secondary_image_predictions = None
            future_wrist_image_predictions = None
            if (
                "future_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image"] is not None
            ):
                future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
            if (
                "future_image2" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_image2"] is not None
            ):
                future_secondary_image_predictions = [x["future_image2"] for x in future_image_predictions_list]
            if (
                "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
            # Save video with predictions if all three camera predictions are available
            if (
                future_primary_image_predictions is not None
                and future_secondary_image_predictions is not None
                and future_wrist_image_predictions is not None
            ):
                save_rollout_video_with_future_image_predictions(
                    replay_primary_images,
                    replay_secondary_images,
                    replay_wrist_images,
                    episode_idx,
                    success=success,
                    task_description=task_description,
                    rollout_data_dir=rollout_data_dir,
                    chunk_size=cfg.chunk_size,
                    num_open_loop_steps=cfg.num_open_loop_steps,
                    future_primary_image_predictions=future_primary_image_predictions,
                    future_secondary_image_predictions=future_secondary_image_predictions,
                    future_wrist_image_predictions=future_wrist_image_predictions,
                    # Per-inference-step planned value (best-of-N selected value), aligned with the
                    # future-image predictions; drives the 3rd-row value graph.
                    value_predictions=[s["best_value"] for s in planning_stats] if planning_stats else None,
                    show_diff=False,
                    log_file=log_file,
                    show_timestep=True,
                )
            else:
                log_message(
                    f"Skipping video with future predictions - not all camera predictions available "
                    f"(primary: {future_primary_image_predictions is not None}, "
                    f"secondary: {future_secondary_image_predictions is not None}, "
                    f"wrist: {future_wrist_image_predictions is not None})",
                    log_file,
                )
        # Save collected data if data_collection is enabled
        if cfg.data_collection and collected_data is not None:
            # Skip episodes that are less than 5 timesteps long (because sometimes success is detected immediately upon starting since the envs are buggy)
            if len(collected_data["actions"]) < 5:
                log_message(
                    f"Skipping saving this episode: less than 5 timesteps long (only {len(collected_data['actions'])} timesteps).",
                    log_file,
                )
                continue
            # Save episodic HDF5 data
            processed_task_description = (
                task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
            )
            ep_filename = f"{DATE_TIME}--episode_data--task={processed_task_description}--ep={episode_idx}--success={success}.hdf5"
            ep_filepath = os.path.join(rollout_data_dir, ep_filename)
            with h5py.File(ep_filepath, "w") as f:
                for k, v in collected_data.items():
                    if isinstance(v, np.ndarray):
                        is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                        if is_image and cfg.jpeg_compress:
                            jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                            if len(jpeg_list) == 1:
                                # Skip saving the array if it only has one element (causes error during create_dataset())
                                continue
                            dt = h5py.vlen_dtype(np.dtype("uint8"))
                            f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                        else:
                            f.create_dataset(k, data=v)
                    else:
                        f.attrs[k] = v
                f.attrs["task_description"] = task_description
            log_message(f"Saved episode data to: {ep_filepath}", log_file)
        # Close environment after each episode. In fixed-init repeat mode the single env is reused
        # across all repeats, so keep it alive and close it once after the loop.
        if not cfg.repeat_fixed_init:
            env.close()
        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    # Close the single reused environment in fixed-init repeat mode (kept alive across all repeats above).
    if cfg.repeat_fixed_init and fixed_init_env is not None:
        fixed_init_env.close()
    # Calculate statistics
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    log_message(f"Task {task_name} results:", log_file)
    log_message(f"  Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"  Average episode length: {avg_length:.1f}", log_file)
    log_message(
        f"  Successes: {sum(successes)}/{len(successes)}",
        log_file,
    )
    # Value prediction gap metric summary (across all episodes)
    task_value_gap_summary = None
    if len(all_value_gap_stats) > 0:
        task_mean_gap = float(np.mean([s["value_pred_gap"] for s in all_value_gap_stats]))
        task_mean_abs_gap = float(np.mean([s["abs_value_pred_gap"] for s in all_value_gap_stats]))
        task_mean_value_wm = float(np.mean([s["value_pred_wm"] for s in all_value_gap_stats]))
        task_mean_value_real = float(np.mean([s["value_pred_real"] for s in all_value_gap_stats]))
        task_value_gap_summary = {
            "num_samples": len(all_value_gap_stats),
            "mean_value_wm": task_mean_value_wm,
            "mean_value_real": task_mean_value_real,
            "mean_gap": task_mean_gap,
            "mean_abs_gap": task_mean_abs_gap,
        }
        log_message(f"  Value prediction gap ({len(all_value_gap_stats)} inference steps across all episodes):", log_file)
        log_message(f"    Mean WM-state value:   {task_mean_value_wm:.4f}", log_file)
        log_message(f"    Mean real-state value: {task_mean_value_real:.4f}", log_file)
        log_message(f"    Mean gap (WM - real):  {task_mean_gap:.4f}", log_file)
        log_message(f"    Mean |gap|:            {task_mean_abs_gap:.4f}", log_file)
    # Value-function agreement metric summary (world model vs. simulator, across all episodes)
    task_value_agreement_summary = None
    if len(all_value_agreement_stats) > 0:
        agreement_rate = float(np.mean([s["argmax_agree"] for s in all_value_agreement_stats]))
        valid_spearmans = [s["spearman"] for s in all_value_agreement_stats if not np.isnan(s["spearman"])]
        mean_spearman = float(np.mean(valid_spearmans)) if len(valid_spearmans) > 0 else float("nan")
        mean_regret_sim = float(np.mean([s["regret_sim"] for s in all_value_agreement_stats]))
        task_value_agreement_summary = {
            "num_samples": len(all_value_agreement_stats),
            "argmax_agreement_rate": agreement_rate,
            "mean_spearman": mean_spearman,
            "mean_regret_sim": mean_regret_sim,
        }
        log_message(
            f"  Value-function agreement (WM vs sim) over {len(all_value_agreement_stats)} inference steps:",
            log_file,
        )
        log_message(f"    Argmax agreement rate (same seed picked): {agreement_rate:.4f}", log_file)
        log_message(f"    Mean Spearman rank corr (WM vs sim):      {mean_spearman:.4f}", log_file)
        log_message(f"    Mean sim regret from trusting WM's pick:  {mean_regret_sim:.4f}", log_file)
    # Adaptive planning-gating summary (across all episodes)
    task_gating_summary = None
    if len(all_gating_stats) > 0:
        plan_trigger_rate = float(np.mean([s["did_plan"] for s in all_gating_stats]))
        mean_gating_advantage = float(np.mean([s["advantage"] for s in all_gating_stats]))
        task_gating_summary = {
            "num_requeries": len(all_gating_stats),
            "plan_trigger_rate": plan_trigger_rate,
            "mean_advantage": mean_gating_advantage,
        }
        log_message(f"  Planning gating over {len(all_gating_stats)} requeries:", log_file)
        log_message(f"    Planning triggered rate:  {plan_trigger_rate:.4f}", log_file)
        log_message(f"    Mean predicted advantage: {mean_gating_advantage:.4f}", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_name}": success_rate,
                f"avg_episode_length/{task_name}": avg_length,
                f"num_successes/{task_name}": sum(successes),
                f"num_episodes/{task_name}": len(successes),
            }
        )
        if task_value_gap_summary is not None:
            wandb.log(
                {
                    f"value_gap/mean_gap/{task_name}": task_value_gap_summary["mean_gap"],
                    f"value_gap/mean_abs_gap/{task_name}": task_value_gap_summary["mean_abs_gap"],
                    f"value_gap/mean_value_wm/{task_name}": task_value_gap_summary["mean_value_wm"],
                    f"value_gap/mean_value_real/{task_name}": task_value_gap_summary["mean_value_real"],
                }
            )
        if task_value_agreement_summary is not None:
            wandb.log(
                {
                    f"value_agreement/argmax_agreement_rate/{task_name}": task_value_agreement_summary[
                        "argmax_agreement_rate"
                    ],
                    f"value_agreement/mean_spearman/{task_name}": task_value_agreement_summary["mean_spearman"],
                    f"value_agreement/mean_regret_sim/{task_name}": task_value_agreement_summary["mean_regret_sim"],
                }
            )
        if task_gating_summary is not None:
            wandb.log(
                {
                    f"gating/plan_trigger_rate/{task_name}": task_gating_summary["plan_trigger_rate"],
                    f"gating/mean_advantage/{task_name}": task_gating_summary["mean_advantage"],
                }
            )
    return success_rate, avg_length, successes


@draccus.wrap()
def eval_robocasa(cfg: PolicyEvalConfig) -> float:
    """Main function to evaluate a trained policy on RoboCasa tasks."""
    # Set DETERMINISTIC environment variable if on deterministic mode
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)
    # Validate evaluation configuration
    validate_config(cfg)
    # Initialize T5 text embeddings cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    # Load Cosmos Policy dataset stats
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        worker_pool.start_workers()
        # Set model to None here because each worker will load its own copy
        model = None
        planning_model = None
    # If using serial inference, initialize model and Cosmos config
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None
        # Initialize planning model if specified
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)
    # Log parallel inference configuration if enabled
    if cfg.use_parallel_inference and worker_pool:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
    # Run evaluation
    log_message(f"\nStarting evaluation for task: {cfg.task_name}", log_file)
    log_message(f"Number of trials: {cfg.num_trials_per_task}", log_file)
    success_rate, avg_length, successes = run_task(
        cfg,
        cfg.task_name,
        model,
        planning_model,
        dataset_stats,
        worker_pool,
        log_file,
    )
    # Log final results
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Task: {cfg.task_name}", log_file)
    log_message(f"Success rate: {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log_message(f"Average episode length: {avg_length:.1f}", log_file)
    log_message(f"Total episodes: {len(successes)}", log_file)
    log_message(f"Total successes: {sum(successes)}", log_file)
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "final_success_rate": success_rate,
                "final_avg_episode_length": avg_length,
                "total_episodes": len(successes),
                "total_successes": sum(successes),
            }
        )
        wandb.save(local_log_filepath)
        wandb.finish()
    # Cleanup
    if worker_pool:
        worker_pool.shutdown()
    log_message(f"\nResults saved to: {local_log_filepath}", log_file)
    return success_rate


if __name__ == "__main__":
    eval_robocasa()
