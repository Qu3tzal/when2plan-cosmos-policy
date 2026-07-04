#!/bin/bash
# Fixed-initial-condition repeat eval for PnPCounterToCab.
#
# Runs the SAME initial condition (object, placement, scene) 10 times and varies ONLY the policy
# action-generation seed across the repeats. This isolates how much the policy's own sampling
# stochasticity changes the rollout, holding the environment fixed.
#
# - The environment is held deterministic (`--deterministic True`); its seed is fixed at `--seed`.
# - `--repeat_fixed_init True` enables the mode; `--num_fixed_init_repeats` sets the number of repeats.
# - Repeat r uses policy generation seed `policy_seed + r` (defaults to `seed` when `--policy_seed`
#   is omitted), so each repeat samples actions differently from the identical start state.
# - `--fixed_init_scene_idx` picks which test scene (index into layout_and_style_ids) to freeze on.
#
# Note: keep `--randomize_seed False` so the per-repeat policy seed actually takes effect (randomize
# would draw a fresh random seed each query and defeat the controlled comparison).
set -e

# HF_TOKEN and WANDB_API_KEY are read from the environment; do not hardcode them here.
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running}"

# Keep MuJoCo's EGL offscreen renderer and model inference on DIFFERENT physical GPUs.
# When they share a GPU, the inference bursts (best-of-N + ensembles) fill GPU memory and EGL
# rendering silently returns corrupted observations (tiled/garbage/gray frames) mid-episode.
# EGL enumerates a single device in this container (NVIDIA_DRIVER_CAPABILITIES has no "graphics"),
# which is physical GPU 0, so we remap CUDA so that torch's cuda:0 is physical GPU 1 instead.
# Both are overridable from the calling environment.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

uv sync --extra cu128 --group robocasa  --python 3.10
uv pip install -e robocasa-cosmos-policy
# uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
# uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
# uv run --extra cu128 --group robocasa --python 3.10 hf auth login

uv run --extra cu128 --group robocasa --python 3.10 \
python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --planning_model_config_name cosmos_predict2_2b_480p_robocasa_50_demos_per_task__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only \
    --planning_model_ckpt_path /workspace/checkpoints/iter_000018000/model \
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
    --task_name PnPCounterToCab \
    --repeat_fixed_init True \
    --num_fixed_init_repeats 10 \
    --fixed_init_scene_idx 1 \
    --policy_seed 195 \
    --run_id_note chkpt45000--5stepAct--fixedInit--policySeedSweep \
    --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
    --seed 195 \
    --randomize_seed False \
    --deterministic True \
    --use_variance_scale False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 10 \
    --num_denoising_steps_future_state 5 \
    --num_denoising_steps_value 5 \
    --num_denoising_steps_action_nonplanning 5 \
    --num_denoising_steps_value_nonplanning 1 \
    --data_collection True \
    --use_parallel_inference False \
    --num_queries_best_of_n 8 \
    --use_ensemble_future_state_predictions True \
    --use_ensemble_value_predictions True \
    --ar_future_prediction True \
    --ar_value_prediction True \
    --mask_current_state_action_for_value_prediction True \
    --parallel_timeout 300 \
    --use_planning_gating False \
    --use_simulator_for_planning False \
    --search_depth 1
