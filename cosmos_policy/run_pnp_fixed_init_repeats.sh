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

uv run --extra cu128 --group robocasa --python 3.10 \
python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
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
    --task_name PnPCounterToCab \
    --repeat_fixed_init True \
    --num_fixed_init_repeats 10 \
    --fixed_init_scene_idx 0 \
    --policy_seed 195 \
    --run_id_note chkpt45000--5stepAct--fixedInit--policySeedSweep \
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
