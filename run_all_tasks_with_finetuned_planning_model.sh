#!/bin/bash
set -e


uv sync --extra cu128 --group robocasa  --python 3.10
uv pip install -e robocasa-cosmos-policy
# uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
# uv run --extra cu128 --group robocasa --python 3.10 hf auth login

tasks=("PnPCounterToCab" "PnPCabToCounter" "PnPCounterToSink" "PnPSinkToCounter" "PnPCounterToMicrowave" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPStoveToCounter" "OpenSingleDoor" "CloseSingleDoor" "OpenDoubleDoor" "CloseDoubleDoor" "OpenDrawer" "CloseDrawer" "TurnOnStove" "TurnOffStove" "TurnOnSinkFaucet" "TurnOffSinkFaucet" "TurnSinkSpout" "CoffeeSetupMug" "CoffeeServeMug" "CoffeePressButton" "TurnOnMicrowave" "TurnOffMicrowave")

for item in "${tasks[@]}"; do
    uv run --extra cu128 --group robocasa --python 3.10 \
    python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --planning_model_config_name cosmos_predict2_2b_480p_robocasa_50_demos_per_task__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only \
        --planning_model_ckpt_path /workspace/checkpoints/iter_000018000/model \
        --use_wrist_image True \
        --num_wrist_images 1 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 32 \
        --num_open_loop_steps 32 \
        --task_name $item \
        --num_trials_per_task 20 \
        --run_id_note chkpt45000--5stepAct--seed195--deterministic \
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
        --data_collection True \
        --use_parallel_inference False \
        --num_queries_best_of_n 8 \
        --use_ensemble_future_state_predictions True \
        --use_ensemble_value_predictions True \
        --ar_future_prediction True \
        --ar_value_prediction True \
        --parallel_timeout 300 \
        --search_depth 1
done
