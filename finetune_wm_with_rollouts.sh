#!/bin/bash
set -e

export WORK_DIR=/work/02/gb20/b20080/Workspace/when2plan/cosmos-policy/
export HF_HOME=$WORK_DIR/.cache/huggingface
export TMPDIR=$WORK_DIR/.tmp
export UV_CACHE_DIR=$WORK_DIR/.cache
export UV_PYTHON_INSTALL_DIR=$WORK_DIR/.local/share/uv/python
export XDG_CACHE_HOME=$WORK_DIR/.cache          # catches torch hub, pip wheels-as-cache, etc.
export WANDB_DIR=$WORK_DIR/.wandb               # wandb run logs
export WANDB_CACHE_DIR=$WORK_DIR/.cache/wandb
export TORCH_HOME=$WORK_DIR/.cache/torch        # torchvision/torch.hub weights
export IMAGINAIRE_OUTPUT_ROOT=$WORK_DIR/.imaginaire_output

export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# Multi-node: rank is set per-process by mpirun in job.sh.
export NODE_RANK=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}
export NNODES=${NUM_NODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12341}

# NCCL over Wisteria-A InfiniBand
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3

uv sync --extra cu128 --group robocasa  --python 3.10

uv pip install -e robocasa-cosmos-policy
# uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/download_kitchen_assets.py
uv run --extra cu128 --group robocasa --python 3.10 robocasa-cosmos-policy/robocasa/scripts/setup_macros.py

export BASE_DATASETS_DIR=/work/02/gb20/b20080/Workspace/when2plan/cosmos-policy/data

uv run --extra cu128 --group robocasa --python 3.10 \
  torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} \
           --nproc_per_node=8 \
           --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
           -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__resumeFrom50K_648_rollouts_Vsprime_value_func" \
  trainer.grad_accum_iter=12
