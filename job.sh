#!/bin/sh

# Wisteria job options
#PJM -L rscgrp=regular-a
#PJM -L node=4
#PJM --mpi proc=4
#PJM -L elapse=48:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j

set -eu

module purge
module load aquarius
module load cuda/12.6
module load hpcx/2.15.0
module load singularity/3.9.5

cd "${PJM_O_WORKDIR}"

export NUM_NODES=$(wc -l < "${PJM_O_NODEINF}")
export MASTER_ADDR=$(head -1 "${PJM_O_NODEINF}")
export MASTER_PORT=29547

echo "NUM_NODES=${NUM_NODES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

mpirun -np "${PJM_MPI_PROC}" \
  --hostfile "${PJM_O_NODEINF}" \
  -map-by ppr:1:node -mca pml ob1 \
  -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
  -x NUM_NODES -x MASTER_ADDR -x MASTER_PORT \
  -x PATH -x LD_LIBRARY_PATH \
  singularity exec --nv --bind "$(pwd)" docker/cosmos-policy.sif bash finetune_wm_with_rollouts.sh
