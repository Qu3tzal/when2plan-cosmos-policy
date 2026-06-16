#!/bin/sh

# Wisteria job options — 2-node smoke test for multi-node launch
#PJM -L rscgrp=short-a
#PJM -L node=2
#PJM --mpi proc=2
#PJM -L elapse=00:15:00
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

echo "=== Allocation ==="
echo "PJM_MPI_PROC=${PJM_MPI_PROC}"
echo "PJM_O_NODEINF=${PJM_O_NODEINF}"
cat "${PJM_O_NODEINF}"

export NUM_NODES=$(wc -l < "${PJM_O_NODEINF}")
export MASTER_ADDR=$(head -1 "${PJM_O_NODEINF}")
export MASTER_PORT=29547

echo "NUM_NODES=${NUM_NODES} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

echo
echo "=== Rank var detection inside Singularity ==="
mpirun -np "${PJM_MPI_PROC}" \
  --hostfile "${PJM_O_NODEINF}" \
  -map-by ppr:1:node -mca pml ob1 \
  -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
  -x PATH -x LD_LIBRARY_PATH \
  singularity exec --nv --bind "$(pwd)" docker/cosmos-policy.sif bash -c '
    echo "host=$(hostname) OMPI=${OMPI_COMM_WORLD_RANK:-unset} PMIX=${PMIX_RANK:-unset}"
  '

echo
echo "=== Fabric names ==="
mpirun -np "${PJM_MPI_PROC}" \
  --hostfile "${PJM_O_NODEINF}" \
  -map-by ppr:1:node -mca pml ob1 \
  -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
  bash -c '
    echo "--- host=$(hostname) ---"
    ip -o link show | awk "{print \$2}" | tr -d ":"
    ls /sys/class/infiniband/ 2>/dev/null
  '

echo
echo "=== Multi-node NCCL rendezvous smoke test ==="
mpirun -np "${PJM_MPI_PROC}" \
  --hostfile "${PJM_O_NODEINF}" \
  -map-by ppr:1:node -mca pml ob1 \
  -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
  -x NUM_NODES -x MASTER_ADDR -x MASTER_PORT \
  -x PATH -x LD_LIBRARY_PATH \
  singularity exec --nv --bind "$(pwd)" docker/cosmos-policy.sif bash -c "
    export NODE_RANK=\${OMPI_COMM_WORLD_RANK:-\${PMIX_RANK:-0}}
    export NCCL_IB_HCA=mlx5
    export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
    cd $PWD
    echo \"[host=\$(hostname) NODE_RANK=\$NODE_RANK] launching torchrun\"
    uv run --extra cu128 --group robocasa --python 3.10 \
      torchrun --nnodes=\${NUM_NODES} --node_rank=\${NODE_RANK} \
               --nproc_per_node=8 \
               --master_addr=\${MASTER_ADDR} --master_port=\${MASTER_PORT} \
               _nccl_smoke_test.py
  "

echo
echo "=== Done ==="
