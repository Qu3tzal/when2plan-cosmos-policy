import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
local_rank = rank % torch.cuda.device_count()
torch.cuda.set_device(local_rank)

x = torch.ones(1, device=torch.cuda.current_device())
dist.all_reduce(x)
print(f"rank={rank} world={world} host={torch.cuda.get_device_name(local_rank)[:20]} sum={x.item()}", flush=True)

dist.destroy_process_group()
