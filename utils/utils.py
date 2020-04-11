import torch
import torch.distributed
import os

# from NVIDIA/UnsupervisedLandmarkLearning
def initialize_distributed(config):
    """
    Sets up necessary stuff for distributed
    training if the world_size is > 1
    Args:
        config (dict): configurations for this run
    """
    if not config['use_ddp']:
        return
    # Manually set the device ids.
    local_rank = config['local_rank']
    world_size = config['world_size']
    rank = config['rank']
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print('set_device')
    print(rank % torch.cuda.device_count())
    print('Global Rank:')
    print(rank)

    # Call the init process
    if world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
        master_port = os.getenv('MASTER_PORT', '6666')
        init_method += master_ip+':'+master_port
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size, rank=rank,
            init_method=init_method)
