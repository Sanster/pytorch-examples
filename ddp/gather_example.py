# Most code copy from https://github.com/facebookresearch/detectron2

import pickle
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor
    
def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor
    
def f(rank):
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:42', world_size=4, rank=rank)

    data = {
        'loss': float(f'{rank}.1'),
        'correct_count': rank
    }

    if dist.get_backend() == "nccl":
        group = dist.new_group(backend="gloo")
    else:
        group = dist.group.WORLD

    # convert all picklable to byte tensor
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    if rank==0:
        max_size = max(size_list)
        gather_t = [torch.empty((max_size,), dtype=torch.uint8) for _ in size_list]
        dist.gather(tensor, gather_t, 0, group=group)

        data_list = [] # store all 'loss' and 'correct_count' data from other ranks
        for size, tensor in zip(size_list, gather_t):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
    else:
        dist.gather(tensor, [], 0, group=group)

if __name__ == "__main__":
    mp.spawn(f, nprocs=4, args=())
