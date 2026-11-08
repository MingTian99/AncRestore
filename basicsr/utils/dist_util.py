import functools
import datetime
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _coerce_timeout(timeout):
    if isinstance(timeout, datetime.timedelta):
        return timeout
    if isinstance(timeout, (int, float)):
        return datetime.timedelta(seconds=float(timeout))
    if isinstance(timeout, str):
        text = timeout.strip().lower()
        if text.endswith('h'):
            return datetime.timedelta(hours=float(text[:-1]))
        if text.endswith('m'):
            return datetime.timedelta(minutes=float(text[:-1]))
        if text.endswith('s'):
            return datetime.timedelta(seconds=float(text[:-1]))
        return datetime.timedelta(seconds=float(text))
    raise TypeError(f'Unsupported distributed timeout type: {type(timeout)}')


def _normalize_dist_kwargs(kwargs):
    kwargs = dict(kwargs)
    timeout = kwargs.pop('timeout', None)
    timeout_seconds = kwargs.pop('timeout_seconds', None)
    if timeout is None:
        timeout = timeout_seconds
    if timeout is not None:
        kwargs['timeout'] = _coerce_timeout(timeout)
    return kwargs


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    kwargs = _normalize_dist_kwargs(kwargs)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None, **kwargs):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    kwargs = _normalize_dist_kwargs(kwargs)
    dist.init_process_group(backend=backend, **kwargs)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
