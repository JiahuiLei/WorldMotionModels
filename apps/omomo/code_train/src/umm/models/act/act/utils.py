"""Utility classes and functions for ACT (Action Chunking Transformer) training.

Includes data structures (NestedTensor), data loading from HDF5 demos,
iteration-based batch sampling, and distributed helper functions.

Example:
    >>> from act.utils import load_demo_dataset, IterationBasedBatchSampler
"""

from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from h5py import File, Group, Dataset
from typing import Optional


class NestedTensor(object):
    """Container for a tensor paired with an optional mask tensor.

    Used to handle variable-size inputs in detection/transformer models
    where padding masks are needed.

    Args:
        tensors: The data tensor.
        mask: Optional boolean mask tensor of the same spatial dimensions.

    Example:
        >>> nt = NestedTensor(torch.randn(2, 3, 4, 4), torch.zeros(2, 4, 4, dtype=torch.bool))
    """

    def __init__(self, tensors, mask: Optional[Tensor]):
        """Initialize NestedTensor with a data tensor and optional mask.

        Args:
            tensors: The data tensor.
            mask: Optional boolean mask tensor indicating padding positions.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """Move tensor and mask to the specified device.

        Args:
            device: Target torch device.

        Returns:
            New NestedTensor on the target device.

        Example:
            >>> nt_cuda = nt.to("cuda")
        """
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """Decompose into the raw tensor and mask.

        Returns:
            Tuple of (tensors, mask).

        Example:
            >>> tensor, mask = nt.decompose()
        """
        return self.tensors, self.mask

    def __repr__(self):
        """Return string representation of the underlying tensor.

        Returns:
            String representation.

        Example:
            >>> print(nt)
        """
        return str(self.tensors)


def is_dist_avail_and_initialized():
    """Check if PyTorch distributed is available and initialized.

    Returns:
        True if distributed is available and initialized.

    Example:
        >>> if is_dist_avail_and_initialized(): ...
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """Get the rank of the current process in distributed training.

    Returns:
        Integer rank (0 if not in distributed mode).

    Example:
        >>> rank = get_rank()
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if this is the main (rank 0) process.

    Returns:
        True if rank is 0.

    Example:
        >>> if is_main_process(): print("main process")
    """
    return get_rank() == 0


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        """Initialize the iteration-based batch sampler.

        Args:
            batch_sampler: Underlying batch sampler to wrap.
            num_iterations: Total number of iterations to sample.
            start_iter: Starting iteration (for resuming).

        Example:
            >>> ibs = IterationBasedBatchSampler(batch_sampler, 100000)
        """
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        """Yield batches until num_iterations is reached.

        Yields:
            Batch indices from the underlying sampler.

        Example:
            >>> for batch in sampler: ...
        """
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        """Return the remaining number of iterations.

        Returns:
            num_iterations - start_iter.

        Example:
            >>> len(sampler)
            100000
        """
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """Initialize the random seed for a dataloader worker process.

    The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    Args:
        worker_id: The worker process index assigned by the DataLoader.
        base_seed: Base random seed. If None, a random seed is generated
            using PyTorch's random generator.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4,
        ...     worker_init_fn=worker_init_fn)
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    """Recursively load content from an HDF5 file, group, or dataset.

    Args:
        file: An h5py File, Group, or Dataset object.

    Returns:
        Nested dictionary for Files/Groups, numpy array for Datasets.

    Example:
        >>> data = load_content_from_h5_file(h5py.File("data.h5", "r"))
    """
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    """Load an entire HDF5 file into a nested dictionary.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Nested dictionary with all groups and datasets.

    Example:
        >>> data = load_hdf5("demos.h5")
    """
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    """Load trajectory data from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing trajectory groups.
        num_traj: Number of trajectories to load (None for all).

    Returns:
        Dictionary mapping trajectory keys to their contents.

    Example:
        >>> trajs = load_traj_hdf5("demos.h5", num_traj=10)
    """
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    """Load a demonstration dataset from an HDF5 trajectory file.

    Args:
        path: Path to the HDF5 file.
        keys: List of target keys to extract (mapped via TARGET_KEY_TO_SOURCE_KEY).
        num_traj: Number of trajectories to load (None for all).
        concat: Whether to concatenate trajectories into single arrays.

    Returns:
        Dictionary mapping target keys to concatenated arrays or lists of arrays.

    Example:
        >>> dataset = load_demo_dataset("demos.h5", num_traj=10, concat=False)
        >>> obs_list = dataset["observations"]
    """
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset
