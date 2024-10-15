import contextlib
import os
import sys
from typing import List, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


def iterate_files(base_paths: Iterable[str]) -> List[str]:
    for name in base_paths:
        if not os.path.isdir(name):
            yield name
            continue
        for root, dirs, files in os.walk(name):
            for fname in files:
                path = os.path.join(root, fname)
                yield path


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


# from: https://stackoverflow.com/a/45735618/2289509
@contextlib.contextmanager
def smart_open(filename: str, mode: str = 'r', *args, **kwargs):
    """Open files and i/o streams transparently."""
    if filename == '-':
        if 'r' in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if 'b' in mode:
            fh = stream.buffer
        else:
            fh = stream
        close = False
    else:
        fh = open(filename, mode, *args, **kwargs)
        close = True

    try:
        yield fh
    finally:
        if close:
            try:
                fh.close()
            except AttributeError:
                pass


def pad_sequences(sequences, maxlen, dtype, value):
    # based on keras' pad_sequences()
    num_samples = len(sequences)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        trunc = [value] + s[:maxlen - 1]
        x[idx, :len(trunc)] = np.asarray(trunc, dtype=dtype)
    return x


def pad_lists(sequences, maxlen, value):
    x = []
    for seq in sequences:
        x.append([value] + seq + [value] * (maxlen - len(seq) - 1))
    return x


def shuffle_in_unison(*arrs):
    rng_state = np.random.get_state()
    for arr in arrs:
        np.random.set_state(rng_state)
        np.random.shuffle(arr)


def flatten(dataloader: DataLoader):
    """
    Combines batches from a DataLoader into a single tensor. If
    there are multiple tensors returned in each batch, they will be
    flattened separately, returning multiple tensors.

    For example, if the DataLoader returns a samples tensor of shape NxD and a
    labels tensor of shape Nx1 for each batch (N is the batch size),
    this function will return a tuple of two tensors of shapes
    (N*M)xD and (N*M)x1 where M is the number of batches.

    :param dataloader: The DataLoader to flatten.
    :return: A tuple of one or more tensors containing the data from all
        batches.
    """

    out_tensors_cache = []

    for batch in dataloader:

        # Handle case of batch being a tensor (no labels)
        if torch.is_tensor(batch):
            batch = (batch,)
        # Handle case of batch being a dict
        elif isinstance(batch, dict):
            batch = tuple(batch[k] for k in sorted(batch.keys()))
        elif not isinstance(batch, tuple) and not isinstance(batch, list):
            raise TypeError("Unexpected type of batch object")

        for i, tensor in enumerate(batch):
            if i >= len(out_tensors_cache):
                out_tensors_cache.append([])

            out_tensors_cache[i].append(tensor)

    out_tensors = tuple(
        # 0 is batch dimension
        torch.cat(tensors_list, dim=0) for tensors_list in out_tensors_cache
    )

    return out_tensors


def compare_model_weights(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    """Compares the weights of two models to check if they are identical.

    Args:
        model1 (torch.nn.Module): First model.
        model2 (torch.nn.Module): Second model.

    Returns:
        bool: True if the weights are identical, False otherwise.
    """
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    # Compare each tensor in the state dict of both models
    for key in model1_state_dict:
        if key not in model2_state_dict:
            print(f"Model2 missing key: {key}")
            return False
        if not torch.equal(model1_state_dict[key], model2_state_dict[key]):
            print(f"Difference found in layer: {key}")
            return False

    print("The models have identical weights.")
    return True
