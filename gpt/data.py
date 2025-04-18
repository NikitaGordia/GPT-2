import os
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import torch


def load_tokens(filename: Union[str, Path]) -> torch.Tensor:
    """Load token data from a numpy file and convert to PyTorch tensor.

    Args:
        filename: Path to the numpy file containing token data

    Returns:
        PyTorch tensor of token data with dtype torch.long
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """A lightweight data loader for token data used in language model training.

    This loader handles distributed training scenarios by managing data shards
    and providing batches of data appropriate for the current rank in distributed setup.
    """

    def __init__(
        self, data_dir: Union[str, Path], batch_size: int, seq_length: int, split: str, env: Any
    ):
        """Initialize the data loader.

        Args:
            data_dir: Directory containing data shards
            batch_size: Batch size (B)
            seq_length: Sequence length (T)
            split: Data split to use ('train' or 'val')
            env: Training environment object with ddp_config and logger
        """
        self.B = batch_size
        self.T = seq_length

        # Handle case where ddp_config might be None
        if env.ddp_config is None:
            self.rank, self.world_size = 0, 1
        else:
            self.rank, self.world_size = env.ddp_config

        if split not in {"train", "val"}:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")

        shard_paths = os.listdir(data_dir)
        self.shards = sorted(
            [os.path.join(data_dir, shard) for shard in shard_paths if split in shard]
        )

        if len(self.shards) == 0:
            raise FileNotFoundError(
                f"No shards found for split '{split}' in directory '{data_dir}'"
            )

        env.logger(f"Found {len(self.shards)} shards for split '{split}'")
        self.reset()

    def reset(self, to_shard: int = 0) -> None:
        """Reset the data loader to a specific shard.

        Args:
            to_shard: Index of the shard to reset to
        """
        if to_shard >= len(self.shards):
            raise IndexError(f"Shard index {to_shard} out of range (0-{len(self.shards) - 1})")

        self.current_shard = to_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.rank

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of data.

        Returns:
            Tuple of (input_tokens, target_tokens) tensors shaped as (B, T)

        Raises:
            RuntimeError: If there is not enough data for a batch
        """
        B, T = self.B, self.T

        start, end = self.current_position, self.current_position + T * B + 1
        if end > len(self.tokens):
            raise RuntimeError(
                f"Not enough data for the batch. Rank: {self.rank}/{self.world_size}, "
                f"end: {end}, n_tokens: {len(self.tokens)}"
            )

        buf = self.tokens[start:end]
        X = buf[:-1].view(B, T)
        Y = buf[1:].view(B, T)

        self.current_position += B * T * self.world_size

        # Check if we need to move to the next shard
        if self.current_position + (B * T * self.world_size + 1) > len(self.tokens):
            next_shard = (self.current_shard + 1) % len(self.shards)
            self.reset(to_shard=next_shard)

        return X, Y
