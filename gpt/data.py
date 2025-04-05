import os

import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, data_dir, B, T, split, env):
        self.B = B
        self.T = T
        self.rank, self.world_size = env.ddp_config
        assert split in {"train", "val"}

        self.shards = sorted(
            [os.path.join(data_dir, shard) for shard in os.listdir(data_dir) if split in shard]
        )
        assert len(self.shards) > 0, f"no shards found for split {split}"
        env.logger(f"Found {len(self.shards)} shards for split {split}")

        self.reset()

    def reset(self, to_shard=0):
        self.current_shard = to_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.rank

    def next_batch(self):
        B, T = self.B, self.T

        start, end = self.current_position, self.current_position + T * B + 1
        if end > len(self.tokens):
            raise Exception(
                f"Not enough data for the first batch. Rank: {self.rank}/{self.world_size}, end: {end}, n_tokens: {len(self.tokens)}"
            )

        buf = self.tokens[start:end]
        X = buf[:-1].view(B, T)
        Y = buf[1:].view(B, T)

        self.current_position += B * T * self.world_size

        if self.current_position + (B * T * self.world_size + 1) > len(self.tokens):
            self.reset(to_shard=(self.current_shard + 1) % len(self.shards))
        return X, Y
