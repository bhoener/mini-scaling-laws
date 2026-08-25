import os
import torch
import numpy as np

class DataLoader:
    def __init__(self, filepath: str, batch_size: int, seq_len: int, device: torch.device | None = None):
        self.filepath = filepath
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.device = device

        self.shards = sorted(os.listdir(filepath))

        self.data = torch.from_numpy(np.load(os.path.join(filepath, self.shards[0])))
        self.idx = 0
        self.shard_idx = 0
        self.datalen = len(self.data)

    def next_shard(self) -> None:
        self.shard_idx += 1
        if self.shard_idx >= len(self.shards):
            self.shard_idx = 0
        self.data = torch.from_numpy(np.load(os.path.join(self.filepath, self.shards[self.shard_idx])))
        self.idx = 0
        self.datalen = len(self.data)

    def next(self):
        if self.idx + self.batch_size * self.seq_len + 1 >= self.datalen:
            self.idx = 0

        batch = self.data[self.idx:self.idx + self.batch_size * self.seq_len + 1]
        xs = batch[:-1].view(self.batch_size, self.seq_len)
        ys = batch[1:].view(self.batch_size, self.seq_len)
        self.idx += self.batch_size * self.seq_len
        return xs.to(self.device), ys.to(self.device)

def main() -> None:
    dl = DataLoader("data/", 2, 8)
    for _ in range(5):
        print(dl.next())

if __name__ == "__main__":
    main()