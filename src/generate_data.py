import os
import multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from tokenizer import STOI, EOT

print(STOI)

def encode(src):
    return list(src.encode("utf-8")) + [EOT]


def main() -> None:
    CHUNK_SIZE = 100000000
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")

    shards_dir = "data/"

    if not os.path.exists(shards_dir):
        os.makedirs(shards_dir)

    chunk = [EOT]
    chunk_len = 1
    chunk_id = 0

    pool = mp.Pool(max(1, os.cpu_count() // 2))

    for batch in tqdm(pool.imap(encode, ds["train"]["text"], 16)):
        chunk.extend(batch)

        chunk_len += len(batch)

        if chunk_len > CHUNK_SIZE:
            np.save(os.path.join(shards_dir, f"chunk_{chunk_id:04d}.npy"), np.asarray(chunk))
            chunk = [EOT]
            chunk_len = 1
            chunk_id += 1



    

if __name__ == "__main__":
    main()
