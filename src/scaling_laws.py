import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from train import train
from copy import deepcopy
import time
from tokenizer import VOCAB_SIZE


def main() -> None:
    import wandb

    DEFAULT_CONFIG = {
        "vocab_size": VOCAB_SIZE,
        "head_dim": 20,
        "n_heads": 16,
        "n_layers": 12,
        "steps": 2000,
        "batch_size": 64,
        "seq_len": 512,
        "lr_adamw": 3e-4,
        "lr_muon": 3e-4,
        "cooldown_frac": 0.2,
        "moe": False,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    head_dims = range(10, 41, 4)
    losses = []
    group_name = f"head-dim-{int(time.time())}"
    for head_dim in head_dims:
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg["head_dim"] = head_dim
        run = wandb.init(project="MiniScalingLaws", config=cfg, group=group_name)
        model = GPT(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["head_dim"] * cfg["n_heads"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
        )
        model = model.to(device)
        model = torch.compile(model, mode="reduce-overhead")

        train_loss = train(
            "data/",
            model=model,
            steps=cfg["steps"],
            bsz=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            lr_adamw=cfg["lr_adamw"],
            lr_muon=cfg["lr_muon"],
            cooldown_frac=cfg["cooldown_frac"],
            logging=True,
            device=device,
            wandb_run=run,
        )

        losses.append(train_loss)

        print(cfg)
        print(train_loss)


if __name__ == "__main__":
    main()
