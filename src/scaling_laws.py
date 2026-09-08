import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, ModelConfig, MoEConfig
from train import train
from copy import deepcopy
import time
from tokenizer import VOCAB_SIZE


def main() -> None:
    import wandb

    torch.set_float32_matmul_precision("high")

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
        "moe_experts": 8,
        "moe_active": 2,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    head_dims = range(10, 41, 4)
    losses = []
    group_name = f"head-dim-{int(time.time())}"
    for head_dim in head_dims:
        train_cfg = deepcopy(DEFAULT_CONFIG)
        train_cfg["head_dim"] = head_dim
        cfg = ModelConfig(
            vocab_size=DEFAULT_CONFIG["vocab_size"],
            d_model=head_dim * DEFAULT_CONFIG["n_heads"],
            n_heads=DEFAULT_CONFIG["n_heads"],
            n_layers=DEFAULT_CONFIG["n_layers"],
            moe=DEFAULT_CONFIG["moe"],
            moe_config=MoEConfig(
                num_experts=DEFAULT_CONFIG["moe_experts"],
                num_active=DEFAULT_CONFIG["moe_active"],
            ),
        )

        run = wandb.init(project="MiniScalingLaws", config=train_cfg, group=group_name)
        model = GPT(cfg)
        model = model.to(device)
        model = torch.compile(model, mode="reduce-overhead")

        train_loss = train(
            "data/",
            model=model,
            steps=train_cfg["steps"],
            bsz=train_cfg["batch_size"],
            seq_len=train_cfg["seq_len"],
            lr_adamw=train_cfg["lr_adamw"],
            lr_muon=train_cfg["lr_muon"],
            cooldown_frac=train_cfg["cooldown_frac"],
            logging=True,
            device=device,
            wandb_run=run,
        )

        losses.append(train_loss)

        print(cfg)
        print(train_loss)


if __name__ == "__main__":
    main()
