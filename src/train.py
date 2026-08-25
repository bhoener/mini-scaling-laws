import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT
from dataloader import DataLoader
import wandb


def train(
    train_filepath: str,
    model: GPT,
    steps: int,
    bsz: int,
    seq_len: int,
    lr_adamw: float = 3e-4,
    lr_muon: float = 3e-4,
    cooldown_frac: float = 0.2,
    logging: bool = True,
    log_every: int = 10,
    device: torch.device | None = None,
    wandb_run: wandb.Run | None = None,
) -> tuple[float, float]:
    train_dl = DataLoader(train_filepath, bsz, seq_len, device=device)

    params = sum(p.numel() for p in model.parameters())

    optim_adamw = torch.optim.AdamW([p for p in model.parameters() if p.ndim != 2], lr=lr_adamw, fused=True)
    optim_muon = torch.optim.AdamW([p for p in model.parameters() if p.ndim == 2], lr=lr_muon)

    def get_lr(it: int, lr_base: float, cooldown_frac: float, total_steps: int) -> float:
        progress = it / total_steps
        if progress < (1 - cooldown_frac):
            return lr_base

        cooldown_progress = (it - (total_steps * (1 - cooldown_frac))) / (total_steps * cooldown_frac)
        return (1 - cooldown_progress) * lr_base

    total_flops = 0

    for step in range(steps):
        xs, ys = train_dl.next()

        pred = model(xs)

        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), ys.view(-1))

        optim_adamw.zero_grad()
        optim_muon.zero_grad()

        loss.backward()

        for param_group in optim_adamw.param_groups:
            param_group["lr"] = get_lr(step, optim_adamw.defaults["lr"], cooldown_frac, steps)
        for param_group in optim_muon.param_groups:
            param_group["lr"] = get_lr(step, optim_muon.defaults["lr"], cooldown_frac, steps)

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim_adamw.step()
        optim_muon.step()

        total_flops += 6 * params * xs.numel()

        wandb_run.log({"train_loss": loss.item(), "norm": norm.item(), "total_flops": total_flops, "lr_mult": get_lr(step, 1.0, cooldown_frac, steps)})

        if logging and step % log_every == 0:
            print(f"step: {step:8d} | loss: {loss:8.4f} | norm: {norm:8.4f}")

    wandb_run.finish()

    return loss.item()

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpt = GPT(vocab_size=14, d_model=512, n_heads=16, n_layers=12).to(device)

    print(train("numbers.npy", "numbers_val.npy", gpt, 1000, 16, device=device))

if __name__ == "__main__":
    main()
