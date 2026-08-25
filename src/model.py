import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))

class RoPE(nn.Module):
    def __init__(self, d: int, base: float = 10000):
        super().__init__()
        self.d = d
        self.base = base

        angles = torch.empty(d)
        for i in range(d // 2):
            angles[2 * i] = base ** (-2 * i / d)
            angles[2 * i + 1] = base ** (-2 * i / d)

        self.register_buffer("angles", angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, d = x.size()

        position_angles = einsum(torch.arange(l, device=x.device), self.angles, "t, d -> t d").unsqueeze(0).unsqueeze(0)

        x_shuffled = torch.stack([-x[:, :, :, 1::2], x[:, :, :, ::2]], dim=-1).flatten(-2)

        return x * torch.cos(position_angles) + x_shuffled * torch.sin(position_angles)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.rope = RoPE(d_model // n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.rope(norm(rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.n_heads)))
        K = self.rope(norm(rearrange(self.wk(x), "b t (h d) -> b h t d", h=self.n_heads)))
        V = rearrange(self.wv(x), "b t (h d) -> b h t d", h=self.n_heads)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        return self.wo(rearrange(attn_out, "b h t d -> b t (h d)"))

class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.W = nn.Linear(d_in, d_h)
        self.V = nn.Linear(d_in, d_h)
        self.W2 = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.W(x)
        return self.W2((o * F.sigmoid(o)) * self.V(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.mha = SelfAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.mlp = SwiGLU(d_in=d_model, d_h=d_model*4, d_out=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([DecoderBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 * (2 * self.n_layers) ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)

        for layer in self.blocks:
            x = layer(x)

        return self.out_proj(x)

def main() -> None:
    rope = RoPE(16)
    assert rope(torch.randn(2, 4, 32, 16)).size() == (2, 4, 32, 16)
    print("rope shapes correct")

    mha = SelfAttentionBlock(128, 16)
    assert mha(torch.randn(2, 4, 128)).size() == (2, 4, 128)
    print("mha shapes correct")

    mlp = SwiGLU(32, 32*4, 32)
    assert mlp(torch.randn(2, 4, 32)).size() == (2, 4, 32)
    print("mlp shapes correct")

    db = DecoderBlock(128, 16)
    assert db(torch.randn(2, 4, 128)).size() == (2, 4, 128)
    print("decoder block shapes correct")

    gpt = GPT(vocab_size=14, d_model=20*16, n_heads=16, n_layers=12)
    assert gpt(torch.randint(0, 14, (2, 8))).size() == (2, 8, 14)
    print("gpt shapes correct")
    print("gpt parameters:", sum(p.numel() for p in gpt.parameters()))

if __name__ == "__main__":
    main()