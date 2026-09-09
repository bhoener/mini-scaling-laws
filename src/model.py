import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat
from dataclasses import dataclass


@dataclass
class MoEConfig:
    num_experts: int = 8
    num_active: int = 2


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    moe: bool = False
    moe_config: MoEConfig | None = None


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=1e-7)


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

        position_angles = (
            einsum(torch.arange(l, device=x.device), self.angles, "t, d -> t d")
            .unsqueeze(0)
            .unsqueeze(0)
        )

        x_shuffled = torch.stack([-x[:, :, :, 1::2], x[:, :, :, ::2]], dim=-1).flatten(
            -2
        )

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
        Q = self.rope(
            norm(rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.n_heads))
        )
        K = self.rope(
            norm(rearrange(self.wk(x), "b t (h d) -> b h t d", h=self.n_heads))
        )
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


class MoE(nn.Module):
    def __init__(self, config: MoEConfig, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.config = config
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.WG = nn.Linear(d_in, config.num_experts, bias=False)
        self.WN = nn.Linear(d_in, config.num_experts, bias=False)

        nn.init.zeros_(self.WG.weight)
        nn.init.zeros_(self.WN.weight)

        self.W1 = nn.Parameter(torch.randn(config.num_experts, d_in, d_h))
        self.W2 = nn.Parameter(torch.randn(config.num_experts, d_h, d_out))

        with torch.no_grad():
            self.W1 /= d_in**0.5
            self.W2 /= d_h**0.5

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, use_kernel=False) -> tuple[torch.Tensor, ...]:
        B, T, C = x.size()
        xWg = self.WG(x)
        noise_weights = F.softplus(self.WN(x))
        H = xWg + torch.randn_like(xWg, device=x.device) * noise_weights

        # we want to group tokens into batches and send those groups through their corresponding experts
        # probably best to flatten batch/time?
        # just get (-1, C)
        # then rearrange to (groups, per_group_length*, C)
        # then do matmul with weight of shape (groups, C, H)

        # also need to remember we have multiple experts per token
        # so (num_active, groups, per_group_length*, C)

        # we have a list of per-token selected experts
        # we want to get a jagged tensor containing groups of tokens

        # but how to do this?

        # can we flatten G before idx?

        H = rearrange(H, "b t a -> (b t) a")

        top_k = torch.topk(H, self.config.num_active)
        idx = top_k.indices
        kth_highest = rearrange(torch.amin(top_k.values, dim=-1, keepdim=True), "(b t) a -> b t a", b=B, t=T)

        mask = torch.ones_like(H, device=x.device).bool()
        mask.scatter_(-1, idx, 0)

        H[mask] = float("-inf")

        G = F.softmax(H, dim=-1)

        # now we have a sparse tensor of topk probs of shape (B x T x N)

        # seems like there are two ways to build the jagged tensor
        # 1) go per-token through the topk probs and conditionally insert
        # 2) go per-expert through the topk probs (eg. probs[:, :, i]) and take nonzeros

        # not sure if things will be differentiable

        jagged = []
        indices = []

        flattened_x = rearrange(x, "b t c -> (b t) c")

        for expert in range(self.config.num_experts):
            expert_indices = torch.nonzero(G[:, expert]).squeeze(-1)
            jagged.append(flattened_x[expert_indices])
            indices.append(expert_indices)

        grouped = torch.nested.nested_tensor(jagged, layout=torch.jagged)

        # ok good. now we have a group tensor of shape (num_experts, j1, d_model)
        # where j1 represents the tokens per expert
        # in theory all we have to do from here is matmul by the weight

        # ooh https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grouped_mm.html
        # need to make sure we cast to bf16 and have >= 80 SMs

        if use_kernel:
            l1preact = F.grouped_mm(grouped, self.W1)
            mlp_outs = F.grouped_mm(self.act(l1preact), self.W2)
        else:
            l1preact = grouped @ self.W1
            mlp_outs = self.act(l1preact) @ self.W2

        # now we have mlp outs in shape (num_experts, j1, d_model)
        # we must rearrange to (B, T, d_model) given the previous indices
        # but we can first do (b t) d_model since we flattened

        # our indices are of size num_experts (b t)
        # need to do some sort of inverse

        # in order to create the grouped tensor, we take indices along probs[:, expert]
        # we could do the very long way:

        # out = torch.empty(B * T, C)
        # for expert, expert_indices in enumerate(indices):
        #     for k, i in enumerate(expert_indices):
        #         out[i] = out[i] + mlp_outs[expert, k] * flattened_gates[i, expert]

        # but it would prob be extremely slow, esp in backward pass

        # seems like the play is to either use some hidden built-in function
        # or to use .values and .offsets of the jagged tensor

        # it looks like if we use .values, we get a tensor of size (B * T * active_experts, d_model)
        # this is very good (assuming we can backward through it all, which looks like yes)
        # it would be nice if we could use scatter, but we need to multiply by gates and sum
        # across experts first. we could concatenate the indices and use them as an index tensor
        # to sort the values with

        # if we have some tokens per expert
        # [[t2 t7 t1 t4]
        #  [t6 t5 t0 t3 t2]
        #  [t3 t8 t4]]

        # flattened to
        # [t2 t7 t1 t4 t6 t5 t0 t3 t2 t3 t8 t4]

        # we have idx
        # [ 2  7  1  4  6  5  0  3  2  3  8  4]

        # what if we construct an empty out tensor
        # then do out[idx] = values

        # nope. recall that we will have multiple of the same token across experts
        # because we have multiple active experts per token

        # indices are the same shape as the flattened tokens, but they only go up to B * T
        # and not B * T * num_active

        # so what to do?
        # if we rearranged the gate values and multiplied them, we would still have to rearrange to reduce
        # seems we are forced to rearrange the tokens
        # there are two ways to do this:

        # - make an empty (B * T * num_active, d_model) tensor and index into it, putting tokens of the same
        # index next to each other
        # - make an empty (num_active, B * T, d_model) tensor and do something similar

        # i'm just not sure how to do either of these efficiently

        # maybe need to think bigger picture
        # we have some tokens that we have gathered into a jagged tensor
        # we have sent this jagged tensor through an FFN
        # we want to un-nest this tensor back into the original shape
        # the problem is that we constructed the tensor per-expert (probs[:, expert])
        # i guess we could do the reverse
        # make a (b t active) c empty tensor
        # take our per-expert indices
        # these are guaranteed not to contain duplicates

        # then we can do out[idx] = out[idx] + group * gate

        # how to get gates?

        # we have idx
        # torch.gather(G, idx, dim=-1) -> (B, T, active)
        # then rearrange to (B T, active)
        # but this is bad bc we are going by expert
        # how to know which gate is for the given expert?

        # =========================================================
        # gates = torch.tensor([[0, 0.5, 0.5],
        #                      [0.9, 0.1, 0],
        #                      [0.6, 0, 0.4]])

        # group_indices = torch.tensor([0, 1])

        # idx = torch.tensor([[1, 2],
        #                   [0, 1],
        #                   [2, 0]])

        # expert = 2

        # print(torch.gather(gates, -1, idx))
        # tensor([[0.5000, 0.5000],
        #         [0.9000, 0.1000],
        #         [0.4000, 0.6000]])
        # print(torch.gather(gates, -1, idx)[group_indices][idx[group_indices]==expert])
        # tensor([0.5000])
        # =========================================================

        # went down a very long rabbit hole of debugging to realize:
        # 1) forgot to 1/sqrt(fan_in) initialize my mlp layers
        # 2) forgot that i'm adding and not assigning to out so can't use torch.empty()
        out = torch.zeros(B * T, C)

        gates = torch.gather(G, dim=-1, index=idx)

        # bro i genuinely don't even know what i did here. it's probably wrong.
        # essentially un-nests the nested mlp output tensor
        # for each expert group, we find the gates (given the indices saved from earlier)
        # then just add to the entry in out
        # this is doable because inside each group there are no duplicate indices

        for group_indices, (expert, group) in zip(
            indices, enumerate(mlp_outs.unbind())
        ):
            if len(group_indices > 0):
                group_gates = gates[group_indices][
                    idx[group_indices] == expert
                ].unsqueeze(-1)

                out[group_indices] = out[group_indices] + group * group_gates

        importance = G.sum(0, keepdim=True).unsqueeze(0)
        importance_loss = (importance.std() / importance.mean()) ** 2

        load = (
            0.5 * (1 + torch.erf(xWg - kth_highest / (2**0.5 * noise_weights)))
        ).sum((0, 1))
        load_loss = (load.std() / load.mean()) ** 2

        return rearrange(out, "(b t) c -> b t c", b=B, t=T), importance_loss, load_loss


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        moe: bool = False,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.moe = moe

        self.mha = SelfAttentionBlock(d_model=d_model, n_heads=n_heads)
        self.mlp = (
            MoE(moe_config, d_in=d_model, d_h=d_model * 4, d_out=d_model)
            if moe
            else SwiGLU(d_in=d_model, d_h=d_model * 4, d_out=d_model)
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x + self.mha(norm(x))
        if self.moe:
            moe_out, importance_loss, load_loss = self.mlp(norm(x))
            x = x + moe_out
            return x, importance_loss, load_loss
        else:
            x = x + self.mlp(norm(x))
            return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    moe=config.moe,
                    moe_config=config.moe_config,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.out_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(
            self.out_proj.weight, mean=0.0, std=0.02 * (2 * config.n_layers) ** -0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)

        for layer in self.blocks:
            if self.config.moe:
                importance_loss_accum = 0.0
                load_loss_accum = 0.0

                x, importance_loss, load_loss = layer(x)
                importance_loss_accum = importance_loss_accum + importance_loss
                load_loss_accum = load_loss_accum + load_loss
            else:
                x = layer(x)

        if self.config.moe:
            return (
                self.out_proj(x),
                importance_loss_accum / self.config.n_layers,
                load_loss_accum / self.config.n_layers,
            )
        else:
            return self.out_proj(x)


def main() -> None:
    rope = RoPE(16)
    assert rope(torch.randn(2, 4, 32, 16)).size() == (2, 4, 32, 16)
    print("rope shapes correct")

    mha = SelfAttentionBlock(128, 16)
    assert mha(torch.randn(2, 4, 128)).size() == (2, 4, 128)
    print("mha shapes correct")

    mlp = SwiGLU(32, 32 * 4, 32)
    assert mlp(torch.randn(2, 4, 32)).size() == (2, 4, 32)
    print("mlp shapes correct")

    db = DecoderBlock(128, 16)
    assert db(torch.randn(2, 4, 128)).size() == (2, 4, 128)
    print("decoder block shapes correct")

    moe = MoE(MoEConfig(8, 2), 64, 128, 64)
    assert moe(torch.randn(4, 12, 64))[0].size() == (4, 12, 64)
    print("moe shapes correct")

    gpt = GPT(ModelConfig(14, 64, 16, 12))
    assert gpt(torch.randint(0, 14, (2, 8))).size() == (2, 8, 14)
    print("gpt shapes correct")
    print("gpt parameters:", sum(p.numel() for p in gpt.parameters()))

    gpt_moe = GPT(ModelConfig(14, 64, 16, 12, True, MoEConfig(8, 2)))
    assert gpt_moe(torch.randint(0, 14, (2, 8)))[0].size() == (2, 8, 14)
    print("gpt moe shapes correct")
    print("gpt moe parameters:", sum(p.numel() for p in gpt_moe.parameters()))


if __name__ == "__main__":
    main()
