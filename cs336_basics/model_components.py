import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_feat: int, out_feat: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_feat, in_feat), device=device, dtype=dtype))
        std = np.sqrt(2.0 / (in_feat + out_feat))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def set_weights(self, weight: torch.Tensor) -> None:
        self.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        >>> layer = Linear(4, 2)
        >>> layer.set_weights(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        >>> x = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])
        >>> layer.forward(x).data()
        tensor([[[1., 2.],
                 [5., 6.]]])
        """
        return einsum(x, self.weight, "... din, dout din -> ... dout")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        self.embeddings.data = embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones((dim,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_scaled = x_normed * self.scale
        return x_scaled.to(in_dtype)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = d_model * 8 // 3
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model)))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model)))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff)))

    def set_parameters(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> None:
        self.w1.data = w1
        self.w2.data = w2
        self.w3.data = w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part1 = SiLU(einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff"))
        part2 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        combined = part1 * part2
        output = einsum(combined, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "sin_t",
            torch.zeros((max_seq_len, d_k), device=device, dtype=dtype),
            persistent=False,
        )
        self.sin_t.data = torch.sin(
            torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
            / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        ).repeat_interleave(2, dim=1)
        self.register_buffer(
            "cos_t",
            torch.zeros((max_seq_len, d_k), device=device, dtype=dtype),
            persistent=False,
        )
        self.cos_t.data = torch.cos(
            torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
            / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        ).repeat_interleave(2, dim=1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin_t = self.sin_t[token_positions]
        cos_t = self.cos_t[token_positions]

        x_pairs = rearrange(x, "... d_k -> ... (d_k 2) 2")
        x_swapped = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1).flatten(-2)

        x_rotated = (x * cos_t) + (x_swapped * sin_t)

        return x_rotated


def Softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def ScaledDotProductAttention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_K = query.shape[-1]
    scores = einsum(query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / np.sqrt(d_K)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = Softmax(scores, dim=-1)
    output = einsum(attn_weights, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.rope = rope

        # 形状应该是 (num_heads, d_k, d_model)
        self.wq = nn.Parameter(torch.empty((num_heads * self.d_k, d_model)))
        self.wk = nn.Parameter(torch.empty((num_heads * self.d_k, d_model)))
        self.wv = nn.Parameter(torch.empty((num_heads * self.d_v, d_model)))
        self.wo = nn.Parameter(torch.empty((d_model, num_heads * self.d_v)))

        self.rope = RotaryPositionalEmbedding(
            theta=10000.0,
            d_k=self.d_k,
            max_seq_len=2048,
        )

    def set_parameters(self, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor) -> None:
        self.wq.data = wq
        self.wk.data = wk
        self.wv.data = wv
        self.wo.data = wo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = rearrange(self.wq, " (num_heads d_k) d_model -> num_heads d_k d_model", num_heads=self.num_heads)
        query = einsum(x, wq, "... seq_len d_model, num_heads d_k d_model -> ... seq_len num_heads d_k")
        wk = rearrange(self.wk, " (num_heads d_k) d_model -> num_heads d_k d_model", num_heads=self.num_heads)
        key = einsum(x, wk, "... seq_len d_model, num_heads d_k d_model -> ... seq_len num_heads d_k")

        if self.rope:
            query = self.rope(
                query,
                token_positions=torch.arange(x.shape[-2], device=x.device),
            )
            key = self.rope(
                key,
                token_positions=torch.arange(x.shape[-2], device=x.device),
            )
        wv = rearrange(self.wv, " (num_heads d_v) d_model -> num_heads d_v d_model", num_heads=self.num_heads)
        value = einsum(x, wv, "... seq_len d_model, num_heads d_v d_model -> ... seq_len num_heads d_v")

        mask = torch.triu(torch.ones((x.shape[-2], x.shape[-2]), device=x.device), diagonal=1).bool()

        attn_weight = ScaledDotProductAttention(query, key, value, mask)
        attn_weight = rearrange(attn_weight, "... seq_len num_heads d_v -> ... seq_len (num_heads d_v)")

        output = einsum(
            attn_weight, self.wo, "... seq_len (num_heads d_v), d_model (num_heads d_v) -> ... seq_len d_model"
        )

        return output


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
