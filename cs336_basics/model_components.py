import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_feat, in_feat), device=device, dtype=dtype)
        )
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
        self.embeddings = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embeddings, std=1, a=-3, b=3)

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        self.embeddings.data = embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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

    def set_parameter(self, scale: torch.Tensor) -> None:
        self.scale.data = scale


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model)))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model)))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff)))

    def set_parameters(
        self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
    ) -> None:
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

        x_pairs = rearrange(x, "... (d c) -> ... d c", c=2)
        x_swapped = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1).flatten(-2)

        x_rotated = (x * cos_t) + (x_swapped * sin_t)

        return x_rotated


def Softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def SDPA(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_K = key.shape[-1]
    scores = einsum(
        query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / np.sqrt(d_K)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = Softmax(scores, dim=-1)
    output = einsum(
        attn_weights,
        value,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
    )
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Parameter(torch.empty((d_model, d_model)))
        self.Wk = nn.Parameter(torch.empty((d_model, d_model)))
        self.Wv = nn.Parameter(torch.empty((d_model, d_model)))
        self.Wo = nn.Parameter(torch.empty((d_model, d_model)))

        if rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=self.Wq.device,
                dtype=self.Wq.dtype,
            )
            self.token_positions = token_positions
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(
            torch.ones((x.shape[-2], x.shape[-2]), device=x.device), diagonal=0
        ).bool()

        Q = einsum(
            x,
            self.Wq,
            "... seq_length din, ... din dout -> ... seq_length dout",
        )
        K = einsum(
            x,
            self.Wk,
            "... seq_length din, ... din dout -> ... seq_length dout",
        )
        V = einsum(
            x,
            self.Wv,
            "... seq_length din, ... din dout -> ... seq_length dout",
        )

        Q = rearrange(
            Q,
            "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k",
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            seq_length = x.shape[-2]
            if self.token_positions is None:
                token_positions = torch.arange(seq_length, device=x.device)
            else:
                token_positions = self.token_positions
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn_output = SDPA(Q, K, V, mask=mask)
        attn_output = rearrange(
            attn_output,
            "... num_heads seq_length d_k -> ... seq_length (num_heads d_k)",
        )

        output = einsum(
            attn_output,
            self.Wo,
            "... seq_length din, ... din dout -> ... seq_length dout",
        )
        return output

    def set_parameters(
        self, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor, Wo: torch.Tensor
    ) -> None:
        self.Wq.data = Wq.T
        self.Wk.data = Wk.T
        self.Wv.data = Wv.T
        self.Wo.data = Wo.T


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: bool,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.rms1 = RMSNorm(dim=d_model)
        self.ffn = SwiGLU(d_model=d_ff)
        self.rms2 = RMSNorm(dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape (batch_size, seq_length, d_model)

        returns: Output tensor of shape (batch_size, seq_length, d_model)
        """
        a = self.rms1(x)
        b = self.mha(a)
        c = x + b
        d = self.rms2(c)
        e = self.ffn(d)
        output = c + e
        return output

    def set_parameters(self, state: dict):
        self.mha.set_parameters(
            Wq=state["attn.q_proj.weight"],
            Wk=state["attn.k_proj.weight"],
            Wv=state["attn.v_proj.weight"],
            Wo=state["attn.output_proj.weight"],
        )

        self.ffn.set_parameters(
            w1=state["ffn.w1.weight"],
            w2=state["ffn.w2.weight"],
            w3=state["ffn.w3.weight"],
        )

        self.rms1.set_parameter(
            scale=state["ln1.weight"],
        )
        self.rms2.set_parameter(
            scale=state["ln2.weight"],
        )


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()

        self.token_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=True,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
            )
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        self.norm = RMSNorm(dim=d_model)
        self.linear = Linear(in_feat=d_model, out_feat=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(x)
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        normed = self.norm(embeddings)
        logits = self.linear(normed)
        return logits

    def set_parameters(self, state: dict) -> None:
        self.token_embedding.set_embeddings(state["token_embeddings.weight"])

        for i, block in enumerate(self.transformer_blocks):
            block_state = {}
            for key in state.keys():
                if key.startswith(f"layers.{i}."):
                    block_state[key[len(f"layers.{i}.") :]] = state[key]
            block.set_parameters(block_state)

        self.norm.set_parameter(state["ln_final.weight"])
        self.linear.set_weights(state["lm_head.weight"])


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
