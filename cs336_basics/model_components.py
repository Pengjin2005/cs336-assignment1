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
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def set_weights(self, weight: torch.Tensor) -> None:
        self.weight.data = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
