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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
