import math
from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """PositionalEncoding module.

    This class inheriting nn.Module calculate positional encoding.
    The formulation is from original nerf paper.
    Note that the instance is callable, that aliased to forward method

    Attributes:
        embed_dim (int): count of frequency used in positional encoding
        freq (Tensor[embed_dim, float]): frequencies of each channel.
    """

    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        """Initializer

        This method initialize PositionalEncoding module.

        Args:
            embed_dim (int): Dimension of PE output.
        """
        super().__init__()
        self.embed_dim: int = embed_dim
        # Note: Original NeRF paper use pi * (2.0**t) for frequency, but for adjust
        # scale of each datasets, this implementation use (2.0**t)
        self.freq: Tensor = torch.tensor([(2.0 ** t) for t in range(self.embed_dim)])

    def forward(self, x: Tensor, scale: Optional[Tensor] = None) -> Tensor:
        """Forward

        This method calculate positional encoding of input positions.

        Args:
            x (Tensor[batch_size, 3, float32]):
                input point positions [x, y, z]
            scale (Optional[Tensor[batch_size, embed_dim * input_dim]]):
                scale of each feature

        Returns:
            Tensor[batch_size, 6*embed_dim, float32]: embed features
        """
        batch_size: int = x.shape[0]
        input_dim: int = x.shape[1]
        if scale is None:
            scale = torch.ones(
                batch_size,
                self.embed_dim * input_dim,
                dtype=torch.float32,
            )

        freq: Tensor = self.freq.reshape(-1, 1).to(x.device)
        p: Tensor = torch.matmul(freq, x.reshape(batch_size, 1, input_dim)).reshape(
            batch_size, -1
        )
        scale = scale.expand_as(p).to(x.device)
        return torch.cat([scale * torch.sin(p), scale * torch.cos(p)], 1)

    def get_lowpass_scale(self, alpha: float = 1.0, input_dim: int = 3) -> Tensor:
        """get_lowpass_scale

        This method calculate scale for low pass filter

        Args:
            alpha (float): coefficient of low pass filter
            input_dim (int): input field's dimension (always 3)

        Returns:
            Tensor[1, freq*embed_dim, float32]: embed features
        """
        with torch.set_grad_enabled(False):
            if alpha >= self.embed_dim:
                return torch.ones(1, self.embed_dim * input_dim)
            scale: Tensor = torch.ones(self.embed_dim)
            k = int(alpha)
            scale[k] = 0.5 * (1 - math.cos(math.pi * (alpha - k))) + 1e-7
            if k + 1 < self.embed_dim:
                scale[k + 1 :] = 1e-7
            return scale.unsqueeze(1).expand(-1, input_dim).reshape(1, -1)
