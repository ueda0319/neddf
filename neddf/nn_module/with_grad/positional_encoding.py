import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class PositionalEncodingGradLayer(nn.Module):
    """PositionalEncoding module.

    This class inheriting nn.Module calculate positional encoding
    with first order gradient as forward propagation.

    Attributes:
        embed_dim (int): count of frequency used in positional encoding
        freq (Tensor[embed_dim, float]): frequencies of each channel.
    """

    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        """Initializer

        This method initialize PositionalEncodingGradLayer module.

        Args:
            embed_dim (int): Dimension of PE output.
        """
        super().__init__()
        self.embed_dim: int = embed_dim
        self.freq: Tensor = torch.tensor([(2.0 ** t) for t in range(self.embed_dim)])

    def forward(
        self, x: Tensor, J: Tensor, scale: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward

        This method calculate positional encoding of input positions and their gradients.

        Args:
            x (Tensor[batch_size, 3, float32]):
                input point positions [x, y, z]
            J (Tensor[batch_size, 3, 3, float32]):
                input point positions [x, y, z]
            scale (Optional[Tensor[batch_size, embed_dim * input_dim]]):
                scale of each feature

        Returns:
            Tuple[
                Tensor[batch_size, 6*embed_dim, float32]: embed features
                Tensor[batch_size, 3, 6*embed_dim, float32]: gradients
            ]
        """
        batch_size: int = x.shape[0]
        input_dim: int = x.shape[1]
        if scale is None:
            scale = torch.ones(
                batch_size,
                self.embed_dim * input_dim,
                dtype=torch.float32,
                device=x.device,
            )

        freq: Tensor = self.freq.reshape(-1, 1).to(x.device)
        p: Tensor = torch.matmul(freq, x.reshape(batch_size, 1, input_dim)).reshape(
            batch_size, self.embed_dim * input_dim
        )
        pG = (
            J.unsqueeze(2)
            .expand(-1, -1, self.embed_dim, -1)
            .reshape(batch_size, input_dim, self.embed_dim * input_dim)
        )

        scale_y = scale.expand_as(p)
        scale_G = (
            freq.unsqueeze(0).expand(input_dim, -1, input_dim).reshape(1, input_dim, -1)
            * scale_y.unsqueeze(1)
            * pG
        )

        y = torch.cat([scale_y * torch.sin(p), scale_y * torch.cos(p)], 1)
        G = torch.cat(
            [scale_G * torch.cos(p).unsqueeze(1), -scale_G * torch.sin(p).unsqueeze(1)],
            2,
        )
        return y, G

    def withoutGrad(self, x: Tensor, scale: Optional[Tensor] = None) -> Tensor:
        """withoutGrad

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

    def get_grad_scale(self, input_dim: int = 3) -> Tensor:
        """get_grad_scale

        This method calculate scale for first order gradient

        Args:
            input_dim (int): input field's dimension (always 3)

        Returns:
            Tensor[1, freq*embed_dim, float32]: embed features
        """
        return (
            torch.reciprocal(0.5 * self.freq)
            .unsqueeze(1)
            .expand(-1, input_dim)
            .reshape(1, -1)
        )

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
