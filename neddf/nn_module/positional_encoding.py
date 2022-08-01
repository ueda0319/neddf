import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """PositionalEncoding module.

    This class inheriting nn.Module calculate positional encoding.
    The formulation is from original nerf paper.
    Note that the instance is callable, that aliased to forward method

    Attributes:
        embed_dim (int): count of frequency used in positional encoding
    """

    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward

        This method calculate positional encoding of input positions.

        Args:
            x (Tensor[batch_size, 3, float32]):
                input point positions [x, y, z]

        Returns:
            Tensor[batch_size, 6*embed_dim, float32]: embed features
        """
        batch_size: int = x.shape[0]
        input_dim: int = x.shape[1]
        # Note: Original NeRF paper use pi * (2.0**t) for frequency, but for adjust
        # scale of each datasets, this implementation use (2.0**t)
        freq: Tensor = (
            torch.tensor([(2.0 ** t) for t in range(self.embed_dim)])
            .reshape(-1, 1)
            .to(x.device)
        )
        p: Tensor = torch.matmul(freq, x.reshape(batch_size, 1, input_dim)).reshape(
            batch_size, -1
        )
        return torch.cat([torch.sin(p), torch.cos(p)], 1)
