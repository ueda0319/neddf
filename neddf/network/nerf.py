from typing import Callable, Dict, Final, List, Optional

import torch
from torch import Tensor, nn

from neddf.nn_module import PositionalEncoding


class NeRF(nn.Module):
    def __init__(
        self,
        embed_pos_rank: int = 10,
        embed_dir_rank: int = 4,
        layer_count: int = 8,
        layer_width: int = 256,
        activation_type: str = "ReLU",
        skips: Optional[List[int]] = None,
    ) -> None:
        """Initializer

        This method initialize NeRF module.

        Args:
            input_pos_dim (int): dimension of each imput positions
            input_dir_dim (int): dimension of each imput directions
            layer_count (int): count of layers
            layer_width (int): dimension of hidden layers
            activation_type (str): activation function
            skips (List[int]): skip connection layer index start with 0

        """
        super().__init__()
        # calculate mlp input dimensions after positional encoding
        input_pos_dim: Final[int] = embed_pos_rank * 6
        input_dir_dim: Final[int] = embed_dir_rank * 6

        # catch default params with referencial types
        if skips is None:
            skips = [4]
        self.skips = skips

        activation_types: Final[Dict[str, Callable[[Tensor], Tensor]]] = {
            "ReLU": nn.ReLU()
        }

        self.activation = activation_types[activation_type]

        # create positional encoding layers
        self.pe_pos: PositionalEncoding = PositionalEncoding(embed_pos_rank)
        self.pe_dir: PositionalEncoding = PositionalEncoding(embed_dir_rank)

        # create layers
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_pos_dim, layer_width))
        for layer_id in range(layer_count - 1):
            if layer_id in skips:
                layers.append(nn.Linear(layer_width + input_pos_dim, layer_width))
            else:
                layers.append(nn.Linear(layer_width, layer_width))
        # Set layers as optimization targets
        self.layers = nn.ModuleList(layers)
        # Output layer of density
        self.outL_density = nn.Linear(layer_width, 1)
        # Output layer of color
        self.outL_color = nn.Sequential(
            nn.Linear(layer_width + input_dir_dim, layer_width // 2),
            nn.ReLU(),
            nn.Linear(layer_width // 2, 3),
        )

    def forward(
        self,
        input_pos: Tensor,
        input_dir: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This method take radiance field (density + color) with standard MLP.

        Args:
            input_pos (Tensor[batch_size, sampling, 3, float32]):
                input point positions
                If you need to use PE, please enter the tensor you have already applied PE.
            input_dir (Tensor[batch_size, 3, float32]):
                input point positions
                If you need to use PE, please enter the tensor you have already applied PE.

        Returns:
            Dict[str, Tensor]{
                'density' (Tensor[batch_size, 1, float32]): density of each input
                'color' (Tensor[batch_size, 3, float32]): rgb color of each input
            }

        Notes:
            Apply range limit function in volume rendering step
                (original paper use relu for density, sigmoid for color)
            In original paper, final hidden layer use no activation, but
                original implementation use activation.
                Since a hidden layer without activation does not increase the
                amount of information (ideally), this implementation uses activation.
            In original paper, dir_feature take one additional dense layer without activation,
                but our implementation remove it since
                (since it is possible to create M3 that reproduces M2 [M1 x, d] with M3[x, d])

        """
        batch_size: Final[int] = input_pos.shape[0]
        sampling: Final[int] = input_pos.shape[1]

        embed_pos: Tensor = self.pe_pos(input_pos.reshape(-1, 3))
        embed_dir: Tensor = self.pe_dir(input_dir.reshape(-1, 3))

        hx: Tensor = embed_pos
        for layer_id, layer in enumerate(self.layers):
            hx = self.activation(layer(hx))
            if layer_id in self.skips:
                hx = torch.cat([hx, embed_pos], dim=1)
        density = self.outL_density(hx)

        dir_feature = torch.cat([hx, embed_dir], dim=1)
        color = self.outL_color(dir_feature)

        output_dict: Dict[str, Tensor] = {
            "density": density.reshape(batch_size, sampling),
            "color": color.reshape(batch_size, sampling, 3),
        }
        return output_dict
