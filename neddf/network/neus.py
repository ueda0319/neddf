from typing import Callable, Dict, Final, List, Optional

import torch
from neddf.network.base_neuralfield import BaseNeuralField
from neddf.nn_module import PositionalEncoding
from torch import Tensor, nn


class NeuS(BaseNeuralField):
    def __init__(
        self,
        embed_pos_rank: int = 6,
        embed_dir_rank: int = 4,
        sdf_layer_count: int = 8,
        sdf_layer_width: int = 256,
        col_layer_count: int = 8,
        col_layer_width: int = 256,
        activation_type: str = "ReLU",
        init_variance: float = 0.3,
        skips: Optional[List[int]] = None,
    ) -> None:
        """Initializer

        This method initialize NeuS module.

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
        input_sdf_dim: Final[int] = embed_pos_rank * 6
        input_col_dim: Final[int] = 6 + embed_dir_rank * 6 + sdf_layer_width

        # catch default params with referencial types
        if skips is None:
            skips = [4]
        self.skips = skips

        activation_types: Final[Dict[str, Callable[[Tensor], Tensor]]] = {
            "ReLU": nn.ReLU()
        }

        self.activation: Callable[[Tensor], Tensor] = activation_types[activation_type]

        # create positional encoding layers
        self.pe_pos: PositionalEncoding = PositionalEncoding(embed_pos_rank)
        self.pe_dir: PositionalEncoding = PositionalEncoding(embed_dir_rank)

        # create layers
        layers_sdf: List[nn.Module] = []
        layers_col: List[nn.Module] = []
        layers_sdf.append(nn.Linear(input_sdf_dim, sdf_layer_width))
        for layer_id in range(sdf_layer_count - 1):
            if layer_id in skips:
                layers_sdf.append(
                    nn.Linear(sdf_layer_width + input_sdf_dim, sdf_layer_width)
                )
            else:
                layers_sdf.append(nn.Linear(sdf_layer_width, sdf_layer_width))
        layers_col.append(nn.Linear(input_col_dim, col_layer_width))
        for _ in range(col_layer_count - 1):
            layers_col.append(nn.Linear(col_layer_width, col_layer_width))
        layers_col.append(nn.Linear(col_layer_width, 3))
        # Set layers as optimization targets
        self.layers_sdf = nn.ModuleList(layers_sdf)
        self.layers_col = nn.ModuleList(layers_col)

        # Variance for convert from sdf to density
        self.variance: nn.Parameter = nn.Parameter(torch.tensor(init_variance))

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

        input_pos.requires_grad_(True)
        embed_pos: Tensor = self.pe_pos(input_pos.reshape(-1, 3))
        embed_dir: Tensor = self.pe_dir(input_dir.reshape(-1, 3))

        hx: Tensor = embed_pos
        for layer_id, layer in enumerate(self.layers_sdf):
            hx = self.activation(layer(hx))
            if layer_id in self.skips:
                hx = torch.cat([hx, embed_pos], dim=1)
        sdf: Tensor = hx[:, :1]
        sdf_feature: Tensor = hx

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients take normal vector
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=input_pos,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(-1, 3)

        hx = torch.cat(
            [input_pos.reshape(-1, 3), embed_dir, gradients, sdf_feature], dim=1
        )
        for layer in self.layers_col:
            hx = self.activation(layer(hx))
        color: Tensor = hx

        ex: Tensor = torch.exp(-self.variance * 10.0 * sdf)
        density: Tensor = (
            self.variance * 10.0 * ex * torch.reciprocal(torch.square(1 + ex))
        )

        output_dict: Dict[str, Tensor] = {
            "sdf": sdf.reshape(batch_size, sampling),
            "density": density.reshape(batch_size, sampling),
            "color": color.reshape(batch_size, sampling, 3),
        }
        return output_dict
