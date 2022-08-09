from typing import Callable, Dict, Final, List, Literal, Optional

import torch
from neddf.network.base_neuralfield import BaseNeuralField
from neddf.nn_module import PositionalEncoding, tanhExp
from neddf.ray import Sampling
from torch import Tensor, nn

ActivationType = Literal["ReLU", "tanhExp"]


class NeuS(BaseNeuralField):
    """NeuS.

    This class inheriting BaseNeuralField execute network inference.
    The network archtecture is define in NeuS paper.
    (https://arxiv.org/abs/2106.10689)

    Attributes:
        skips (List[int]): Skip connection layer ids.
        activation (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]):
            Activation function with first order gradients.
        pe_pos (PositionalEncoding): Layer of position's PE.
        pe_dir (PositionalEncoding): Layer of direction's PE.
        layers_sdf (nn.ModuleList): Layers for distance field network.
        layers_col (nn.ModuleList): Layers for color field network.
        variance (nn.Parameter): Parameter of density variance
    """

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
            embed_pos_rank (int): Rank for positional encoding of position.
            embed_dir_rank (int): Rank for positional encoding of direction.
            sdf_layer_count (int): count of layers of distance field.
            sdf_layer_width (int): dimension of hidden layers of distance field.
            col_layer_count (int): count of layers of color field.
            col_layer_width (int): dimension of hidden layers of color field.
            activation_type (ActivationType): Type of activation function.
            init_variance (float): Initial variance for convert distance to density.
            skips (List[int]): skip connection layer index start with 0.

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
            "ReLU": nn.ReLU(),
            "tanhExp": tanhExp.apply,
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
        sampling: Sampling,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This method take radiance field (density + color) with standard MLP.

        Args:
            sampling (Sampling[batch_size, sampling, 3]): Input samplings

        Returns:
            Dict[str, Tensor]{
                'density' (Tensor[batch_size, 1, float32]): density of each input
                'color' (Tensor[batch_size, 3, float32]): rgb color of each input
            }
        """
        batch_size: Final[int] = sampling.sample_pos.shape[0]
        sampling_size: Final[int] = sampling.sample_pos.shape[1]

        sampling.sample_pos.requires_grad_(True)
        embed_pos: Tensor = self.pe_pos(sampling.sample_pos.reshape(-1, 3))
        embed_dir: Tensor = self.pe_dir(sampling.sample_dir.reshape(-1, 3))

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
            inputs=sampling.sample_pos,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(-1, 3)

        hx = torch.cat(
            [sampling.sample_pos.reshape(-1, 3), embed_dir, gradients, sdf_feature],
            dim=1,
        )
        for layer in self.layers_col:
            hx = self.activation(layer(hx))
        color: Tensor = hx

        ex: Tensor = torch.exp(-self.variance * 10.0 * sdf)
        density: Tensor = (
            self.variance * 10.0 * ex * torch.reciprocal(torch.square(1 + ex))
        )

        output_dict: Dict[str, Tensor] = {
            "sdf": sdf.reshape(batch_size, sampling_size),
            "density": density.reshape(batch_size, sampling_size),
            "color": color.reshape(batch_size, sampling_size, 3),
        }
        return output_dict
