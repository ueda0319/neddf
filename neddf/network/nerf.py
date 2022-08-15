from typing import Callable, Dict, Final, List, Literal, Optional

import torch
from neddf.network.base_neuralfield import BaseNeuralField
from neddf.nn_module import PositionalEncoding, tanhExp
from neddf.ray import Sampling
from torch import Tensor, nn
from torch.nn.functional import leaky_relu, relu

ActivationType = Literal["LeakyReLU", "ReLU", "tanhExp"]


class NeRF(BaseNeuralField):
    """NeRF.

    This class inheriting BaseNeuralField execute network inference.
    The network archtecture is define in NeRF paper.
    (https://arxiv.org/abs/2003.08934)

    Attributes:
        skips (List[int]): Skip connection layer ids.
        activation (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]):
            Activation function for hidden layers.
        density_activation (Callable[[Tensor], Tensor]):
            Activation function for density output layer.
        pe_pos (PositionalEncoding): Layer of position's PE.
        pe_dir (PositionalEncoding): Layer of direction's PE.
        layers (ModuleList): Layers for radiance field network.
        outL_density (nn.Linear): Output layer for density.
        lowpass_alpha (float): Coefficient of lowpass filter, used in warmup.
        lowpass_alpha_offset (float): Offset of lowpass_alpha
    """

    def __init__(
        self,
        embed_pos_rank: int = 10,
        embed_dir_rank: int = 4,
        layer_count: int = 8,
        layer_width: int = 256,
        activation_type: ActivationType = "ReLU",
        density_activation_type: ActivationType = "ReLU",
        skips: Optional[List[int]] = None,
        lowpass_alpha_offset: float = 10.0,
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
            lowpass_alpha_offset (float):
                Offset of progressive training in lowpass
                Set lowpass_alpha_offset=embed_pos_rank to disable it.

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
            "LeakyReLU": leaky_relu,
            "ReLU": relu,
            "tanhExp": tanhExp.apply,
        }

        self.activation = activation_types[activation_type]
        self.density_activation: Callable[[Tensor], Tensor] = activation_types[
            density_activation_type
        ]

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
        self.lowpass_alpha_offset: float = lowpass_alpha_offset
        self.lowpass_alpha: float = lowpass_alpha_offset

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
        batch_size: Final[int] = sampling.sample_pos.shape[0]
        sampling_size: Final[int] = sampling.sample_pos.shape[1]
        device: torch.device = sampling.sample_pos.device

        # scale PE with lowpass
        pe_lowpass_scale: Tensor = self.pe_pos.get_lowpass_scale(self.lowpass_alpha).to(
            device
        )
        # get weight of PE from sampling size
        pe_weights = sampling.get_pe_weights(self.pe_pos.freq).to(device)
        embed_pos: Tensor = self.pe_pos(
            sampling.sample_pos.reshape(-1, 3),
            pe_lowpass_scale * pe_weights,
        )
        embed_dir: Tensor = self.pe_dir(sampling.sample_dir.reshape(-1, 3))

        hx: Tensor = embed_pos
        for layer_id, layer in enumerate(self.layers):
            hx = self.activation(layer(hx))
            if layer_id in self.skips:
                hx = torch.cat([hx, embed_pos], dim=1)
        density = self.density_activation(self.outL_density(hx))

        dir_feature = torch.cat([hx, embed_dir], dim=1)
        color = self.outL_color(dir_feature)

        output_dict: Dict[str, Tensor] = {
            "density": density.reshape(batch_size, sampling_size),
            "color": color.reshape(batch_size, sampling_size, 3),
        }
        return output_dict

    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        if iter == -1:
            self.lowpass_alpha = self.pe_pos.embed_dim
        else:
            self.lowpass_alpha = self.lowpass_alpha_offset + 0.001 * iter
