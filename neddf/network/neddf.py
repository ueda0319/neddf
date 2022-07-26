from typing import Callable, Dict, Final, List, Literal, Optional, Tuple

import torch
from neddf.network.base_neuralfield import BaseNeuralField
from neddf.nn_module import PositionalEncoding, tanhExp
from neddf.nn_module.with_grad import (
    LeakyReLUGradFunction,
    LinearGradLayer,
    PositionalEncodingGradLayer,
    ReLUGradFunction,
    SigmoidGradFunction,
    SoftplusGradFunction,
    TanhExpGradFunction,
)
from neddf.ray import Sampling
from torch import Tensor, nn
from torch.nn.functional import leaky_relu, relu

ActivationType = Literal["LeakyReLU", "ReLU", "tanhExp"]


class NeDDF(BaseNeuralField):
    """NeDDF.

    This class inheriting BaseNeuralField execute network inference.
    The network archtecture is define in NeDDF paper.
    (https://arxiv.org/abs/2207.14455)

    Attributes:
        skips (List[int]): Skip connection layer ids.
        activation (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]):
            Activation function with first order gradients for hidden layers.
        density_activation (Callable[Tensor, Tensor]):
            Activation function for density output layer.
        pe_pos (PositionalEncodingGradLayer):
            Layer of PE with first order gradients.
        pe_dir (PositionalEncoding): Layer of PE.
        layers_ddf (ModuleList): Layers for distance field network.
        layers_col (ModuleList): Layers for color field network.
        layer_ddf_out (LinearGradLayer): Output layer for distance.
        layer_aux_out (LinearGradLayer): Output layer for aux. gradient.
        layer_col_out (LinearGradLayer): Output layer for color.
        d_near (float): minimal value of distance field.
        aux_grad_scale (float):
            Scale for aux. gradient.
            It is configured in warmup.
        lowpass_alpha (float): Coefficient of lowpass filter, used in warmup.
        lowpass_alpha_offset (float): Offset of lowpass_alpha
        penalty_weight (Dict[str, float]): Weights for field constraints penalty.
    """

    def __init__(
        self,
        embed_pos_rank: int = 10,
        embed_dir_rank: int = 4,
        ddf_layer_count: int = 8,
        ddf_layer_width: int = 256,
        col_layer_count: int = 8,
        col_layer_width: int = 256,
        activation_type: ActivationType = "tanhExp",
        density_activation_type: ActivationType = "ReLU",
        d_near: float = 0.01,
        lowpass_alpha_offset: float = 10.0,
        skips: Optional[List[int]] = None,
        penalty_weight: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initializer

        This method initialize NeDDF module.

        Args:
            embed_pos_rank (int): Rank for positional encoding of position.
            embed_dir_rank (int): Rank for positional encoding of direction.
            ddf_layer_count (int): count of layers of distance field.
            ddf_layer_width (int): dimension of hidden layers of distance field.
            col_layer_count (int): count of layers of color field.
            col_layer_width (int): dimension of hidden layers of color field.
            activation_type (ActivationType): Type of activation function.
            d_near (float): Minimal value of distance field.
            lowpass_alpha_offset (float):
                Offset of progressive training in lowpass
                Set lowpass_alpha_offset=embed_pos_rank to disable it.
            skips (List[int]): skip connection layer index start with 0.
            penalty_weight (Dict[str, float]):
                Weights for field constraints penalty.

        """
        super().__init__()
        # calculate mlp input dimensions after positional encoding
        input_ddf_dim: Final[int] = embed_pos_rank * 6
        input_col_dim: Final[int] = (
            (embed_pos_rank + embed_dir_rank) * 6 + 3 + ddf_layer_width
        )

        # catch default params with referencial types
        if skips is None:
            skips = [4]
        self.skips = skips

        activation_types_with_grad: Final[
            Dict[str, Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]]
        ] = {
            "LeakyReLU": LeakyReLUGradFunction.apply,
            "ReLU": ReLUGradFunction.apply,
            "tanhExp": TanhExpGradFunction.apply,
        }
        activation_types: Final[Dict[str, Callable[[Tensor], Tensor]]] = {
            "LeakyReLU": leaky_relu,
            "ReLU": relu,
            "tanhExp": tanhExp.apply,
        }

        self.activation: Callable[
            [Tensor, Tensor], Tuple[Tensor, Tensor]
        ] = activation_types_with_grad[activation_type]
        self.density_activation: Callable[[Tensor], Tensor] = activation_types[
            density_activation_type
        ]

        # create positional encoding layers
        self.pe_pos: PositionalEncodingGradLayer = PositionalEncodingGradLayer(
            embed_pos_rank
        )
        self.pe_dir: PositionalEncoding = PositionalEncoding(embed_dir_rank)

        # create layers
        layers_ddf: List[nn.Module] = []
        layers_col: List[nn.Module] = []
        layers_ddf.append(LinearGradLayer(input_ddf_dim, ddf_layer_width))
        for layer_id in range(ddf_layer_count - 2):
            if layer_id in skips:
                layers_ddf.append(
                    LinearGradLayer(ddf_layer_width + input_ddf_dim, ddf_layer_width)
                )
            else:
                layers_ddf.append(LinearGradLayer(ddf_layer_width, ddf_layer_width))
        layers_col.append(LinearGradLayer(input_col_dim, col_layer_width))
        for _ in range(col_layer_count - 2):
            layers_col.append(LinearGradLayer(col_layer_width, col_layer_width))
        # Set layers as optimization targets
        self.layers_ddf = nn.ModuleList(layers_ddf)
        self.layers_col = nn.ModuleList(layers_col)
        self.layer_ddf_out = LinearGradLayer(ddf_layer_width, 1)
        self.layer_aux_out = LinearGradLayer(ddf_layer_width, 1)
        self.layer_col_out = LinearGradLayer(ddf_layer_width, 3)

        self.d_near: float = d_near
        self.aux_grad_scale: float = 1.1
        self.distance_range_max: float = 2.0
        self.lowpass_alpha_offset: float = lowpass_alpha_offset
        self.lowpass_alpha: float = lowpass_alpha_offset
        if penalty_weight is None:
            penalty_weight = {
                "constraints_aux_grad": 0.05,
                "constraints_dDdt": 0.05,
                "constraints_color": 0.01,
                "range_distance": 1.0,
                "range_aux_grad": 1.0,
            }
        self.penalty_weight: Dict[str, float] = penalty_weight

    def forward(
        self,
        sampling: Sampling,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This method take density-distance field (distance + density + color) with MLP.

        Args:
            sampling (Sampling[batch_size, sampling, 3]): Input samplings

        Returns:
            Dict[str, Tensor]{
                'distance' (Tensor[batch_size, 1, float32]): distance of each input
                'density' (Tensor[batch_size, 1, float32]): density of each input
                'color' (Tensor[batch_size, 3, float32]): rgb color of each input
                'fields_penalty' (Tensor[batch_size, 1, float32]): penalty of field constraints
                'aux_grad' (Tensor[batch_size, 1, float32]): aux. gradient of each input
            }
        """
        batch_size: Final[int] = sampling.sample_pos.shape[0]
        sampling_size: Final[int] = sampling.sample_pos.shape[1]
        device: torch.device = sampling.sample_pos.device

        sample_pos_grad: Tensor = (
            torch.eye(3)
            .to(device)
            .unsqueeze(0)
            .expand(batch_size * sampling_size, -1, -1)
        )
        # scale PE with distance field to graditent becale same scale
        pe_grad_scale: Tensor = self.pe_pos.get_grad_scale().to(device)
        # scale PE with lowpass
        pe_lowpass_scale: Tensor = self.pe_pos.get_lowpass_scale(self.lowpass_alpha).to(
            device
        )
        # get weight of PE from sampling size
        pe_weights = sampling.get_pe_weights(self.pe_pos.freq.to(device))
        embed_pos_scaled: Tuple[Tensor, Tensor] = self.pe_pos(
            sampling.sample_pos.view(-1, 3),
            sample_pos_grad,
            pe_grad_scale * pe_lowpass_scale * pe_weights,
        )
        embed_pos: Tensor = self.pe_pos(
            sampling.sample_pos.view(-1, 3),
            sample_pos_grad,
            pe_lowpass_scale * pe_weights,
        )
        embed_dir: Tensor = self.pe_dir(sampling.sample_dir.view(-1, 3))

        hx: Tensor = embed_pos_scaled[0]
        hJ: Tensor = embed_pos_scaled[1]
        for layer_id, layer in enumerate(self.layers_ddf):
            hx, hJ = layer(hx, hJ)
            hx, hJ = self.activation(hx, hJ)
            if layer_id in self.skips:
                hx = torch.cat([embed_pos_scaled[0], hx], dim=1)
                hJ = torch.cat([embed_pos_scaled[1], hJ], dim=2)
        ddf_out, ddf_outJ = self.layer_ddf_out(hx, hJ)
        distance_tuple = SoftplusGradFunction.apply(ddf_out, ddf_outJ)
        distance: Tensor = distance_tuple[0] + self.d_near
        distance_grad: Tensor = distance_tuple[1][:, :, 0]

        aux_out, aux_outJ = self.layer_aux_out(hx, hJ)
        aux_grad_tuple: Tuple[Tensor, Tensor] = SigmoidGradFunction.apply(
            aux_out, aux_outJ
        )
        aux_grad: Tensor = self.aux_grad_scale * aux_grad_tuple[0]
        aux_gg: Tensor = self.aux_grad_scale * aux_grad_tuple[1][:, :, 0]
        features: Tensor = hx
        featuresJ: Tensor = hJ

        nabla_distance: Tensor = torch.cat([distance_grad, aux_grad], dim=1)
        distance_grad_norm: Tensor = torch.norm(distance_grad, dim=1)[:, None]  # type: ignore

        # calculate density from nabla_distance and inverse distance
        dDdt: Tensor = torch.norm(nabla_distance, dim=1)[:, None]  # type: ignore
        distance_inv: Tensor = torch.reciprocal(distance)
        density: Tensor = self.density_activation(distance_inv * (1 - dDdt))
        norm_dir = torch.reciprocal(distance_grad_norm + 1e-7) * distance_grad

        hx = torch.cat([embed_pos[0], embed_dir, norm_dir.detach(), features], dim=1)
        hJ = torch.cat(
            [
                embed_pos[1],
                torch.zeros(batch_size * sampling_size, 3, embed_dir.shape[1] + 3).to(
                    device
                ),
                featuresJ,
            ],
            dim=2,
        )
        for layer in self.layers_col:
            hx, hJ = layer(hx, hJ)
            hx, hJ = self.activation(hx, hJ)
        color, colorJ = self.layer_col_out(hx, hJ)

        # for calculate penalties of field constraints
        penalties: Dict[str, Tensor] = {}
        # penalty for aux. gradient
        d2D_dwdt = torch.sum(aux_gg * norm_dir, 1)[:, None]
        d2D_dwdt_rest = 3 * aux_grad * distance_inv.detach()
        ag_penalty_scale = (
            aux_grad.detach() * distance_grad_norm.detach() * distance.detach()
        )
        penalties["constraints_aux_grad"] = ag_penalty_scale * torch.square(
            d2D_dwdt - d2D_dwdt_rest
        )
        # penalty for dDdt (dDdt < 1.0)
        penalties["constraints_dDdt"] = torch.square(relu(-1.0 + dDdt))

        # penalty for values which over ranges
        # note that sigmoid(-4.6) and softplus(-4.6) is simillar to 0.01
        # -4.6 < (distance before softplus) < 2.0
        penalties["range_distance"] = torch.square(
            relu(-4.6 - ddf_out) + relu(-self.distance_range_max + ddf_out)
        )
        # -4.6 < (aux_grad before softplus) < 4.6
        penalties["range_aux_grad"] = torch.square(
            relu(-4.6 - aux_out) + relu(-4.6 + aux_out)
        )
        # 0.0 < (color) < 1.0
        penalties["range_color"] = torch.square(
            relu(-0.0 - color) + relu(-1.0 + color)
        ).sum(1, keepdim=True)

        # penalty for color field
        penalties["constraints_color"] = (
            (colorJ * distance_grad.detach().unsqueeze(2))
            .sum(1)
            .square()
            .sum(1, keepdim=True)
        )
        # composit penalties
        for key in penalties:
            if key not in self.penalty_weight:
                continue
            penalties[key] = penalties[key] * self.penalty_weight[key]
        fields_penalty = torch.sum(torch.stack(list(penalties.values()), dim=2), dim=2)

        output_dict: Dict[str, Tensor] = {
            "distance": distance.view(batch_size, sampling_size),
            "density": density.view(batch_size, sampling_size),
            "color": color.view(batch_size, sampling_size, 3),
            "fields_penalty": fields_penalty.view(batch_size, sampling_size),
            "aux_grad": aux_grad.view(batch_size, sampling_size),
        }
        return output_dict

    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        if iter == -1:
            self.aux_grad_scale = 1.1
            self.distance_range_max = 2.0
            self.lowpass_alpha = self.pe_pos.embed_dim
        else:
            self.aux_grad_scale = min(1.1, max(0.01, 0.0001 * iter))
            self.distance_range_max = min(2.0, 2.0 + 0.0001 * iter)
            self.lowpass_alpha = self.lowpass_alpha_offset + 0.001 * iter
