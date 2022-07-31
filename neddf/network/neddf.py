from typing import Callable, Dict, Final, List, Optional

import torch
from torch import Tensor, nn
from torch.nn.functional import relu, sigmoid, softplus

from neddf.network.base_neuralfield import BaseNeuralField
from neddf.nn_module import ScaledPositionalEncoding


class NeDDF(BaseNeuralField):
    def __init__(
        self,
        embed_pos_rank: int = 6,
        embed_dir_rank: int = 4,
        ddf_layer_count: int = 8,
        ddf_layer_width: int = 256,
        col_layer_count: int = 8,
        col_layer_width: int = 256,
        activation_type: str = "ReLU",
        d_near: float = 0.01,
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
        input_col_dim: Final[int] = 6 + embed_dir_rank * 6 + ddf_layer_width

        # catch default params with referencial types
        if skips is None:
            skips = [4]
        self.skips = skips

        activation_types: Final[Dict[str, Callable[[Tensor], Tensor]]] = {
            "ReLU": nn.ReLU()
        }

        self.activation: Callable[[Tensor], Tensor] = activation_types[activation_type]

        # create positional encoding layers
        self.pe_pos: ScaledPositionalEncoding = ScaledPositionalEncoding(embed_pos_rank)
        self.pe_dir: ScaledPositionalEncoding = ScaledPositionalEncoding(embed_dir_rank)

        # create layers
        layers_ddf: List[nn.Module] = []
        layers_col: List[nn.Module] = []
        layers_ddf.append(nn.Linear(input_sdf_dim, ddf_layer_width))
        for layer_id in range(ddf_layer_count - 1):
            if layer_id in skips:
                layers_ddf.append(
                    nn.Linear(ddf_layer_width + input_sdf_dim, ddf_layer_width)
                )
            else:
                layers_ddf.append(nn.Linear(ddf_layer_width, ddf_layer_width))
        layers_col.append(nn.Linear(input_col_dim, col_layer_width))
        for layer_id in range(col_layer_count - 1):
            layers_ddf.append(nn.Linear(col_layer_width, col_layer_width))
        layers_col.append(nn.Linear(col_layer_width, 3))
        # Set layers as optimization targets
        self.layers_ddf = nn.ModuleList(layers_ddf)
        self.layers_col = nn.ModuleList(layers_col)

        self.d_near: float = d_near

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
        for layer_id, layer in enumerate(self.layers_ddf):
            hx = self.activation(layer(hx))
            if layer_id in self.skips:
                hx = torch.cat([hx, embed_pos], dim=1)
        distance: Tensor = softplus(hx[:, :1]) + self.d_near
        aux_grad: Tensor = sigmoid(hx[:, 1:2])
        features: Tensor = hx

        d_output = torch.ones_like(
            distance, requires_grad=False, device=distance.device
        )
        # gradients of distance field take normal vector
        distance_grad = torch.autograd.grad(
            outputs=distance,
            inputs=input_pos,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(-1, 3)
        aux_gg = torch.autograd.grad(
            outputs=aux_grad,
            inputs=input_pos,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(-1, 3)
        nabla_distance: Tensor = torch.cat([distance_grad, aux_grad], dim=1)
        distance_grad_norm: Tensor = torch.norm(distance_grad, dim=1)[:, None]  # type: ignore

        # calculate density from nabla_distance and inverse distance
        dDdt: Tensor = torch.norm(nabla_distance, dim=1)[:, None]  # type: ignore
        distance_inv: Tensor = torch.reciprocal(distance)
        density: Tensor = distance_inv * (1 - dDdt)

        # for calculate penalties
        norm_dir = torch.reciprocal(distance_grad_norm + 1e-7) * distance_grad
        d2D_dwdt = torch.sum(aux_gg * norm_dir, 1)[:, None]
        d2D_dwdt_rest = 3 * aux_grad * distance_inv.detach()

        # penalty for aux. gradient
        w_penalty_scale = (
            aux_grad.detach() * distance_grad_norm.detach() * distance.detach()
        )
        w_penalty = w_penalty_scale * torch.square(d2D_dwdt - d2D_dwdt_rest)

        # penalty for values which over ranges
        # dDdt < 1.0
        s_penalty_dDdt = torch.square(relu(-1.0 + dDdt))
        # -4.0 < (distance before softplus) < 2.0
        s_penalty_distance = torch.square(
            relu(-4.0 - hx[:, :1]) + relu(-2.0 + hx[:, :1])
        )
        # -4.0 < (aux_grad before softplus) < 4.0
        s_penalty_aux_grad = torch.square(
            relu(-4.0 - hx[:, 1:2]) + relu(-4.0 + hx[:, 1:2])
        )
        s_penalty = s_penalty_dDdt + s_penalty_distance + s_penalty_aux_grad

        hx = torch.cat(
            [input_pos.reshape(-1, 3), embed_dir, distance_grad, features], dim=1
        )
        for layer in self.layers_col:
            hx = self.activation(layer(hx))
        color: Tensor = hx

        output_dict: Dict[str, Tensor] = {
            "distance": distance.reshape(batch_size, sampling),
            "density": density.reshape(batch_size, sampling),
            "color": color.reshape(batch_size, sampling, 3),
            "w_penalty": w_penalty.reshape(batch_size, sampling),
            "s_penalty": s_penalty.reshape(batch_size, sampling),
        }
        return output_dict
