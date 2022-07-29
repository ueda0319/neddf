from typing import Dict

import torch
from torch import Tensor

from neddf.network import NeRF


class TestNetwork:
    def test_nerf_constructor(self):
        network: NeRF = NeRF(
            embed_pos_rank=10,
            embed_dir_rank=4,
            layer_count=8,
            layer_width=64,
            activation_type="ReLU",
            skips=[4],
        )
        assert hasattr(network, "pe_pos")
        assert hasattr(network, "pe_dir")
        assert hasattr(network, "layers")
        assert hasattr(network, "outL_density")
        assert hasattr(network, "outL_color")
        assert hasattr(network, "activation")

    def test_nerf(self, nerf_fixture):
        batch_size: int = 10
        sampling: int = 64
        input_pos: Tensor = torch.zeros(batch_size, sampling, 3, dtype=torch.float32)
        input_dir: Tensor = torch.zeros(batch_size, sampling, 3, dtype=torch.float32)

        network_output: Dict[str, Tensor] = nerf_fixture(input_pos, input_dir)
        density: Tensor = network_output["density"]
        color: Tensor = network_output["color"]
        assert density.shape == (batch_size, sampling)
        assert color.shape == (batch_size, sampling, 3)
