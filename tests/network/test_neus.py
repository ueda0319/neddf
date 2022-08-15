from typing import Dict

import torch
from torch import Tensor

from neddf.network import NeuS
from neddf.ray import Sampling


class TestNeuSNetwork:
    def test_neus_constructor(self):
        network: NeuS = NeuS(
            embed_pos_rank=6,
            embed_dir_rank=4,
            sdf_layer_count=8,
            sdf_layer_width=256,
            col_layer_count=8,
            col_layer_width=256,
            init_variance=0.3,
            activation_type="ReLU",
            skips=[4],
        )
        assert hasattr(network, "pe_pos")
        assert hasattr(network, "pe_dir")
        assert hasattr(network, "layers_sdf")
        assert hasattr(network, "layers_col")
        assert hasattr(network, "variance")
        assert hasattr(network, "activation")

    def test_neus(self, neus_fixture):
        batch_size: int = 10
        sampling_size: int = 64
        input_pos: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        input_dir: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        diag_variance: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        sampling: Sampling = Sampling(input_pos, input_dir, diag_variance)

        network_output: Dict[str, Tensor] = neus_fixture(sampling)
        sdf: Tensor = network_output["sdf"]
        density: Tensor = network_output["density"]
        color: Tensor = network_output["color"]
        assert sdf.shape == (batch_size, sampling_size)
        assert density.shape == (batch_size, sampling_size)
        assert color.shape == (batch_size, sampling_size, 3)
