from typing import Dict

import torch
from torch import Tensor

from neddf.network import NeDDF


class TestNeDDFNetwork:
    def test_neddf_constructor(self):
        network: NeDDF = NeDDF(
            embed_pos_rank=6,
            embed_dir_rank=4,
            ddf_layer_count=8,
            ddf_layer_width=256,
            col_layer_count=8,
            col_layer_width=256,
            d_near=0.01,
            activation_type="ReLU",
            skips=[4],
        )
        assert hasattr(network, "pe_pos")
        assert hasattr(network, "pe_dir")
        assert hasattr(network, "layers_ddf")
        assert hasattr(network, "layers_col")
        assert hasattr(network, "activation")

    def test_neddf(self, neddf_fixture):
        batch_size: int = 10
        sampling: int = 64
        input_pos: Tensor = torch.zeros(batch_size, sampling, 3, dtype=torch.float32)
        input_dir: Tensor = torch.zeros(batch_size, sampling, 3, dtype=torch.float32)

        network_output: Dict[str, Tensor] = neddf_fixture(input_pos, input_dir)
        distance: Tensor = network_output["distance"]
        density: Tensor = network_output["density"]
        color: Tensor = network_output["color"]
        assert distance.shape == (batch_size, sampling)
        assert density.shape == (batch_size, sampling)
        assert color.shape == (batch_size, sampling, 3)
