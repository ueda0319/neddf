from typing import Dict

import torch
from torch import Tensor

from neddf.network import NeDDF
from neddf.ray import Sampling


class TestNeDDFNetwork:
    def test_neddf_constructor(self):
        penalty_weight: Dict[str, float] = {
            "constraints_aux_grad": 0.05,
            "constraints_dDdt": 0.05,
            "constraints_color": 0.0,
            "range_distance": 1.0,
            "range_aux_grad": 1.0,
        }
        network: NeDDF = NeDDF(
            embed_pos_rank=6,
            embed_dir_rank=4,
            ddf_layer_count=8,
            ddf_layer_width=256,
            col_layer_count=8,
            col_layer_width=256,
            d_near=0.01,
            activation_type="ReLU",
            penalty_weight=penalty_weight,
            skips=[4],
        )
        assert hasattr(network, "pe_pos")
        assert hasattr(network, "pe_dir")
        assert hasattr(network, "layers_ddf")
        assert hasattr(network, "layers_col")
        assert hasattr(network, "activation")

    def test_neddf(self, neddf_fixture):
        batch_size: int = 10
        sampling_size: int = 64
        input_pos: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        input_dir: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        diag_variance: Tensor = torch.zeros(batch_size, sampling_size, 3, dtype=torch.float32)
        sampling: Sampling = Sampling(input_pos, input_dir, diag_variance)

        network_output: Dict[str, Tensor] = neddf_fixture(sampling)
        distance: Tensor = network_output["distance"]
        density: Tensor = network_output["density"]
        color: Tensor = network_output["color"]
        assert distance.shape == (batch_size, sampling_size)
        assert density.shape == (batch_size, sampling_size)
        assert color.shape == (batch_size, sampling_size, 3)
