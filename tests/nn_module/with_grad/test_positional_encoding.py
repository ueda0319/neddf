import torch
from torch import Tensor

from neddf.nn_module.with_grad import PositionalEncodingGradLayer


class TestPositionalEncodingGradLayer:
    def test_positional_grad_layer(self):
        # test settings
        batch_size: int = 10
        input_dim: int = 3
        embed_dim: int = 4
        # layer for test
        layer = PositionalEncodingGradLayer(embed_dim)
        # tensors for test
        torch.manual_seed(1)
        x = torch.rand(batch_size, input_dim)
        J = torch.rand(batch_size, input_dim, input_dim)
        y, G = layer(x, J)
        # check the result size
        assert y.shape == (batch_size, embed_dim * 6)
        assert G.shape == (batch_size, input_dim, embed_dim * 6)
        # check gradients is collect
        for axis in range(3):
            delta_t = torch.zeros_like(x)
            delta_t[:, axis] = 0.0001
            delta_x = torch.matmul(delta_t[:,None, :], J)[:, 0, :]
            y2, G2 = layer(x+delta_x, J)
            assert torch.mean(((y2-y)*10000 - G[:, axis, :]).abs()) < 5e-4

