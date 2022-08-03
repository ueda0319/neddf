import torch
from torch import Tensor

from neddf.nn_module.with_grad import ReLUGradFunction


class TestReLUGradFunction:
    def test_relu_grad_functino(self):
        # test settings
        batch_size: int = 10
        input_dim: int = 3
        feature_dim: int = 15
        # tensors for test
        torch.manual_seed(1)
        x = torch.rand(batch_size, feature_dim)
        J = torch.rand(batch_size, input_dim, feature_dim)
        y, G = ReLUGradFunction.apply(x, J)
        # check the result size
        assert y.shape == (batch_size, feature_dim)
        assert G.shape == (batch_size, input_dim, feature_dim)
        # check gradients is collect
        for axis in range(3):
            delta_t = torch.zeros(batch_size, input_dim)
            delta_t[:, axis] = 0.0001
            delta_x = torch.matmul(delta_t[:,None, :], J)[:, 0, :]
            y2, G2 = ReLUGradFunction.apply(x+delta_x, J)
            assert torch.mean(((y2-y)*10000 - G[:, axis, :]).abs()) < 5e-4

