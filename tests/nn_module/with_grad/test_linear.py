import torch
from torch import Tensor

from neddf.nn_module.with_grad import LinearGradFunction


class TestLinearGradFunction:
    def test_linear_grad_function(self):
        # test settings
        batch_size: int = 10
        input_dim: int = 3
        output_dim: int = 128
        # tensors for test
        torch.manual_seed(1)
        x = torch.rand(batch_size, input_dim)
        J = torch.eye(3).unsqueeze(0).expand(batch_size, input_dim, input_dim)
        weight_t = torch.rand(input_dim, output_dim)
        bias = torch.rand(output_dim)
        y, G = LinearGradFunction.apply(x, J, weight_t, bias)
        # check the result size
        assert y.shape == (batch_size, output_dim)
        assert G.shape == (batch_size, input_dim, output_dim)
        # check gradients is collect
        for axis in range(3):
            delta_t = torch.zeros_like(x)
            delta_t[:, axis] = 0.01
            y2, G2 = LinearGradFunction.apply(x+delta_t, J, weight_t, bias)
            assert torch.mean((G-G2).abs()) < 1e-5
            assert torch.mean(((y2-y)*100 - G[:, axis, :]).abs()) < 1e-5

