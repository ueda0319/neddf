import torch
from torch import Tensor

from neddf.nn_module.with_grad import TanhExpGradFunction


class TestTanhExpGradFunction:
    def test_TanhExp_grad_function(self):
        # test settings
        batch_size: int = 10
        input_dim: int = 3
        feature_dim: int = 15
        # tensors for test
        torch.manual_seed(1)
        x = torch.rand(batch_size, feature_dim)
        J = torch.rand(batch_size, input_dim, feature_dim)
        x.requires_grad_(True)
        J.requires_grad_(True)
        y, G = TanhExpGradFunction.apply(x, J)
        # check the result size
        assert y.shape == (batch_size, feature_dim)
        assert G.shape == (batch_size, input_dim, feature_dim)
        
        grad_output = torch.ones_like(y, requires_grad=False)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(batch_size, feature_dim)
        dydJ = torch.autograd.grad(
            outputs=y,
            inputs=J,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(batch_size, input_dim, feature_dim)
        grad_output = torch.ones_like(G, requires_grad=False)
        dGdx = torch.autograd.grad(
            outputs=G,
            inputs=x,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(batch_size, feature_dim)
        dGdJ = torch.autograd.grad(
            outputs=G,
            inputs=J,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].reshape(batch_size, input_dim, feature_dim)
        assert torch.mean(((dydx[:, None, :] * J) - G).abs()) < 1e-4
        assert torch.mean(dydJ.abs()) < 1e-4

        # check gradients is collect
        for axis in range(3):
            delta_t = torch.zeros(batch_size, input_dim)
            delta_t[:, axis] = 0.001
            delta_x = torch.matmul(delta_t[:,None, :], J)[:, 0, :]
            y2, G2 = TanhExpGradFunction.apply(x+delta_x, J)
            assert torch.mean(((y2-y)*1000 - G[:, axis, :]).abs()) < 5e-4
            
            assert torch.mean(((G2-G)[:, axis, :] * 1000 - dGdx[:, axis, :]).abs()) < 5e-4
            
        

