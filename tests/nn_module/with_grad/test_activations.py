import torch
from torch import Tensor

from neddf.nn_module.with_grad import LeakyReLUGradFunction, ReLUGradFunction, SoftplusGradFunction, TanhExpGradFunction


class TestActivationss:
    def test_activations(self):
        target_activations = [
            #LeakyReLUGradFunction.apply,
            ReLUGradFunction.apply, 
            #SoftplusGradFunction.apply,
            #TanhExpGradFunction.apply,
        ]
        for activation in target_activations:
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
            y, G = activation(x, J)
            # check the result size
            assert y.shape == (batch_size, feature_dim)
            assert G.shape == (batch_size, input_dim, feature_dim)
            
            grad_output = torch.ones_like(y, requires_grad=False)
            dydx_tuple = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True,
            )
            dydx = dydx_tuple[0].reshape(batch_size, feature_dim)
            dydJ = torch.autograd.grad(
                outputs=y,
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
                y2, G2 = activation(x+delta_x, J)
                assert torch.mean(((y2-y)*1000 - G[:, axis, :]).abs()) < 5e-4

                grad_output = torch.zeros_like(G, requires_grad=False)
                grad_output[:, axis, :] = 1.0
                dGdx = torch.autograd.grad(
                    outputs=G,
                    inputs=x,
                    grad_outputs=grad_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].reshape(batch_size, feature_dim)            
                assert torch.mean(((G2-G)[:, axis, :] * 1000 - dGdx*J[:, axis, :]).abs()) < 5e-4
            
            input_axis = 1
            feature_axis = 2
            delta_J = 0.001 * torch.ones(batch_size, input_dim, feature_dim)
            y3, G3 = activation(x, J+delta_J)
            grad_output = torch.ones_like(G, requires_grad=False)
            dGdJ = torch.autograd.grad(
                outputs=G,
                inputs=J,
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0].reshape(batch_size, input_dim, feature_dim)   
            assert torch.mean(((G3-G)*1000 - dGdJ).abs()) < 5e-4


            
            

