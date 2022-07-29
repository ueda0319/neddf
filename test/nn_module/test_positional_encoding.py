import torch
from torch import Tensor

from melon.nn_module import PositionalEncoding


class TestPositionalEncoding:
    def test_original_positional_encoding(self):
        # test settings
        batch_size: int = 10
        input_pos_dim: int = 3
        embed_dim: int = 10
        # test 3d position datas initialized by zeros
        input_pos: Tensor = torch.zeros(batch_size, input_pos_dim, dtype=torch.float32)

        # create positional encoding instance
        pe = PositionalEncoding(embed_dim)
        # get embed positions of input_pos with positional encoding
        embed_pos = pe(input_pos)
        # check the result size
        assert embed_pos.shape == (batch_size, 2 * embed_dim * input_pos_dim)
