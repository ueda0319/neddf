import torch
from torch import Tensor


class Sampling:
    def __init__(
        self,
        sample_pos: Tensor,
        sample_dir: Tensor,
        diag_variance: Tensor,
    ) -> None:
        """Initializer

        This method Initialize Sampling class

        Args:
            sample_pos (Tensor[batch_size, sample_size, 3]):
                Position of sampling point
            sample_dir (Tensor[batch_size, sample_size, 3]):
                Direction of sampling point
            diag_variance (Tensor[batch_size, sample_size, 3]):
                Diagonal of covariance matrix.
                Set 0 for point sampling(without sampling radius)
        """
        self.sample_pos: Tensor = sample_pos
        self.sample_dir: Tensor = sample_dir
        self.diag_variance: Tensor = diag_variance

    @property
    def device(self) -> torch.device:
        return self.sample_pos.device

    def get_pe_weights(self, freq: Tensor) -> Tensor:
        """Get PE Weights

        This method calculate weight for each frequency in Positional Encoding

        Args:
            freq (Tensor[freq_dim]): frequency values for each pe channels
        """
        with torch.set_grad_enabled(False):
            # count of sampling (batch_size * sample_size)
            sample_dim: int = self.diag_variance.shape[0] * self.diag_variance.shape[1]
            # dimension of field (always be 3)
            field_dim: int = self.diag_variance.shape[2]

            diag_variance: Tensor = (
                self.diag_variance[:, :, None, :]
                .expand(-1, -1, freq.shape[0], -1)
                .reshape(sample_dim, -1)
            )
            freq_sq = (
                torch.square(freq)[None, :, None]
                .expand(sample_dim, -1, field_dim)
                .reshape(sample_dim, -1)
            )
            return torch.exp(-0.5 * freq_sq * diag_variance)
