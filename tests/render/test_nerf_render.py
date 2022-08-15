from typing import Dict, Final

import torch
from torch import Tensor

from neddf.render import NeRFRender


class TestNeRFRender:
    def test_nerf_render_constructor(self, nerf_config_fixture):
        # test construct of nerf_render
        NeRFRender(
            network_config=nerf_config_fixture,
        )

    def test_integrate_volume_render(self, nerf_config_fixture):
        # test data
        batch_size: Final[int] = 32
        sampling_count: Final[int] = 64
        distances: Tensor = (
            torch.linspace(0.0, 2.0, sampling_count, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, sampling_count)
        )
        densities: Tensor = torch.ones(batch_size, sampling_count)
        colors: Tensor = torch.ones(batch_size, sampling_count, 3)

        # make nerf render instance
        # TODO: add other networks list and test them
        neural_render: NeRFRender = NeRFRender(
            network_config=nerf_config_fixture,
        )

        render_result: Dict[str, Tensor] = neural_render.integrate_volume_render(
            distances,
            densities,
            colors,
        )
        # check that the result dict have collect keys
        assert "depth" in render_result
        assert "color" in render_result
        assert "transmittance" in render_result

        assert render_result["depth"].shape == (batch_size,)
        assert render_result["color"].shape == (batch_size, 3)
        assert render_result["transmittance"].shape == (batch_size,)

    def test_render_rays(self, camera_fixture, nerf_config_fixture):
        # test data: 2D positions in uv
        batch_size: Final[int] = 32
        us: Tensor = torch.linspace(0, 480, batch_size, dtype=torch.float32)
        vs: Tensor = torch.linspace(0, 640, batch_size, dtype=torch.float32)
        uv: Tensor = torch.stack([us, vs], 1)

        # make nerf render instance
        # TODO: add other networks list and test them
        neural_render: NeRFRender = NeRFRender(
            network_config=nerf_config_fixture,
        )
        render_result: Dict[str, Tensor] = neural_render.render_rays(uv, camera_fixture)
        # check that the result dict have collect keys
        assert "depth" in render_result
        assert "color" in render_result
        assert "transmittance" in render_result

        assert render_result["depth"].shape == (batch_size,)
        assert render_result["color"].shape == (batch_size, 3)
        assert render_result["transmittance"].shape == (batch_size,)
