import numpy as np
import torch
from torch import Tensor


class TestCamera:
    def test_camera_calib(self, calib_fixture):
        # check that camera_params is set collectly
        assert calib_fixture.fx == 100.0
        assert calib_fixture.fy == 100.0
        assert calib_fixture.cx == 320.0
        assert calib_fixture.cy == 240.0

    def test_camera(self, camera_fixture):
        # test data: 3D position in world coordinate
        uv: Tensor = torch.from_numpy(np.array([[640, 480]]))
        pos_world: Tensor = camera_fixture.unproject(uv)
        uv_recovered: Tensor = camera_fixture.project(pos_world)
        # check that the restoration error is sufficiently small
        assert (uv - uv_recovered).abs().sum().item() < 1e-5
