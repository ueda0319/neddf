import numpy as np
import pytest
from numpy import ndarray

from neddf.camera import Camera, PinholeCalib
from neddf.network import NeDDF, NeRF, NeuS
from neddf.render import NeRFRender


@pytest.fixture
def calib_fixture():
    # test params for camera intrinsic parameter
    calib_param: ndarray = np.array([100.0, 100.0, 320.0, 240.0])
    # make camera_calib instance from parameters
    camera_calib: PinholeCalib = PinholeCalib(calib_param)
    return camera_calib


@pytest.fixture
def camera_fixture():
    # test params for camera intrinsic parameter
    calib_param: ndarray = np.array([100.0, 100.0, 320.0, 240.0])
    # test params for camera extrinsic parameter
    camera_param: ndarray = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # np.array([0.02, 0.04, 0.06, 0.1, 0.2, 0.3])

    # make camera_calib and camera instance from parameters
    camera_calib: PinholeCalib = PinholeCalib(calib_param)
    camera: Camera = Camera(camera_calib, camera_param)
    return camera


@pytest.fixture
def nerf_fixture():
    nerf: NeRF = NeRF(
        embed_pos_rank=10,
        embed_dir_rank=4,
        layer_count=8,
        layer_width=256,
        activation_type="ReLU",
        skips=[4],
    )
    return nerf


@pytest.fixture
def neus_fixture():
    neus: NeuS = NeuS(
        embed_pos_rank=6,
        embed_dir_rank=4,
        sdf_layer_count=8,
        sdf_layer_width=256,
        col_layer_count=8,
        col_layer_width=256,
        init_variance=0.3,
        activation_type="ReLU",
        skips=[4],
    )
    return neus


@pytest.fixture
def neddf_fixture():
    neddf: NeDDF = NeDDF(
        embed_pos_rank=6,
        embed_dir_rank=4,
        ddf_layer_count=8,
        ddf_layer_width=256,
        col_layer_count=8,
        col_layer_width=256,
        d_near=0.01,
        activation_type="ReLU",
        skips=[4],
    )
    return neddf


@pytest.fixture
def nerf_render_fixture():
    network_coarse: NeRF = NeRF(
        embed_pos_rank=10,
        embed_dir_rank=4,
        layer_count=8,
        layer_width=256,
        activation_type="ReLU",
        skips=[4],
    )
    network_fine: NeRF = NeRF(
        embed_pos_rank=10,
        embed_dir_rank=4,
        layer_count=8,
        layer_width=256,
        activation_type="ReLU",
        skips=[4],
    )
    neural_render: NeRFRender = NeRFRender(
        network_coarse=network_coarse,
        network_fine=network_fine,
    )
    return neural_render
