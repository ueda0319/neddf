import os
from pathlib import Path
from typing import Final, List

import hydra
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from numpy import ndarray
from omegaconf import DictConfig
from open3d.visualization.rendering import MaterialRecord
from scipy.spatial.transform import Rotation

from neddf.dataset.base_dataset import BaseDataset


class Visualizer:
    def __init__(self, dataset: BaseDataset) -> None:
        # Member variables to operate with gui
        self.show_rgb_image: bool = True
        self.show_bounding_box: bool = False
        self.show_visible_range: bool = False
        self.bounding_box_range: ndarray = np.array(
            [
                [-1.0, 1.0],  # X-axis
                [-1.0, 1.0],  # Y-axis
                [-1.0, 1.0],  # Z-axis
            ]
        )
        self.visible_range: ndarray = np.array([4.0, 6.0])

        # Constant member variables
        self.dataset: Final[BaseDataset] = dataset
        self.default_material: Final[MaterialRecord] = MaterialRecord()
        self.default_material.shader = "defaultUnlit"
        self.line_material: Final[MaterialRecord] = MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 3
        self.thin_line_material: Final[MaterialRecord] = MaterialRecord()
        self.thin_line_material.shader = "unlitLine"
        self.thin_line_material.line_width = 1

        # Window
        w: gui.Window = gui.Application.instance.create_window(
            "neddf Dataset Visualizer", 1024, 768
        )
        # Scene 3D viewer panel
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(w.renderer)
        scene.scene.set_background([1, 1, 1, 1])
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        scene.setup_camera(60, bbox, [0, 0, 0])
        # Setting panel
        em: float = w.theme.font_size
        settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        # Store main components to attributes for track from events
        self.scene: Final[gui.SceneWidget] = scene
        self.settings_panel: Final[gui.Widget] = settings_panel
        self.window: Final[gui.Window] = w

        # Viwer options
        show_option_layout = gui.Vert()
        show_rgb_checkbox = gui.Checkbox("show rgb images")
        show_rgb_checkbox.checked = self.show_rgb_image
        show_rgb_checkbox.set_on_checked(self._on_show_rgb_image)
        show_bb_checkbox = gui.Checkbox("show bounding box")
        show_bb_checkbox.checked = self.show_bounding_box
        show_bb_checkbox.set_on_checked(self._on_show_bounding_box)
        show_visible_range_checkbox = gui.Checkbox("show visible range")
        show_visible_range_checkbox.checked = self.show_visible_range
        show_visible_range_checkbox.set_on_checked(self._on_show_visible_range)
        show_option_layout.add_stretch()
        show_option_layout.add_child(gui.Label("Viewer options"))
        show_option_layout.add_child(show_rgb_checkbox)
        show_option_layout.add_child(show_bb_checkbox)
        show_option_layout.add_child(show_visible_range_checkbox)

        # Visible range options
        visible_range_layout = gui.Vert()
        visible_range_near_slider = gui.Slider(gui.Slider.DOUBLE)
        visible_range_near_slider.set_limits(1.0, 8.0)
        visible_range_near_slider.double_value = self.visible_range[0]
        visible_range_far_slider = gui.Slider(gui.Slider.DOUBLE)
        visible_range_far_slider.set_limits(1.0, 8.0)
        visible_range_far_slider.double_value = self.visible_range[1]
        visible_range_far_slider.set_on_value_changed(self._on_visible_range_far_slider)
        visible_range_layout.add_child(gui.Label("Camera Visible area"))
        visible_range_layout.add_child(visible_range_near_slider)
        visible_range_layout.add_child(visible_range_far_slider)

        # Refresh button
        button_layout = gui.Vert()
        refresh_button = gui.Button("Refresh render")
        refresh_button.set_on_clicked(self._on_refresh_render)
        button_layout.add_stretch()
        button_layout.add_child(refresh_button)

        # Add setting layouts to setting_panels
        separation_height = int(round(0.5 * em))
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(show_option_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(visible_range_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(button_layout)

        w.set_on_layout(self._on_layout)
        w.add_child(scene)
        w.add_child(settings_panel)

        self.refresh_render()

    def _on_refresh_render(self) -> None:
        self.refresh_render()

    def _on_show_rgb_image(self, show: bool) -> None:
        self.show_rgb_image = show
        self.refresh_render()

    def _on_show_bounding_box(self, show: bool) -> None:
        self.show_bounding_box = show
        self.refresh_render()

    def _on_show_visible_range(self, show: bool) -> None:
        self.show_visible_range = show
        self.refresh_render()

    def _on_visible_range_near_slider(self, new_val: float) -> None:
        if new_val > self.visible_range[1]:
            print("invalid visible range modified")
            new_val = self.visible_range[1]
        self.visible_range[0] = new_val

        # Note: Refresh_render is not called because rendering
        # every time the slide bar moves would be slow.
        # self.refresh_render()

    def _on_visible_range_far_slider(self, new_val: float) -> None:
        if new_val < self.visible_range[0]:
            print("invalid visible range modified")
            new_val = self.visible_range[0]
        self.visible_range[1] = new_val

        # Note: Refresh_render is not called because rendering
        # every time the slide bar moves would be slow.
        # self.refresh_render()

    def _on_layout(self, layout_context: gui.Widget) -> None:
        r = self.window.content_rect
        self.scene.frame = r
        settings_panel_width = 14 * layout_context.theme.font_size
        self.settings_panel.frame = gui.Rect(
            r.get_right() - settings_panel_width, r.y, settings_panel_width, r.height
        )

    def refresh_render(self) -> None:
        self.scene.scene.clear_geometry()

        self.draw_coordinate_grid()
        self.draw_camera_pyramid()
        if self.show_rgb_image:
            self.draw_camera_img()
        if self.show_bounding_box:
            self.draw_bounding_box()
        if self.show_visible_range:
            self.draw_visible_range()

    def draw_coordinate_grid(self) -> None:
        # draw axis arrows
        coordinate_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.scene.scene.add_geometry(
            "coordinate_axis", coordinate_axis, self.default_material
        )

        # draw coordinate grid
        x_line_vtx = [
            [[-5.0, float(i), 0.0], [5.0, float(i), 0.0]] for i in range(-5, 6)
        ]
        y_line_vtx = [
            [[float(i), -5.0, 0.0], [float(i), 5.0, 0.0]] for i in range(-5, 6)
        ]
        vtx: ndarray = np.array(x_line_vtx + y_line_vtx).reshape(-1, 3)
        edges: List[List[int]] = [[i * 2, i * 2 + 1] for i in range(22)]
        colors: ndarray = np.ones((22, 3), np.float32) * 0.5
        grid: o3d.geometry.LineSet = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(vtx)
        grid.lines = o3d.utility.Vector2iVector(edges)
        grid.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.add_geometry("coordinate_grid", grid, self.line_material)

    def draw_bounding_box(self) -> None:
        bb = self.bounding_box_range
        vtx: List[List[float]] = [
            [bb[0, 0], bb[1, 0], bb[2, 0]],
            [bb[0, 0], bb[1, 0], bb[2, 1]],
            [bb[0, 0], bb[1, 1], bb[2, 1]],
            [bb[0, 0], bb[1, 1], bb[2, 0]],
            [bb[0, 1], bb[1, 0], bb[2, 0]],
            [bb[0, 1], bb[1, 0], bb[2, 1]],
            [bb[0, 1], bb[1, 1], bb[2, 1]],
            [bb[0, 1], bb[1, 1], bb[2, 0]],
        ]
        edges: List[List[int]] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        colors: ndarray = np.ones((12, 3), np.float32)
        colors[:, 0] = 0.8
        colors[:, 1] = 0.6
        colors[:, 2] = 0.2

        bounding_box: o3d.geometry.LineSet = o3d.geometry.LineSet()
        bounding_box.points = o3d.utility.Vector3dVector(vtx)
        bounding_box.lines = o3d.utility.Vector2iVector(edges)
        bounding_box.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.add_geometry("bounding_box", bounding_box, self.line_material)

    def draw_camera_img(self, f: float = 0.5) -> None:
        for idx, data in enumerate(self.dataset):
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            tx: float = f * 0.5 * self.dataset.image_width / camera_calib_param[0]
            ty: float = f * 0.5 * self.dataset.image_height / camera_calib_param[1]
            vtx: List[List[float]] = [
                [-tx, ty, -f],
                [tx, ty, -f],
                [tx, -ty, -f],
                [-tx, -ty, -f],
            ]
            faces: List[List[int]] = [
                [2, 1, 0],
                [0, 3, 2],  # outside
            ]
            face_uv: List[List[float]] = [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
            rgb_orig: ndarray = data["rgb_images"]
            # flip up to down and convert BGR to RGB
            rgb: ndarray = np.flip(np.flipud(rgb_orig), axis=2).copy()

            image_panel: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vtx), o3d.utility.Vector3iVector(faces)
            )
            image_panel.compute_vertex_normals()
            image_panel.triangle_uvs = o3d.open3d.utility.Vector2dVector(face_uv)
            image_panel.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
            # image_panel.textures = [o3d.geometry.Image(rgb)]

            transform: ndarray = np.eye(4)
            transform[:3, :3] = Rotation.from_rotvec(camera_param[:3]).as_matrix()
            transform[:3, 3] = camera_param[3:6]
            image_panel.transform(transform)

            material: MaterialRecord = MaterialRecord()
            material.shader = "defaultUnlit"
            material.albedo_img = o3d.geometry.Image(rgb)

            self.scene.scene.add_geometry(
                "camera_{}_rgb".format(idx), image_panel, material
            )

    def draw_camera_pyramid(self, f: float = 0.5) -> None:
        for idx, data in enumerate(self.dataset):
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            tx: float = f * 0.5 * self.dataset.image_width / camera_calib_param[0]
            ty: float = f * 0.5 * self.dataset.image_height / camera_calib_param[1]
            vtx: List[List[float]] = [
                [0.0, 0.0, 0.0],
                [-tx, ty, -f],
                [tx, ty, -f],
                [tx, -ty, -f],
                [-tx, -ty, -f],
            ]
            edges: List[List[int]] = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 1],
            ]
            colors: ndarray = np.zeros((8, 3), np.float32)
            colors[:, 1] = 0.5
            colors[:, 2] = 0.9

            lines: o3d.geometry.LineSet = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(vtx)
            lines.lines = o3d.utility.Vector2iVector(edges)
            lines.colors = o3d.utility.Vector3dVector(colors)

            transform: ndarray = np.eye(4)
            transform[:3, :3] = Rotation.from_rotvec(camera_param[:3]).as_matrix()
            transform[:3, 3] = camera_param[3:6]
            lines.transform(transform)

            self.scene.scene.add_geometry(
                "camera_{}_pyramid".format(idx), lines, self.thin_line_material
            )

    def draw_visible_range(self) -> None:
        for idx, data in enumerate(self.dataset):
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            d_near: float = self.visible_range[0]
            d_far: float = self.visible_range[1]
            tx_near: float = (
                d_near * 0.5 * self.dataset.image_width / camera_calib_param[0]
            )
            ty_near: float = (
                d_near * 0.5 * self.dataset.image_height / camera_calib_param[1]
            )
            tx_far: float = (
                d_far * 0.5 * self.dataset.image_width / camera_calib_param[0]
            )
            ty_far: float = (
                d_far * 0.5 * self.dataset.image_height / camera_calib_param[1]
            )
            vtx: List[List[float]] = [
                [-tx_near, ty_near, -d_near],
                [tx_near, ty_near, -d_near],
                [tx_near, -ty_near, -d_near],
                [-tx_near, -ty_near, -d_near],
                [-tx_far, ty_far, -d_far],
                [tx_far, ty_far, -d_far],
                [tx_far, -ty_far, -d_far],
                [-tx_far, -ty_far, -d_far],
            ]
            edges: List[List[int]] = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            colors: ndarray = np.ones((12, 3), np.float32)
            colors[:, 0] = 0.2
            colors[:, 1] = 0.8
            colors[:, 2] = 0.4

            lines: o3d.geometry.LineSet = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(vtx)
            lines.lines = o3d.utility.Vector2iVector(edges)
            lines.colors = o3d.utility.Vector3dVector(colors)

            transform: ndarray = np.eye(4)
            transform[:3, :3] = Rotation.from_rotvec(camera_param[:3]).as_matrix()
            transform[:3, 3] = camera_param[3:6]
            lines.transform(transform)

            self.scene.scene.add_geometry(
                "camera_{}_visible_range".format(idx), lines, self.thin_line_material
            )


@hydra.main(config_path="../../config", config_name="default")
def main(cfg: DictConfig) -> None:
    cwd: Final[Path] = Path(hydra.utils.get_original_cwd())
    cfg.dataset.dataset_dir = str(cwd / cfg.dataset.dataset_dir)
    dataset: Final[BaseDataset] = hydra.utils.instantiate(cfg.dataset)

    gui.Application.instance.initialize()
    Visualizer(dataset)
    # Run the event loop
    gui.Application.instance.run()


if __name__ == "__main__":
    # Set current directory for run from python (not poetry)
    if "neddf/neddf" in os.getcwd():
        os.chdir("..")
    main()
