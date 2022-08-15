from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Final, List

import hydra
import mcubes
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from hydra.core.global_hydra import GlobalHydra
from neddf.trainer import BaseTrainer
from numpy import ndarray
from omegaconf import DictConfig
from open3d.visualization.rendering import MaterialRecord
from scipy.spatial.transform import Rotation


class FieldsVisualizer:
    def __init__(self, trainer: BaseTrainer, output_dir: Path) -> None:
        # Member variables to operate with gui
        self.show_rgb_image: bool = False
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
        self.slice_parameter: float = 0.0
        self.slice_field_name: str = "distance"
        self.output_dir: Path = output_dir
        self.meshed_field: o3d.geometry.TriangleMesh
        self.meshing_resolution: int = 64
        self.meshing_threshold: float = 0.0275
        self.generate_mesh()

        # Constant member variables
        self.trainer: Final[BaseTrainer] = trainer
        self.default_material: Final[MaterialRecord] = MaterialRecord()
        self.default_material.shader = "defaultUnlit"
        self.line_material: Final[MaterialRecord] = MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 3
        self.thin_line_material: Final[MaterialRecord] = MaterialRecord()
        self.thin_line_material.shader = "unlitLine"
        self.thin_line_material.line_width = 1
        self.meshed_field_material: Final[MaterialRecord] = MaterialRecord()
        self.meshed_field_material.shader = "defaultLitTransparency"
        self.meshed_field_material.base_color = [0.467, 0.467, 0.467, 0.7]

        # Window
        w: gui.Window = gui.Application.instance.create_window(
            "neddf Fields Visualizer", 1280, 768
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

        # Slice parameter
        slice_param_layout = gui.Vert()
        slice_field_name_combo = gui.Combobox()
        slice_field_name_combo.add_item("distance")
        slice_field_name_combo.add_item("density")
        slice_field_name_combo.add_item("color")
        slice_field_name_combo.add_item("aux_grad")
        slice_field_name_combo.set_on_selection_changed(
            self._on_slice_fieldname_selection
        )
        slice_param_slider = gui.Slider(gui.Slider.DOUBLE)
        slice_param_slider.set_limits(-1.0, 1.0)
        slice_param_slider.double_value = self.slice_parameter
        slice_param_slider.set_on_value_changed(self._on_slice_parameter_slider)
        slice_param_layout.add_child(gui.Label("Slice field name"))
        slice_param_layout.add_child(slice_field_name_combo)
        slice_param_layout.add_child(gui.Label("Slice Z"))
        slice_param_layout.add_child(slice_param_slider)

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

        # Meshing
        meshing_button_layout = gui.Vert()
        meshing_resolution = gui.NumberEdit(gui.NumberEdit.Type.INT)
        meshing_resolution.set_value(64)
        meshing_resolution.set_limits(8, 256)
        meshing_resolution.set_on_value_changed(self._on_meshing_resolution)
        meshing_threshold = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        meshing_threshold.set_value(0.0275)
        meshing_threshold.set_limits(0.001, 50.0)
        meshing_threshold.set_on_value_changed(self._on_meshing_threshold)
        meshing_button = gui.Button("Generate mesh model")
        meshing_button.set_on_clicked(self._on_meshing)
        meshing_button_layout.add_stretch()
        meshing_button_layout.add_child(gui.Label("Marching cube resolution"))
        meshing_button_layout.add_child(meshing_resolution)
        meshing_button_layout.add_child(gui.Label("Marching cube threshold"))
        meshing_button_layout.add_child(meshing_threshold)
        meshing_button_layout.add_child(meshing_button)

        # Refresh button
        refresh_button_layout = gui.Vert()
        refresh_button = gui.Button("Refresh render")
        refresh_button.set_on_clicked(self._on_refresh_render)
        refresh_button_layout.add_stretch()
        refresh_button_layout.add_child(refresh_button)

        # Add setting layouts to setting_panels
        separation_height = int(round(0.5 * em))
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(show_option_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(slice_param_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(visible_range_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(meshing_button_layout)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(refresh_button_layout)

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

    def _on_slice_fieldname_selection(self, field_name: str, idx: int) -> None:
        self.slice_field_name = field_name
        self.refresh_render()

    def _on_slice_parameter_slider(self, new_val: float) -> None:
        self.slice_parameter = new_val
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

    def _on_meshing_resolution(self, new_val: float) -> None:
        self.meshing_resolution = int(new_val)

    def _on_meshing_threshold(self, new_val: float) -> None:
        self.meshing_threshold = new_val

    def _on_meshing(self) -> None:
        self.generate_mesh()
        self.refresh_render()

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
        self.draw_field_slice()
        self.draw_meshed_field()
        if self.show_rgb_image:
            self.draw_camera_img()
        if self.show_bounding_box:
            self.draw_bounding_box()
        if self.show_visible_range:
            self.draw_visible_range()

    def draw_coordinate_grid(self) -> None:
        # draw axis arrows
        # coordinate_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # self.scene.scene.add_geometry(
        #    "coordinate_axis", coordinate_axis, self.default_material
        # )

        # draw coordinate grid
        grid_range: int = 3
        x_line_vtx = [
            [[-float(grid_range), float(i), 0.0], [float(grid_range), float(i), 0.0]]
            for i in range(-grid_range, grid_range + 1)
        ]
        y_line_vtx = [
            [[float(i), -float(grid_range), 0.0], [float(i), float(grid_range), 0.0]]
            for i in range(-grid_range, grid_range + 1)
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

    def draw_field_slice(self, slice_size: float = 1.1) -> None:
        slice_t = self.slice_parameter
        slice_images: Dict[
            str, ndarray
        ] = self.trainer.neural_render.render_field_slice(
            slice_t=slice_t, render_size=1.1, render_resolution=128
        )

        vtx: List[List[float]] = [
            [-slice_size, slice_size, slice_t],
            [slice_size, slice_size, slice_t],
            [slice_size, -slice_size, slice_t],
            [-slice_size, -slice_size, slice_t],
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
        # flip up to down and convert BGR to RGB
        rgb: ndarray = np.flip(slice_images[self.slice_field_name], axis=2).copy()

        image_panel: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vtx), o3d.utility.Vector3iVector(faces)
        )
        image_panel.compute_vertex_normals()
        image_panel.triangle_uvs = o3d.open3d.utility.Vector2dVector(face_uv)
        image_panel.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))

        material: MaterialRecord = MaterialRecord()
        material.shader = "defaultUnlit"
        material.albedo_img = o3d.geometry.Image(rgb)

        self.scene.scene.add_geometry("slice_plane", image_panel, material)

    def draw_camera_img(self, f: float = 0.5) -> None:
        for idx, data in enumerate(self.trainer.dataset):  # type: ignore
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            tx: float = (
                f * 0.5 * self.trainer.dataset.image_width / camera_calib_param[0]
            )
            ty: float = (
                f * 0.5 * self.trainer.dataset.image_height / camera_calib_param[1]
            )
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
            rgb_orig: ndarray = data["rgb_images"].astype(np.uint8)
            # flip up to down and convert BGR to RGB
            rgb: ndarray = np.flip(np.flipud(rgb_orig), axis=2).copy()

            image_panel: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vtx), o3d.utility.Vector3iVector(faces)
            )
            image_panel.compute_vertex_normals()
            image_panel.triangle_uvs = o3d.open3d.utility.Vector2dVector(face_uv)
            image_panel.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))

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

    def draw_meshed_field(self) -> None:
        material: MaterialRecord = MaterialRecord()
        material.shader = "defaultUnlit"

        self.scene.scene.add_geometry(
            "meshed_field", self.meshed_field, self.meshed_field_material
        )

    def draw_camera_pyramid(self, f: float = 0.5) -> None:
        for idx, data in enumerate(self.trainer.dataset):  # type: ignore
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            tx: float = (
                f * 0.5 * self.trainer.dataset.image_width / camera_calib_param[0]
            )
            ty: float = (
                f * 0.5 * self.trainer.dataset.image_height / camera_calib_param[1]
            )
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
        for idx, data in enumerate(self.trainer.dataset):  # type: ignore
            camera_calib_param: ndarray = data["camera_calib_params"]
            camera_param: ndarray = data["camera_params"]
            d_near: float = self.visible_range[0]
            d_far: float = self.visible_range[1]
            tx_near: float = (
                d_near * 0.5 * self.trainer.dataset.image_width / camera_calib_param[0]
            )
            ty_near: float = (
                d_near * 0.5 * self.trainer.dataset.image_height / camera_calib_param[1]
            )
            tx_far: float = (
                d_far * 0.5 * self.trainer.dataset.image_width / camera_calib_param[0]
            )
            ty_far: float = (
                d_far * 0.5 * self.trainer.dataset.image_height / camera_calib_param[1]
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

    def generate_mesh(self) -> None:
        cube_range: float = 1.1
        mesh_dir: Path = self.output_dir / "mesh"
        mesh_dir.mkdir(exist_ok=True)
        voxel_file_path: Path = mesh_dir / "voxel_{}.npy".format(
            self.meshing_resolution
        )
        if voxel_file_path.exists():
            voxel: ndarray = np.load(voxel_file_path.as_posix())
        else:
            voxel = self.trainer.neural_render.get_network().voxelize(
                field_name="distance",
                cube_range=cube_range,
                cube_resolution=self.meshing_resolution,
            )
            np.save(voxel_file_path.as_posix(), voxel)

        vertices, triangles = mcubes.marching_cubes(voxel, self.meshing_threshold)
        vertices -= self.meshing_resolution / 2.0
        vertices *= 2.0 * cube_range / self.meshing_resolution
        vertices_list = np.asarray(vertices)
        triangles_list = np.asarray(triangles)
        self.meshed_field = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices_list),
            o3d.utility.Vector3iVector(triangles_list),
        )
        self.meshed_field.compute_vertex_normals()

        transform: ndarray = np.zeros((4, 4))
        transform[0, 2] = -1
        transform[1, 0] = -1
        transform[2, 1] = 1
        transform[3, 3] = 1
        self.meshed_field.transform(transform)

        # mesh_file_path: Path = mesh_dir / "mesh_{}_threshold{}.dae".format(
        #     cube_resolution, threshold
        # )
        # mcubes.export_mesh(vertices, triangles, mesh_file_path.as_posix, "mcube")


def main() -> None:
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory path where models and render are located",
    )
    parser.add_argument("--epoch", type=int, default=2000, help="epoch number of model")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    # reconstruct config
    conf_dir: Path = output_dir / ".hydra"
    assert conf_dir.is_dir()
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=conf_dir.as_posix())
    cfg: DictConfig = hydra.compose(config_name="config")
    trainer: BaseTrainer = hydra.utils.instantiate(
        cfg.trainer,
        global_config=cfg,
        _recursive_=False,
    )

    # load model path
    model_path: Path = output_dir / "models/model_{:05}.pth".format(args.epoch)
    # assert model_path.exists()
    trainer.load_pretrained_model(model_path)

    # render all
    save_dir: Path = args.output_dir / "visualize"
    save_dir.mkdir(exist_ok=True)

    gui.Application.instance.initialize()
    FieldsVisualizer(trainer, output_dir)
    # Run the event loop
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
