import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import cv2
import numpy as np

# Intrinsics and extrinsics
M1 = np.array([[1305.7481984348224, 0., 849.53637102757307],
               [0., 1303.9712600149542, 654.92692268353301],
               [0., 0., 1.]])
fx = M1[0, 0]
D1 = np.array([0.0232, -0.0072, -0.0001, 0.0013, -0.0609])
M2 = np.array([[1303.1897, 0., 766.6739],
               [0., 1301.1464, 640.1612],
               [0., 0., 1.]])
D2 = np.array([0.0257, -0.0183, -0.00007, -0.00012, -0.0393])
R = np.array([[0.89039, -0.01347, 0.455], [0.0133, 0.9999, 0.0036], [-0.455, 0.0028, 0.8905]])
T = np.array([-0.1057, -0.0011, 0.0250])
baseline = np.linalg.norm(T)

# Load and rectify images
image_left = cv2.imread('./images/left_0.png', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread('./images/right_0.png', cv2.IMREAD_GRAYSCALE)
image_left_color = cv2.imread('./images/right_0.png', cv2.IMREAD_COLOR)  # Load color for point colors

def rectify_images(image_left, image_right):
    h, w = image_left.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, D1, M2, D2, (w, h), R, T, alpha=1)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w, h), cv2.CV_32FC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w, h), cv2.CV_32FC2)
    rectified_left = cv2.remap(image_left, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(image_right, right_map1, right_map2, cv2.INTER_LINEAR)
    return rectified_left, rectified_right, Q

rectified_left, rectified_right, Q = rectify_images(image_left, image_right)

def compute_disparity(rectified_left, rectified_right, min_disp, num_disp, block_size, uniqueness_ratio, speckle_window_size, speckle_range):
    sbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=1,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
    )
    disparity = sbm.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
    return disparity

def disparity_to_3d(disparity, Q):
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 1  # Filter out invalid disparities
    points_3d = points_3d[mask]
    return points_3d, mask

def create_point_cloud(points_3d, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# AppWindow class for Open3D GUI
class AppWindow:
    def __init__(self, width, height):
        self.window = gui.Application.instance.create_window("Stereo 3D Viewer", width, height)
        
        # 3D widget for displaying point cloud
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._scene)
        
        # Initialize point cloud and depth settings
        self.min_disp = 0
        self.num_disp = 30 * 16
        self.block_size = 3
        self.uniqueness_ratio = 20
        self.speckle_window_size = 45
        self.speckle_range = 13
        self.min_depth = 0.24
        self.max_depth = 0.7

        # Set initial camera position
        self._scene.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, -1, 0])

        # Add sliders for SGBM parameters and depth filtering
        self.panel = gui.Vert(0.25 * self.window.theme.font_size)
        self._add_slider("Min Disparity", self.min_disp, 0, 100, self._on_min_disp)
        self._add_slider("Num Disparities", 25, 1, 50, self._on_num_disp, 1)
        self._add_slider("Block Size", self.block_size, 3, 51, self._on_block_size, 2)
        self._add_slider("Uniqueness Ratio", self.uniqueness_ratio, 5, 20, self._on_uniqueness_ratio)
        self._add_slider("Speckle Window Size", self.speckle_window_size, 0, 200, self._on_speckle_window_size)
        self._add_slider("Speckle Range", self.speckle_range, 0, 64, self._on_speckle_range)
        self._add_slider("Min Depth", int(self.min_depth * 100), 0, 100, self._on_min_depth)
        self._add_slider("Max Depth", int(self.max_depth * 100), 0, 500, self._on_max_depth)

        # Add checkbox to show/hide axes
        self.show_axes_checkbox = gui.Checkbox("Show Axes")
        self.show_axes_checkbox.checked = True  # Set the checkbox to be checked by default
        self.show_axes_checkbox.set_on_checked(self._on_show_axes)
        self.panel.add_child(self.show_axes_checkbox)
        
        self.window.add_child(self.panel)
        
        # Create axes geometry but do not add it to the scene yet
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Initial point cloud rendering
        self.update_point_cloud()
        self.window.set_on_layout(self._on_layout)

    def _add_slider(self, name, value, min_val, max_val, callback, step=1):
        label = gui.Label(name)
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(min_val, max_val)
        slider.int_value = value
        slider.set_on_value_changed(callback)
        self.panel.add_child(label)
        self.panel.add_child(slider)

    def _on_min_disp(self, value):
        self.min_disp = int(value)
        self.update_point_cloud()

    def _on_num_disp(self, value):
        self.num_disp = int(value) * 16
        self.update_point_cloud()

    def _on_block_size(self, value):
        self.block_size = int(value)
        self.update_point_cloud()

    def _on_uniqueness_ratio(self, value):
        self.uniqueness_ratio = int(value)
        self.update_point_cloud()

    def _on_speckle_window_size(self, value):
        self.speckle_window_size = int(value)
        self.update_point_cloud()

    def _on_speckle_range(self, value):
        self.speckle_range = int(value)
        self.update_point_cloud()

    def _on_min_depth(self, value):
        self.min_depth = value / 100.0
        self.update_point_cloud()

    def _on_max_depth(self, value):
        self.max_depth = value / 100.0
        self.update_point_cloud()

    def _on_show_axes(self, is_checked):
        if is_checked:
            self._scene.scene.add_geometry("axes", self.axes, rendering.MaterialRecord())
        else:
            self._scene.scene.remove_geometry("axes")

    def update_point_cloud(self):
        # Compute disparity and 3D points
        disparity = compute_disparity(rectified_left, rectified_right,
                                      self.min_disp, self.num_disp,
                                      self.block_size, self.uniqueness_ratio,
                                      self.speckle_window_size, self.speckle_range)

        points_3d, mask = disparity_to_3d(disparity, Q)
        colors = cv2.cvtColor(image_left_color, cv2.COLOR_BGR2RGB) / 255.0
        colors = colors[mask]
        
        # Filter points based on depth
        depths = points_3d[:, 2]  # Z-axis represents depth
        valid_depth_mask = (depths > self.min_depth) & (depths < self.max_depth)
        points_3d = points_3d[valid_depth_mask]
        colors = colors[valid_depth_mask]
        
        # Create point cloud
        pcd = create_point_cloud(points_3d, colors)

        # Remove previous geometry and add updated point cloud
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("point_cloud", pcd, rendering.MaterialRecord())

        # Add axes if checkbox is checked
        if self.show_axes_checkbox.checked:
            self._scene.scene.add_geometry("axes", self.axes, rendering.MaterialRecord())

        # Set up the camera to fit the point cloud in view
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_layout(self, context):
        rect = self.window.content_rect
        width = 200  # Fixed width for sidebar
        self._scene.frame = gui.Rect(rect.x, rect.y, rect.width - width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - width, rect.y, width, rect.height)

def main():
    gui.Application.instance.initialize()
    w = AppWindow(1024, 768)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
