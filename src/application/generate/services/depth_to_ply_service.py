import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# read the color and the depth image:
depth_raw = o3d.io.read_image("depth_images/test1-midas_v21_small_256.png")
color_raw = o3d.io.read_image("depth_images/test1.jpg")

# create an rgbd image object:
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, convert_rgb_to_intensity=False)
# use the rgbd image to create point cloud:
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# visualize:
o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud("depth_images/test1.ply", pcd)

# read the color and the depth image:
depth_raw = o3d.io.read_image("depth_images/test2-midas_v21_small_256.png")
color_raw = o3d.io.read_image("depth_images/test2.jpg")

# create an rgbd image object:
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, convert_rgb_to_intensity=False)
# use the rgbd image to create point cloud:
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# visualize:
o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud("depth_images/test2.ply", pcd)