import configparser
import copy
import os
import numpy as np
from PIL import Image
import scipy.io as scio
import yaml
import open3d as o3d
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import src.segmentation as segmentation
import src.data_augmentation as data_augmentation

camera = 'realsense'
dataset_root = '/media/rcao/Data/Dataset/graspnet'
color_lookup = np.load(os.path.join('color.npy'))


def process_rgb(rgb_img):
    """ Process RGB image
            - random color warping
    """
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = data_augmentation.standardize_image(rgb_img)

    return rgb_img


def load_config(config_path):
    fp = open(config_path, 'r')
    st = fp.read()
    fp.close()
    config = yaml.load(st, Loader=yaml.FullLoader)
    return config


uois_config = load_config('config.yaml')
uois3d_config = uois_config['uois3d']
dsn_config = uois_config['dsn']
rrn_config = uois_config['rrn']

iter_num = 70401  # 140801 for kinect 70401 for realsense
# dsn_filename = os.path.join('checkpoints', f'DSNWrapper_iter{iter_num}_64c_checkpoint.pth')
dsn_filename = os.path.join('checkpoints', camera, 'train_DSN', f'DSNWrapper_iter{iter_num}_64c_checkpoint.pth')
rrn_filename = os.path.join('checkpoints', 'RRN_OID_checkpoint.pth')

uois_net_3d = segmentation.UOISNet3D(uois3d_config,
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config)

width = 1280
height = 720
scene_idx = 100
anno_idx = 0
# for scene_idx in range(100, 101):
#     for anno_idx in range(147, 148):

rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

color = np.array(Image.open(rgb_path), dtype=np.float32)
depth = np.array(Image.open(depth_path))
gt_mask = np.array(Image.open(mask_path))

meta = scio.loadmat(meta_path)

obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
intrinsics = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

# scene = o3d.geometry.PointCloud()
# scene.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
# scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)/255.0)
# o3d.visualization.draw_geometries([scene], width=1536, height=864)

# depth_mask = (depth > 0)
# camera_poses = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
# align_mat = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
# trans = np.dot(align_mat, camera_poses[anno_idx])
# workspace_mask = get_workspace_mask(cloud, gt_seg, trans=trans, organized=True, outlier=0.02)
# mask = (depth_mask & workspace_mask)
#
# cloud_masked = cloud[mask]
# color_masked = color[mask]
# seg_masked = net_seg[mask]

# scene = o3d.geometry.PointCloud()
# scene.points = o3d.utility.Vector3dVector(cloud_masked)
# scene.colors = o3d.utility.Vector3dVector(color_masked)
# o3d.visualization.draw_geometries([scene], width=1536, height=864)

rgb = process_rgb(color)
rgb_img = data_augmentation.array_to_tensor(rgb) # Shape: [3 x H x W]
xyz_img = data_augmentation.array_to_tensor(cloud) # Shape: [3 x H x W]

batch = dict()
batch['rgb'] = rgb_img.unsqueeze(0)
batch['xyz'] = xyz_img.unsqueeze(0)

fg_masks, center_offsets, object_centers, dsn_masks, initial_masks, refined_masks = uois_net_3d.run_on_batch(batch)

fg_masks = fg_masks.cpu().numpy()
center_offsets = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)
dsn_masks = dsn_masks.cpu().numpy()
initial_masks = initial_masks[0].cpu().numpy()
refined_masks = refined_masks[0].cpu().numpy()

uois_result = refined_masks

seg_idxs = np.unique(uois_result)
mask_vis = np.zeros_like(color)
img_vis = copy.deepcopy(color)
img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

for inst_idx, obj_idx in enumerate(seg_idxs):
    if obj_idx == 0:
        continue
    inst_mask = uois_result == obj_idx

    mask_vis[inst_mask, :] = color_lookup[inst_idx][1:] * 255.0

overlapping = cv2.addWeighted(img_vis, 0.5, mask_vis, 1.0, 0)
cv2.imwrite(os.path.join('mask_{:03}.png'.format(scene_idx)), overlapping)
