import numpy as np
import os
import open3d as o3d
from tqdm import tqdm
from PIL import Image

from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
from graspnetAPI.utils.utils import CameraInfo, create_point_cloud_from_depth_image
import scipy.io as scio
from util import utilities as util_

NUM_VIEWS_PER_SCENE = 256
BACKGROUND_LABEL = 0
TABLE_LABEL = 0
OBJECTS_LABEL = 1
num_points = 1024
camera = 'realsense'
root = '/media/rcao/Data/Dataset/graspnet/'
save_root = os.path.join(root, 'center_label')

def load_obj_models():
    obj_list = list(range(88))
    obj_models = []
    for obj_id in tqdm(obj_list):
        model_path = os.path.join(root, 'models', str(obj_id).zfill(3), 'nontextured_simplified.ply')
        model_pc = o3d.io.read_point_cloud(model_path)
        model_pc = model_pc.voxel_down_sample(voxel_size=0.002)
        obj_models.append(np.asarray(model_pc.points))
    return obj_models


def process_label_3D(foreground_labels, xyz_img, scene_description, obj_models):
    """ Process foreground_labels

        @param foreground_labels: a [H x W] numpy array of labels
        @param xyz_img: a [H x W x 3] numpy array of xyz coordinates (in left-hand coordinate system)
        @param scene_description: a Python dictionary describing scene

        @return: foreground_labels
                 offsets: a [H x W x 2] numpy array of 2D directions. The i,j^th element has (y,x) direction to object center
    """

    # Any zero depth value will have foreground label set to background
    foreground_labels = foreground_labels.copy()
    foreground_labels[xyz_img[..., 2] == 0] = 0

    # Compute object centers and directions
    H, W = foreground_labels.shape
    offsets = np.zeros((H, W, 3), dtype=np.float32)
    cf_3D_centers = np.zeros((100, 3), dtype=np.float32) # 100 max object centers

    inst_pc_list = []

    obj_list = scene_description['obj_list']
    pose_list = scene_description['pose_list']
    camera_pose = scene_description['camera_pose']

    for i, k in enumerate(np.unique(foreground_labels)):

        # Get mask
        mask = foreground_labels == k

        inst_scene_pc_array = xyz_img[mask, :]
        inst_scene_pc = o3d.geometry.PointCloud()
        inst_scene_pc.points = o3d.utility.Vector3dVector(inst_scene_pc_array.reshape((-1, 3)))

        if len(inst_scene_pc_array) <= num_points:
            continue

        # For background/table, prediction direction should point towards origin
        if k in [BACKGROUND_LABEL, TABLE_LABEL]:
            offsets[mask, ...] = 0
            continue

        # Compute 3D object centers in camera frame
        inst_pose_idx = np.where(obj_list == k-1)[0][0]
        obj_pose = pose_list[inst_pose_idx]

        sampled_points = obj_models[k-1]
        target_points = transform_points(sampled_points, obj_pose)
        target_points = transform_points(target_points, np.linalg.inv(camera_pose))

        # inst = o3d.geometry.PointCloud()
        # inst.points = o3d.utility.Vector3dVector(target_points)
        # inst.paint_uniform_color([1, 0, 0])
        # inst_pc_list.append(inst)
        cf_3D_center = np.mean(target_points, axis=0)
        #print(cf_3D_center)
        #print(xyz_img[mask, 0].min(), xyz_img[mask, 0].max())
        #print(xyz_img[mask, 1].min(), xyz_img[mask, 1].max())

        # If center isn't contained within the object, use point cloud average
        # TODO
        if cf_3D_center[0] < xyz_img[mask, 0].min() or \
           cf_3D_center[0] > xyz_img[mask, 0].max() or \
           cf_3D_center[1] < xyz_img[mask, 1].min() or \
           cf_3D_center[1] > xyz_img[mask, 1].max():
            cf_3D_center = xyz_img[mask, ...].mean(axis=0)

        # Get directions
        cf_3D_centers[i-2] = cf_3D_center
        object_center_offsets = (cf_3D_center - xyz_img).astype(np.float32) # Shape: [H x W x 3]

        # Add it to the labels
        offsets[mask, ...] = object_center_offsets[mask, ...]

    return offsets, cf_3D_centers


obj_models = load_obj_models()

for scene_idx in range(130):
    scene_dir = os.path.join(root, 'scenes', 'scene_{:04}'.format(scene_idx))
    for view_num in range(256):

        # meta info
        meta_filename = os.path.join(scene_dir, camera, 'meta', str(view_num).zfill(4) + ".mat")
        meta_info = scio.loadmat(meta_filename)
        intrinsic = meta_info['intrinsic_matrix']
        factor_depth = meta_info['factor_depth']

        camera_poses = np.load(os.path.join(scene_dir, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[view_num]
        scene_reader = xmlReader(os.path.join(scene_dir, camera, 'annotations', '%04d.xml' % view_num))
        pose_vectors = scene_reader.getposevectorlist()
        obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)

        scene_description = {}
        scene_description.update({'obj_list': obj_list})
        scene_description.update({'pose_list': pose_list})
        scene_description.update({'camera_pose': camera_pose})

        # Depth image
        depth_img_filename = os.path.join(scene_dir, camera, 'depth', str(view_num).zfill(4) + ".png")
        depth_img = np.array(Image.open(depth_img_filename)) # This reads a 16-bit single-channel image. Shape: [H x W]

        camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        xyz_img = create_point_cloud_from_depth_image(depth_img, camera_info, organized=True)

        # Labels
        foreground_labels_filename = os.path.join(scene_dir, camera, 'label', str(view_num).zfill(4) + ".png")
        foreground_labels = util_.imread_indexed(foreground_labels_filename)

        center_offset_labels, object_centers = process_label_3D(foreground_labels, xyz_img, scene_description, obj_models)

        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(xyz_img.reshape((-1, 3)))
        # scene.normals = o3d.utility.Vector3dVector(center_offset_labels.reshape((-1, 3)))
        # o3d.visualization.draw_geometries([scene])

        # scene.paint_uniform_color([0, 1, 0])

        save_path = os.path.join(save_root, 'scene_{:04}'.format(scene_idx), camera)
        if not os.path.exists(save_path): os.makedirs(save_path)
        np.savez_compressed(os.path.join(save_path, '{:04}.npz'.format(view_num)), offsets=center_offset_labels, centers=object_centers)
