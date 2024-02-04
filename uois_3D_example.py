#!/usr/bin/env python
# coding: utf-8

# # Unseen Object Instance Segmentation
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# My libraries. Ugly hack to import from sister directory
import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.flowlib as flowlib
from PIL import Image
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import scipy.io as scio

# ## Depth Seeding Network Parameters

dsn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10, 
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'subsample_factor' : 5,
    
    # Misc
    'min_pixels_thresh' : 500,
    'tau' : 15.,
    
}


# ## Region Refinement Network parameters


rrn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,
    
    # architecture parameters
    'use_coordconv' : False,
    
}


# # UOIS-Net-3D Parameters

uois3d_config = {
    
    # Padding for RGB Refinement Network
    'padding_percentage' : 0.25,
    
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology' : True,
    'open_close_morphology_ksize' : 9,
    
    # Largest Connected Component for IMP module
    'use_largest_connected_component' : True,
    
}

# Biqi
checkpoint_dir = './checkpoints/' # TODO: change this to directory of downloaded models
# Biqi
dsn_filename = checkpoint_dir + 'DSNWrapper_iter76801_64c_checkpoint.pth'
# dsn_filename = checkpoint_dir + 'DSNWrapper_iter150000_64c_checkpoint.pth'
rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
uois3d_config['final_close_morphology'] = True
uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config
                                    )


# ## Run on example OSD/OCID images
# 
# We provide a few [OSD](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/osd/) and [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) images and run the network on them. Evaluation metrics are shown for each of the images.

# In[ ]:

scene_idx = 0
frame_idx = 20
camera = 'realsense'
N = 4
dataset_root = ''
rgb_path = '/media/rcao/Data/Dataset/graspnet/scenes/scene_{:04}/{}/rgb/{:04}.png'.format(scene_idx, camera, frame_idx)
depth_path = '/media/rcao/Data/Dataset/graspnet/scenes/scene_{:04}/{}/depth/{:04}.png'.format(scene_idx, camera, frame_idx)
mask_path = '/media/rcao/Data/Dataset/graspnet/scenes/scene_{:04}/{}/label/{:04}.png'.format(scene_idx, camera, frame_idx)
meta_path = '/media/rcao/Data/Dataset/graspnet/scenes/scene_{:04}/{}/meta/{:04}.mat'.format(scene_idx, camera, frame_idx)

width = 1280
height = 720

# example_images_dir = os.path.abspath('.') + '/example_images/'
# # Biqi
# OSD_image_files = sorted(glob.glob(example_images_dir + '/OSD_*.npy'))
# OCID_image_files = sorted(glob.glob(example_images_dir + '/OCID_*.npy'))
# N = len(OSD_image_files) + len(OCID_image_files)

rgb_imgs = np.zeros((N, height, width, 3), dtype=np.float32)
xyz_imgs = np.zeros((N, height, width, 3), dtype=np.float32)
label_imgs = np.zeros((N, height, width), dtype=np.uint8)

for i, img_file in enumerate(range(N)):
# for i, img_file in enumerate(OSD_image_files + OCID_image_files):
#     d = np.load(img_file, allow_pickle=True, encoding='bytes').item()

    color = np.array(Image.open(rgb_path), dtype=np.float32)
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(mask_path))

    meta = scio.loadmat(meta_path)

    intrinsics = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # RGB
    rgb_img = color
    rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

    # XYZ
    xyz_imgs[i] = cloud

    # Label
    label_imgs[i] = seg
    
batch = {
    'rgb': data_augmentation.array_to_tensor(rgb_imgs),
    'xyz': data_augmentation.array_to_tensor(xyz_imgs),
}


print("Number of images: {0}".format(N))

### Compute segmentation masks ###
st_time = time()
fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
total_time = time() - st_time
print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
print('FPS: {0}'.format(round(N / total_time,3)))

# Get results in numpy
seg_masks = seg_masks.cpu().numpy()
fg_masks = fg_masks.cpu().numpy()
center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
initial_masks = initial_masks.cpu().numpy()

rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
total_subplots = 6

fig_index = 1
for i in range(N):
    
    num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1
    rgb = rgb_imgs[i].astype(np.uint8)
    depth = xyz_imgs[i,...,2]
    seg_mask_plot = util_.get_color_mask(seg_masks[i,...], nc=num_objs)
    init_mask_plot = util_.get_color_mask(initial_masks[i,...], nc=num_objs)
    gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)
    
    images = [rgb, depth, init_mask_plot, seg_mask_plot, gt_masks]
    titles = [f'Image {i+1}', 'Depth',
              f"Init Masks. #objects: {np.unique(initial_masks[i,...]).shape[0]-1}",
              f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
              f"Ground Truth. #objects: {np.unique(label_imgs[i,...]).shape[0]-1}"
             ]
    util_.subplotter(images, titles, fig_num=i+1)
    
    # Run evaluation metric
    eval_metrics = evaluation.multilabel_metrics(initial_masks[i,...], label_imgs[i])
    print(f"Image {i+1} Metrics:")
    print(eval_metrics)

