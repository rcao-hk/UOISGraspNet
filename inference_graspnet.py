#!/usr/bin/env python
# coding: utf-8

# TODO: Change this if you have more than 1 GPU
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import matplotlib.pyplot as plt

import resource

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import torch
import random
from PIL import Image
from time import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)

# My libraries
import src.data_loader_graspnet as data_loader
import src.segmentation as segmentation
import src.train as train
import src.util.utilities as util_
import src.util.flowlib as flowlib

# TODO: change this to the dataset you want to train on
graspnet_root = '/home/user/dataset/graspnet/'
mask_save_root = '/home/user/rcao/result/uois'
camera_type = 'realsense'
mask_save_root = os.path.join(mask_save_root, 'uois_v0.1_mask')
data_loading_params = {

    # Camera/Frustum parameters
    'img_width': 1280,
    'img_height': 720,
    'near': 0.01,
    'far': 100,
    'fov': 45,  # vertical field of view in degrees

    'camera': camera_type,
    'use_data_augmentation': False,

    # Multiplicative noise
    'gamma_shape': 1000.,
    'gamma_scale': 0.001,

    # Additive noise
    'gaussian_scale_range': [0., 0.003],  # up to 2.5mm standard dev
    'gp_rescale_factor_range': [12, 20],  # [low, high (exclusive)]

    # Random ellipse dropout
    'ellipse_dropout_mean': 10,
    'ellipse_gamma_shape': 5.0,
    'ellipse_gamma_scale': 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean': 15,
    'gradient_dropout_alpha': 2.,
    'gradient_dropout_beta': 5.,

    # Random pixel dropout
    'pixel_dropout_alpha': 0.2,
    'pixel_dropout_beta': 10.,
}

dsn_config = {

    # Sizes
    'feature_dim': 64,  # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters': 10,
    'num_seeds': 200,  # Used for MeanShift, but not BlurringMeanShift
    'epsilon': 0.05,  # Connected Components parameter
    'sigma': 0.02,  # Gaussian bandwidth parameter
    'subsample_factor': 5,
    'min_pixels_thresh': 500,

    # Differentiable backtracing params
    'tau': 15.,
    'M_threshold': 0.3,

    # Robustness stuff
    'angle_discretization': 100,
    'discretization_threshold': 0.,

}

uois3d_config = {

    # Padding for RGB Refinement Network
    'padding_percentage': 0.25,

    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology': True,
    'open_close_morphology_ksize': 9,

    # Largest Connected Component for IMP module
    'use_largest_connected_component': True,

}

rrn_config = {

    # Sizes
    'feature_dim': 64,  # 32 would be normal
    'img_H': 224,
    'img_W': 224,

    # architecture parameters
    'use_coordconv': False,

}

iter_num = 150000
# dsn_filename = os.path.join('checkpoints', f'DSNWrapper_iter{iter_num}_64c_checkpoint.pth')
dsn_filename = os.path.join('checkpoints', camera_type, 'train_DSN', f'DSNWrapper_iter{iter_num}_64c_checkpoint.pth')
rrn_filename = os.path.join('checkpoints', 'RRN_OID_checkpoint.pth')
uois3d_config['final_close_morphology'] = True
uois_net_3d = segmentation.UOISNet3D(uois3d_config,
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config)

# # Train the network for 1 epoch
# num_epochs = 12
# dsn_trainer.train(num_epochs, dl)
# dsn_trainer.save()

# ## Visualize some stuff
dl = data_loader.get_TOD_test_dataloader(
    graspnet_root, data_loading_params, batch_size=1, num_workers=0, shuffle=True, percompute_center=False)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i, batch in enumerate(dl):
    rgb_imgs = util_.torch_to_numpy(
        batch['rgb'], is_standardized_image=True)  # Shape: [N x H x W x 3]
    xyz_imgs = util_.torch_to_numpy(batch['xyz'])  # Shape: [N x H x W x 3]
    foreground_labels = util_.torch_to_numpy(
        batch['foreground_labels'])  # Shape: [N x H x W]
    # print(np.unique(foreground_labels))

    # center_offset_labels = util_.torch_to_numpy(
    #     batch['center_offset_labels'])  # Shape: [N x 2 x H x W]
    N, H, W = foreground_labels.shape[:3]

    ### Compute segmentation masks ###

    start.record()        
    # st_time = time()
    # fg_masks, center_offsets, object_centers, initial_masks = dsn.run_on_batch(
    #     batch)

    fg_masks, center_offsets, object_centers, dsn_masks, initial_masks, refined_masks = uois_net_3d.run_on_batch(batch)

    # total_time = time() - st_time
    # print('Total time taken for Segmentation: {0} seconds'.format(
    #     round(total_time, 3)))
    # print('FPS: {0}'.format(round(N / total_time, 3)))
    
    end.record()
    torch.cuda.synchronize()
    elasped_time = start.elapsed_time(end)
    print('Total elasped time: {} ms'.format(elasped_time))

    # Get segmentation masks in numpy
    fg_masks = fg_masks.cpu().numpy()
    center_offsets = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)
    dsn_masks = dsn_masks.cpu().numpy()
    initial_masks = initial_masks[0].cpu().numpy()
    refined_masks = refined_masks[0].cpu().numpy()

    scene_idx = int(batch['scene_dir'][0].split('/')[-2].split('_')[-1])
    view_num = batch['view_num'][0]

    # refined_masks = (refined_masks / np.max(refined_masks)) * 255
    # result = Image.fromarray(refined_masks.astype(np.uint8))
    # mask_save_path = os.path.join(mask_save_root, 'scene_{:04}'.format(scene_idx), camera_type)
    # os.makedirs(mask_save_path, exist_ok=True)
    # result.save(os.path.join(mask_save_path, '{:04}.png'.format(view_num)))

    # fig_index = 1
    # for i in range(N):
    #
    #     fig = plt.figure(fig_index)
    #     fig_index += 1
    #     fig.set_size_inches(20, 5)
    #
    #     # Plot image
    #     plt.subplot(1, 5, 1)
    #     plt.imshow(rgb_imgs[i, ...].astype(np.uint8))
    #     plt.title('Image {0}'.format(i+1))
    #
    #     # Plot Depth
    #     plt.subplot(1, 5, 2)
    #     plt.imshow(xyz_imgs[i, ..., 2])
    #     plt.title('Depth')
    #
    #     # Plot prediction
    #     plt.subplot(1, 5, 3)
    #     plt.imshow(util_.get_color_mask(fg_masks[i, ...]))
    #     plt.title("Predicted Masks")
    #
    #     # Plot Center Direction Predictions
    #     plt.subplot(1, 5, 4)
    #     fg_mask = np.expand_dims(fg_masks[i, ...] == 1, axis=-1)
    #     plt.imshow(flowlib.flow_to_image(center_offsets[i, ...] * fg_mask))
    #     plt.title("Center Direction Predictions")
    #
    #     # Plot Initial Masks
    #     plt.subplot(1, 5, 5)
    #     plt.imshow(util_.get_color_mask(refined_masks))
    #     plt.title(
    #         f"Initial Masks. #objects: {np.unique(refined_masks).shape[0]-1}")
