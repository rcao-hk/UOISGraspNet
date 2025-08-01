import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing

import argparse
from src.evaluation import multilabel_metrics


def eval_scene(scene_idx, cfgs):
    result = np.zeros((256, 7))
    dataset_root = cfgs.dataset_root
    segment_root = cfgs.segment_result
    camera = cfgs.camera_type
    segment_method = cfgs.segment_method  # 'GDS' 'uois'
    for frame_idx in range(256):

        gt_mask_path = os.path.join(dataset_root, 'scenes/scene_{:04}/{}/label/{:04}.png'.format(scene_idx, camera, frame_idx))
        pred_mask_path = os.path.join(segment_root, '{}_mask/scene_{:04}/{}/{:04}.png'.format(segment_method, scene_idx, camera, frame_idx))

        gt_mask = np.array(Image.open(gt_mask_path))
        pred_mask = np.array(Image.open(pred_mask_path))
        eval_metrics = multilabel_metrics(pred_mask, gt_mask)
        
        result[frame_idx, 0] = eval_metrics['Objects F-measure']
        result[frame_idx, 1] = eval_metrics['Objects Precision']
        result[frame_idx, 2] = eval_metrics['Objects Recall']
        result[frame_idx, 3] = eval_metrics['Boundary F-measure']
        result[frame_idx, 4] = eval_metrics['Boundary Precision']
        result[frame_idx, 5] = eval_metrics['Boundary Recall']
        result[frame_idx, 6] = eval_metrics['obj_detected_075_percentage']
    return result


def parallel_eval(scene_ids, cfgs, proc=2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes=proc)
    result_list = []
    for scene_id in scene_ids:
        scene_result = p.apply_async(eval_scene, (scene_id, cfgs))
        result_list.append(scene_result)
    p.close()
    p.join()
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet')
    parser.add_argument('--segment_result', default='/media/gpuadmin/rcao/result/uois/graspnet')
    parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')
    parser.add_argument('--segment_method', default='GDS_v0.3.3', help='Segmentation method [uois/uoais/GDS]')
    cfgs = parser.parse_args()
    print(cfgs)
    
    scene_list = list(range(100, 190))
    # scene_list = list(range(100, 130))
    # scene_list = list(range(130, 160))
    # scene_list = list(range(160, 190))
    result_list = parallel_eval(scene_list, cfgs=cfgs, proc=20)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    
    print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}, %75:{}'. \
        format(np.mean(results[:, :, 1]), np.mean(results[:, :, 2]), np.mean(results[:, :, 0]),
               np.mean(results[:, :, 4]), np.mean(results[:, :, 5]), np.mean(results[:, :, 3]), 
               np.mean(results[:, :, 6])))
    np.save('Graspnet_{}_{}_results.npy'.format(cfgs.segment_method, cfgs.camera_type), results)
