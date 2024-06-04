import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing

import argparse
from src.evaluation import multilabel_metrics


def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n')) # 删除\n
    return data_list


def eval_scene(image_path, cfgs):
    result = np.zeros((6))
    dataset_root = cfgs.dataset_root
    segment_root = cfgs.segment_result
    segment_method = cfgs.segment_method
    image_name = os.path.basename(image_path).split('.')[0]
    image_dir = os.path.join(*os.path.dirname(image_path).split('/')[1:-1])
    
    gt_mask_path = os.path.join(dataset_root, 'data', image_path.replace('rgb', 'label'))
    pred_mask_path = os.path.join(segment_root, '{}_mask'.format(segment_method), image_dir, '{}.png'.format(image_name))

    gt_mask = np.array(Image.open(gt_mask_path))
    pred_mask = np.array(Image.open(pred_mask_path))
    eval_metrics = multilabel_metrics(gt_mask, pred_mask)
    
    result[0] = eval_metrics['Objects F-measure']
    result[1] = eval_metrics['Objects Precision']
    result[2] = eval_metrics['Objects Recall']
    result[3] = eval_metrics['Boundary F-measure']
    result[4] = eval_metrics['Boundary Precision']
    result[5] = eval_metrics['Boundary Recall']
    return result


def parallel_eval(image_list, cfgs, proc=2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes=proc)
    result_list = []
    for image_path in image_list:
        scene_result = p.apply_async(eval_scene, (image_path, cfgs))
        result_list.append(scene_result)
    p.close()
    p.join()
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/OCID')
    parser.add_argument('--segment_result', default='/media/gpuadmin/rcao/result/uois/ocid')
    parser.add_argument('--segment_method', default='GDS_v0.3.1', help='Segmentation method [uois/uoais/GDS]')
    cfgs = parser.parse_args()

    image_list = read_file(os.path.join(cfgs.dataset_root, 'data_list.txt'))
    result_list = parallel_eval(image_list, cfgs=cfgs, proc=20)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    
    print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}'. \
        format(np.mean(results[:, 1]), np.mean(results[:, 2]), np.mean(results[:, 0]),
               np.mean(results[:, 4]), np.mean(results[:, 5]), np.mean(results[:, 3])))
    np.save('OCID_{}_results.npy'.format(cfgs.segment_method), results)
