import os
import os.path as osp
import glob
import tqdm
import matplotlib.pyplot as plt
import json
import cv2
import sys
import numpy as np
import PIL
from PIL import Image
import _pickle as cPickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.utils import get_bbox_from_mask, compute_mAP, plot_mAP, zoom_in, xywh_to_cs, load_obj
from lib.transformations import quaternion_matrix
from lib.utils import (get_bbox, calculate_2d_projections, get_3d_bbox, load_depth,
                         transform_coordinates_3d, draw_bboxes, compute_3d_IoU)
import pdb
import gorilla

import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

gorilla.utils.set_cuda_visible_devices(gpu_ids = "2")


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='/mnt/HDD1/dataset/Wild6D/test_set', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='/mnt/HDD6/dzq/GenCOPE/log/backbone/GenCOPE', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--select_class', type=str, default='bowl', help='resume from saved model')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--use_nocs_map', action='store_true')
parser.add_argument('--implicit', action='store_true')
parser.add_argument('--max_point', action='store_true')
parser.add_argument('--with_recon', action='store_true')
parser.add_argument('--test_epoch', default=1000, type=int, help='test epoch')
parser.add_argument('--result_dir', type=str, default=None)

opt = parser.parse_args()

cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
if opt.result_dir is None:
    result_dir = osp.join('/mnt/HDD6/dzq/GenCOPE/results/Wild6D_results/', opt.model.split('/')[-1],opt.select_class)
else:
    result_dir = opt.result_dir


def detect():
    file_path = 'test_list_{}.txt'.format(opt.select_class)
    img_list = [line.rstrip('\n').replace('data','/mnt/HDD1/dataset').replace('rgbd', 'images').replace('UCSD_POSE_RGBD', 'Wild6D') \
        for line in open(os.path.join(opt.data_dir, file_path))]
    print(len(img_list),img_list[0])

    if not osp.exists(osp.join(result_dir, 'vis')):
        os.makedirs(osp.join(result_dir, 'vis'))

    for num_f, img_path in tqdm.tqdm(enumerate(img_list)):

        meta_path = osp.join(opt.data_dir, opt.select_class, img_path.split('/')[-4], img_path.split('/')[-3], 'metadata')
        if not os.path.exists(meta_path):
            print("Not found the meta from {}".format(meta_path))
            continue
        meta = json.load(open(meta_path))
        cam = np.array(meta['K']).reshape(3, 3).T

        result = {}
        frame_idx = int(img_path.split('/')[-1].split('.jpg')[0])
        gt_path = osp.join('/mnt/HDD1/dataset/Wild6D/test_set/pkl_annotations/', opt.select_class, \
                opt.select_class+'-'+img_path.split('/')[-4]+'-'+img_path.split('/')[-3]+'.pkl')
        if not os.path.exists(gt_path):
            print("Not found the ground truth from {}".format(gt_path))
            continue
        gts = cPickle.load(open(gt_path, 'rb'))
        if frame_idx >= len(gts['annotations']):
            continue
        gts = gts['annotations'][frame_idx]
        
        gt_RTs = np.eye(4)
        gt_RTs[:3, :3] = gts['rotation']
        gt_RTs[:3, 3] = gts['translation']
        
        frame_name = gts['name'].replace('/', '_')
        save_path = osp.join(result_dir, 'results_{}.pkl'.format(frame_name))
        if not os.path.exists(save_path):
            print("Not found the save path from {}".format(save_path))
            continue
        result = cPickle.load(open(save_path, 'rb'))

        f_size = result['pred_scales']
        f_sRT = result['pred_RTs']
        
        img = cv2.imread(img_path)
        noc_cube_1 = get_3d_bbox(gts['size'], 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, gt_RTs)
        projected_bbox_1 = calculate_2d_projections(bbox_3d_1, cam)
        img = draw_bboxes(img, projected_bbox_1, (0, 255, 0))
        
        noc_cube_2 = get_3d_bbox(f_size[0], 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, f_sRT[0])
        projected_bbox_2 = calculate_2d_projections(bbox_3d_2, cam)
        img = draw_bboxes(img, projected_bbox_2, (0, 0, 255))
        
        img_save = PIL.Image.fromarray(img[:, :, ::-1])
        save_img_path = osp.join(result_dir, 'vis', 'results_{}.jpg'.format(frame_name))
        img_save.save(save_img_path)

if __name__ == '__main__':
    print('Detecting ...')
    detect()
