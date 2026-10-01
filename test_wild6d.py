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
# from lib.network import PoseNetV3
# from lib.shape_prior import DeformNet
# from lib.align import estimateSimilarityTransform, RansacPnP
from lib.utils import get_bbox_from_mask, compute_mAP, plot_mAP, zoom_in, xywh_to_cs, load_obj
from lib.transformations import quaternion_matrix
from lib.utils import (get_bbox, calculate_2d_projections, get_3d_bbox, load_depth,
                         transform_coordinates_3d, draw_bboxes, compute_3d_IoU)
import pdb
import gorilla

import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

xmap = np.array([[i for i in range(480)] for j in range(640)])
ymap = np.array([[j for i in range(480)] for j in range(640)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

if opt.result_dir is None:
    result_dir = osp.join('/mnt/HDD6/dzq/GenCOPE/results/Wild6D_results/', opt.model.split('/')[-1], opt.select_class)
else:
    result_dir = opt.result_dir

def detect():
    file_path = 'test_list_{}.txt'.format(opt.select_class)
    img_list = [line.rstrip('\n').replace('data','/mnt/HDD1/dataset').replace('rgbd', 'images').replace('UCSD_POSE_RGBD', 'Wild6D') \
        for line in open(os.path.join(opt.data_dir, file_path))]
    norm_scale = 1000.0
    print(len(img_list),img_list[0])

    inst_count = 0
    img_count = 0
    from model.GenCOPE import GenCOPE
    estimator = GenCOPE()
    estimator.cuda()

    checkpoint = os.path.join(opt.model, 'epoch_' + str(opt.test_epoch) + '.pth')
    print("Loading model from {}".format(checkpoint))
    gorilla.solver.load_checkpoint(model=estimator, filename=checkpoint)
    estimator.eval()


    if not osp.exists(result_dir):
        os.makedirs(result_dir)
        os.makedirs(osp.join(result_dir, 'vis'))
    

    for num_f, img_path in tqdm.tqdm(enumerate(img_list)):
        img_name = opt.select_class+'_'+img_path.split('/')[-4]+'_'+img_path.split('/')[-3] + '_' + \
        '{:04d}'.format(int(img_path.split('/')[-1].split('.jpg')[0])) + '.jpg'

        mask_path = img_path.replace('.jpg', '-mask.png')
        depth_path = img_path.replace('.jpg', '-depth.png')
        if not osp.exists(mask_path):
            continue
        raw_rgb = cv2.imread(img_path)
        raw_rgb = raw_rgb[:, :, ::-1]

        mask = cv2.imread(mask_path)[:, :, 2]
        mask = mask / 255.
        raw_depth = cv2.imread(depth_path, -1)

        num_insts = 1
        meta = json.load(open(osp.join(opt.data_dir, opt.select_class, img_path.split('/')[-4], img_path.split('/')[-3], 'metadata')))   
        cam = np.array(meta['K']).reshape(3, 3).T
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]
        cam_cx = cam[0, 2] 
        cam_cy = cam[1, 2]

        pts2 = raw_depth.copy() / norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3


        for i in range(num_insts):
            cat_id = cat_names.index(opt.select_class)
            f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
            f_size = np.zeros((num_insts, 3), dtype=float)
            # prepare frame data
            f_points, f_rgb, f_choose, f_catId, f_prior = [], [], [], [], []
            f_box_s, f_box_c, f_cam, f_bbox = [], [], [], []
            valid_inst = []
            # obtain box
            horizontal_indicies = np.where(np.any(mask, axis=0))[0]
            vertical_indicies = np.where(np.any(mask, axis=1))[0]
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            bbox = np.array([x1, y1, x2-x1, y2-y1]).astype(np.int_)
            c, s = xywh_to_cs(bbox, 1.2, s_max=max(480, 640))
            rgb, c_h_, c_w_, s_, crop_bbox = zoom_in(raw_rgb, c, s, opt.img_size)
            rmin, rmax, cmin, cmax = crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]
            box_c = np.array([c_w_, c_h_])
            box_s = s_

            mask = np.logical_and(mask, raw_depth > 0).astype(np.float32)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        
            mask1 = mask[rmin:rmax, cmin:cmax] 
            mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)
            mask1 = cv2.resize(mask1, (192, 192), interpolation=cv2.INTER_LINEAR)
            rgb = rgb.copy() * mask1.astype(int)
            rgb = rgb.astype(np.uint8)

            if len(choose) < 32 or cat_id != cat_names.index(opt.select_class):
                f_sRT[i] = np.identity(4, dtype=float)
                continue
            else:
                valid_inst.append(i)

            if len(choose) <= opt.n_pts:
                choose_idx = np.random.choice(len(choose), opt.n_pts)
            else:
                choose_idx = np.random.choice(len(choose), opt.n_pts, replace=False)
            choose = choose[choose_idx]

            points = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

            rgb = norm_color(rgb)
            crop_w = cmax - cmin
            crop_h = rmax - rmin
            ratio_w = opt.img_size / float(crop_w)
            ratio_h = opt.img_size / float(crop_h)
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio_h) * opt.img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
            
            f_points.append(points)
            f_rgb.append(rgb)
            f_choose.append(choose)
            f_catId.append(cat_id)
            # f_prior.append(verts)
            f_box_c.append(box_c)
            f_box_s.append(box_s)
            f_bbox.append(bbox)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            f_catId = torch.cuda.LongTensor(f_catId)
            # f_prior = torch.cuda.FloatTensor(f_prior)
            f_box_c = torch.cuda.FloatTensor(f_box_c)
            f_box_s = torch.cuda.FloatTensor(f_box_s)
            f_cam = torch.cuda.FloatTensor(f_cam)
            f_bbox = torch.cuda.FloatTensor(f_bbox)
            # inference
            torch.cuda.synchronize()
            inputs = {}
            inputs["rgb"] = f_rgb
            inputs["pts"] = f_points
            inputs["choose"] = f_choose
            inputs['category_label'] = f_catId
            outputs = estimator(inputs)
            torch.cuda.synchronize()
            pred_translation = outputs['pred_translation']
            pred_size = outputs['pred_size']
            pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
            pred_size = pred_size / pred_scale
            pred_rotation = outputs['pred_rotation']

            num_instance = pred_rotation.size(0)
            pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
            pred_RTs[:, :3, 3] = pred_translation
            pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
            pred_scales = pred_size

            f_sRT = pred_RTs.detach().cpu().numpy()
            f_size = pred_scales.detach().cpu().numpy()

        # save results
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
        
        result['gt_class_ids'] = gts['class_id']
        result['gt_bboxes'] = np.array([y1, x1, y2, x2])
        gt_RTs = np.eye(4)
        gt_RTs[:3, :3] = gts['rotation']
        gt_RTs[:3, 3] = gts['translation']
        result['gt_RTs'] = gt_RTs[np.newaxis, ...]
        result['gt_scales'] = gts['size'][np.newaxis, ...]
        result['gt_handle_visibility'] = 1

        result['pred_class_ids'] = gts['class_id']
        result['pred_bboxes'] = np.array([[y1, x1, y2, x2]])
        result['pred_scores'] = 1.0
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size
        
        frame_name = gts['name'].replace('/', '_')
        save_path = osp.join(result_dir, 'results_{}.pkl'.format(frame_name))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)
        
        
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

def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    print(result_dir)
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            class_idx = cat_names.index(opt.select_class) + 1
            result['gt_class_ids'] = np.array([class_idx], dtype=np.int32)
            result['gt_bboxes'] = result['gt_bboxes'][np.newaxis, ...]
            result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            result['pred_class_ids'] = np.array([class_idx], dtype=np.int32)
            result['pred_scores'] = np.array([result['pred_scores']])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False

    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, 
                                                       select_class=opt.select_class)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':
    print('Detecting ...')
    if not opt.only_eval:
        detect()
    print('Evaluating ...')
    evaluate()
