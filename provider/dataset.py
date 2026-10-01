import os
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from data_augmentation import data_augment, get_rotation
import copy
# import open3d as o3d
from data_utils import load_depth, load_composed_depth, get_bbox, fill_missing

from common_utils import write_obj


class TrainingDataset(Dataset):
    def __init__(self, config, data_dir, data_type='real', num_img_per_epoch=-1, use_fill_miss=True, use_composed_img=True, per_obj='', text_features_file=''):
        self.config = config
        self.data_dir = data_dir
        self.data_type = data_type
        self.use_shape_aug = config.get("use_shape_aug", False)
        self.num_img_per_epoch = num_img_per_epoch

        self.use_fill_miss = use_fill_miss
        self.use_composed_img = use_composed_img

        self.img_size = self.config.img_size
        self.sample_num = self.config.sample_num

        if data_type == 'syn':
            img_path = 'CAMERA/train_list.txt'
            model_path = 'obj_models/camera_train.pkl'
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]
        elif data_type == 'real_withLabel':
            img_path = 'Real/train_list.txt'
            model_path = 'obj_models/real_train.pkl'
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            assert False, 'wrong data type of {} in data loader !'.format(data_type)

        img_list = [os.path.join(img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, img_path))]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        
        self.text_features_file = text_features_file
        self.text_features = {} 
        if self.text_features_file:
            try:
                self._load_text_features(self.text_features_file)
                print(f'Loaded precomputed text features from {self.text_features_file}')
            except Exception as e:
                print(f'WARNING: failed to load text features from {self.text_features_file}: {e}')

        if data_type == "syn":
            self.id2cat_name = self.id2cat_name_CAMERA
        else:
            self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.per_obj = per_obj
        self.per_obj_id = None
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{data_type}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
            img_list = img_list_obj

        self.img_list = img_list
        self.img_index = np.arange(len(self.img_list))


        self.models = {}
        with open(os.path.join(self.data_dir, model_path), 'rb') as f:
            self.models.update(cPickle.load(f))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    
        self.norm_scale = 1000.0    
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        print('{} images found.'.format(len(self.img_list)))
        print('{} models loaded.'.format(len(self.models)))
    def __len__(self):
        if self.num_img_per_epoch == -1:
            return len(self.img_list)
        else:
            return self.num_img_per_epoch

    def _load_text_descriptions(self):
        if not self.text_descriptions_dir or not os.path.exists(self.text_descriptions_dir):
            print(f"WARNING: text_descriptions_dir '{self.text_descriptions_dir}' does not exist or is empty.")
            return
        
        for cat_name in self.cat_names:
            txt_file = os.path.join(self.text_descriptions_dir, f'{cat_name}.txt')
            if not os.path.exists(txt_file):
                print(f"WARNING: Text description file for '{cat_name}' not found at {txt_file}")
                self.text_descriptions[cat_name] = {"2D": [], "3D": []}
                continue
            
            descriptions_2d = []
            descriptions_3d = []
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('2D:'):
                            desc = line[3:].strip() 
                            if desc:
                                descriptions_2d.append(desc)
                        elif line.startswith('3D:'):
                            desc = line[3:].strip()
                            if desc:
                                descriptions_3d.append(desc)
                
                self.text_descriptions[cat_name] = {
                    "2D": descriptions_2d,
                    "3D": descriptions_3d
                }
                print(f"Loaded {len(descriptions_2d)} 2D and {len(descriptions_3d)} 3D descriptions for '{cat_name}'")
            except Exception as e:
                print(f"ERROR loading text descriptions for '{cat_name}': {e}")
                self.text_descriptions[cat_name] = {"2D": [], "3D": []}
    
    def _get_text_descriptions(self, cat_id):
        if 0 <= cat_id < len(self.cat_names):
            cat_name = self.cat_names[cat_id]
        else:
            cat_name = None
        if not cat_name or cat_name not in self.text_descriptions or not self.text_descriptions[cat_name]["2D"] or not self.text_descriptions[cat_name]["3D"]:
            return "", ""
        else:
            text_texture = np.random.choice(self.text_descriptions[cat_name]["2D"])
            text_shape = np.random.choice(self.text_descriptions[cat_name]["3D"])
            return text_texture, text_shape

    def _load_text_features(self, features_path: str):
        if not os.path.exists(features_path):
            raise FileNotFoundError(features_path)
        if features_path.endswith('.npz') or features_path.endswith('.npz'):
            data = np.load(features_path, allow_pickle=True)
            for cat in self.cat_names:
                arr2k = f"{cat}_2D"
                arr3k = f"{cat}_3D"
                self.text_features.setdefault(cat, {"2D": [], "3D": []})
                if arr2k in data:
                    v = data[arr2k]
                    if hasattr(v, 'ndim') and v.ndim == 2:
                        for i in range(v.shape[0]):
                            self.text_features[cat]["2D"].append(v[i].astype(np.float32))
                if arr3k in data:
                    v = data[arr3k]
                    if hasattr(v, 'ndim') and v.ndim == 2:
                        for i in range(v.shape[0]):
                            self.text_features[cat]["3D"].append(v[i].astype(np.float32))
        else:
            try:
                import _pickle as pkl
                with open(features_path, 'rb') as f:
                    raw = pkl.load(f)
                for cat in self.cat_names:
                    self.text_features.setdefault(cat, {"2D": [], "3D": []})
                    if cat in raw:
                        for k in ('2D', '3D'):
                            vals = raw[cat].get(k, []) if isinstance(raw[cat], dict) else []
                            for v in vals:
                                arr = np.array(v, dtype=np.float32)
                                if arr.ndim == 1:
                                    self.text_features[cat][k].append(arr)
            except Exception as e:
                raise RuntimeError(f'Unsupported feature file format: {e}')

    def reset(self):
        assert self.num_img_per_epoch != -1
        num_img = len(self.img_list)
        if num_img <= self.num_img_per_epoch:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch, replace=False)

    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0] 
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]

        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax 
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az 
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

    def get_sym_info(self, c, mug_handle=1):
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int_)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int_)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int_)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int_)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int_)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int_)  
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int_)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int_)
        return sym


    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[self.img_index[index]])
        if self.data_type == 'syn' and self.use_composed_img:
            depth = load_composed_depth(img_path)
        else:
            depth = load_depth(img_path)
        if depth is None:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if self.use_fill_miss:
            depth = fill_missing(depth, self.norm_scale, 1) 

        # mask
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        num_instance = len(gts['instance_ids'])
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]

        if self.per_obj != '':
            idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][idx] - 1 
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        mask = np.equal(mask, gts['instance_ids'][idx])             

        mask = np.logical_and(mask , depth > 0)

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if len(choose) <= self.sample_num:
            choose_idx = np.random.choice(len(choose), self.sample_num)
        else:
            choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        pts2 = depth.copy() / self.norm_scale
        pts0 = (self.xmap - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) 
        pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)      

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] 

        mask1 = mask[rmin:rmax, cmin:cmax] 
        mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)
        rgb = rgb[rmin:rmax, cmin:cmax, :].copy() * mask1.astype(int)
        rgb = rgb.astype(np.uint8)

        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
        rgb = self.transform(np.array(rgb))

        # update choose
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts) 
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        
       
        if self.text_features:
            cat_name = self.cat_names[cat_id]
            feat2d_list = self.text_features[cat_name]["2D"]
            feat3d_list = self.text_features[cat_name]["3D"]
            if len(feat2d_list) > 0:
                idx2 = np.random.randint(len(feat2d_list))
                ret_dict['text_texture_feat'] = torch.from_numpy(np.array(feat2d_list[idx2], dtype=np.float32))
            else:
                ret_dict['text_texture_feat'] = torch.FloatTensor([])
            if len(feat3d_list) > 0:
                idx3 = np.random.randint(len(feat3d_list))
                ret_dict['text_shape_feat'] = torch.from_numpy(np.array(feat3d_list[idx3], dtype=np.float32))
            else:
                ret_dict['text_shape_feat'] = torch.FloatTensor([])
        else:
            ret_dict['text_texture_feat'] = torch.FloatTensor([])
            ret_dict['text_shape_feat'] = torch.FloatTensor([])
    

        if self.data_type == 'syn' or self.data_type == 'real_withLabel':
            model = self.models[gts['model_list'][idx]].astype(np.float32)
            translation = gts['translations'][idx].astype(np.float32)
            rotation = gts['rotations'][idx].astype(np.float32)
            size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map
            qo = (pts - translation[np.newaxis, :]) / (np.linalg.norm(size)+1e-8) @ rotation

            sRT = np.identity(4, dtype=np.float32)
            sRT[:3, :3] = gts['scales'][idx] * rotation
            sRT[:3, 3] = translation


            ret_dict['model'] = torch.FloatTensor(model)
            ret_dict['qo'] = torch.FloatTensor(qo)
            ret_dict['translation_label'] = torch.FloatTensor(translation)
            
            ret_dict['rotation_label'] = torch.FloatTensor(rotation)
            ret_dict['size_label'] = torch.FloatTensor(size)
            sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=1)
            ret_dict['sym_info'] =  torch.IntTensor(sym_info).long()
            # generate augmentation parameters
            if self.use_shape_aug:
                bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

                aug_bb = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
                aug_rt_t = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
                aug_rt_r = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous() 

                PC_da, gt_R_da, gt_t_da, gt_s_da, model_point, PC_nocs = data_augment(self.config, ret_dict['pts'], ret_dict['rotation_label'],
                                                                                        ret_dict['translation_label'], ret_dict['size_label'], sym_info,
                                                                                        aug_bb, aug_rt_t, aug_rt_r, ret_dict['model'], gts['scales'][idx], ret_dict['qo'],
                                                                                        ret_dict['category_label'])
                
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = torch.norm(gt_s_da) * gt_R_da
                sRT[:3, 3] = gt_t_da

                ret_dict['pts'] = PC_da
                ret_dict['rotation_label'] = gt_R_da
                ret_dict['translation_label'] = gt_t_da
                ret_dict['size_label'] = gt_s_da
                ret_dict['model'] = model_point
                ret_dict['qo'] = PC_nocs

        return ret_dict


class TestDataset():
    def __init__(self, config, data_dir, text_features_file):
        self.data_dir = data_dir

        self.img_size = config.img_size
        self.sample_num = config.sample_num
        self.data_name = config.data_name

        if self.data_name == 'real':
            model_path = 'obj_models/real_test.pkl'
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
            result_pkl_list = glob.glob(os.path.join(self.data_dir, 'data', 'segmentation_results', 'test_trainedwithMask', 'results_*.pkl'))
        else:
            model_path = 'obj_models/camera_val.pkl'
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]
            result_pkl_list = glob.glob(os.path.join(self.data_dir, 'data', 'segmentation_results', 'CAMERA25', 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.class_name_map = {1: 'bottle_',
                               2: 'bowl_',
                               3: 'camera_',
                               4: 'can_',
                               5: 'laptop_',
                               6: 'mug_'}
        self.models = {}
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.text_features_file = text_features_file
        self.text_features = {} 
        if self.text_features_file:
            try:
                self._load_text_features(self.text_features_file)
                print(f'Loaded precomputed text features from {self.text_features_file}')
            except Exception as e:
                print(f'WARNING: failed to load text features from {self.text_features_file}: {e}')
        with open(os.path.join(self.data_dir, 'data', model_path), 'rb') as f:
            self.models.update(cPickle.load(f))

    def __len__(self):
        return len(self.result_pkl_list)
    
    def _load_text_features(self, features_path: str):
        if not os.path.exists(features_path):
            raise FileNotFoundError(features_path)
        if features_path.endswith('.npz') or features_path.endswith('.npz'):
            data = np.load(features_path, allow_pickle=True)
            for cat in self.cat_names:
                arr2k = f"{cat}_2D"
                arr3k = f"{cat}_3D"
                self.text_features.setdefault(cat, {"2D": [], "3D": []})
                if arr2k in data:
                    v = data[arr2k]
                    if hasattr(v, 'ndim') and v.ndim == 2:
                        for i in range(v.shape[0]):
                            self.text_features[cat]["2D"].append(v[i].astype(np.float32))
                if arr3k in data:
                    v = data[arr3k]
                    if hasattr(v, 'ndim') and v.ndim == 2:
                        for i in range(v.shape[0]):
                            self.text_features[cat]["3D"].append(v[i].astype(np.float32))
        else:
            try:
                import _pickle as pkl
                with open(features_path, 'rb') as f:
                    raw = pkl.load(f)
                for cat in self.cat_names:
                    self.text_features.setdefault(cat, {"2D": [], "3D": []})
                    if cat in raw:
                        for k in ('2D', '3D'):
                            vals = raw[cat].get(k, []) if isinstance(raw[cat], dict) else []
                            for v in vals:
                                arr = np.array(v, dtype=np.float32)
                                if arr.ndim == 1:
                                    self.text_features[cat][k].append(arr)
            except Exception as e:
                raise RuntimeError(f'Unsupported feature file format: {e}')

    def __getitem__(self, index):
        path = self.result_pkl_list[index]

        with open(path, 'rb') as f:
            data = cPickle.load(f)

        if self.data_name == 'real':
            with open(os.path.join(self.data_dir, 'data', 'segmentation_results', 'test_trainedwoMask', path.split('/')[-1]), 'rb') as f:
                pred_data = cPickle.load(f)
            pred_mask = pred_data['pred_mask']

            image_path = os.path.join(self.data_dir, data['image_path']) 
            image_path = image_path.replace('/data/real/', '/data/Real/') 
        else:
            pred_data = data
            pred_mask = data['pred_masks']
            image_path = os.path.join(self.data_dir, data['image_path'])   
            image_path = image_path.replace('/data/camera/', '/data/CAMERA/') 


        num_instance = len(pred_data['pred_class_ids'])
        
        with open(image_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)

        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1]

        # nocs
        coord = cv2.imread(image_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        if self.data_name == 'real':
            depth = load_depth(image_path) 
        else:
            depth = load_composed_depth(image_path)
        if depth is None:
            # random choose
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)        
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        
        all_rgb = []
        all_nocs = []
        all_pts = []
        all_models = []
        all_cat_ids = []
        all_choose = []
        all_text_texture_feat = []
        all_text_shape_feat = []
        flag_instance = torch.zeros(num_instance) == 1 

        for j in range(num_instance):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(pred_data['pred_bboxes'][j])
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose)>16:
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

                instance_nocs = coord[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                mask1 = mask[rmin:rmax, cmin:cmax] 
                mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)
                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy() * mask1.astype(int)

                instance_rgb = instance_rgb.astype(np.uint8)
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                instance_rgb = self.transform(np.array(instance_rgb))
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                cat_id = pred_data['pred_class_ids'][j] - 1 
                class_name = self.class_name_map[pred_data['pred_class_ids'][j]]

                model = self.models[gts['model_list'][0]].astype(np.float32)
                for gt_class_name in gts['model_list']:
                    if class_name in gt_class_name:
                        model = self.models[gt_class_name].astype(np.float32)
                        break
                
                if self.text_features:
                    cat_name = self.cat_names[cat_id]
                    feat2d_list = self.text_features[cat_name]["2D"]
                    feat3d_list = self.text_features[cat_name]["3D"]
                    if len(feat2d_list) > 0:
                        idx2 = np.random.randint(len(feat2d_list))
                        text_texture_feat = torch.from_numpy(np.array(feat2d_list[idx2], dtype=np.float32))
                    else:
                        text_texture_feat = torch.FloatTensor([])
                    if len(feat3d_list) > 0:
                        idx3 = np.random.randint(len(feat3d_list))
                        text_shape_feat = torch.from_numpy(np.array(feat3d_list[idx3], dtype=np.float32))
                    else:
                        text_shape_feat = torch.FloatTensor([])  

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_nocs.append(torch.FloatTensor(instance_nocs))
                all_models.append(torch.FloatTensor(model))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                all_text_texture_feat.append(text_texture_feat)
                all_text_shape_feat.append(text_shape_feat)
                flag_instance[j] = 1
        
        if len(all_pts) == 0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)        

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_pts) # N*3
        ret_dict['rgb'] = torch.stack(all_rgb)
        ret_dict['ori_img'] = torch.tensor(cv2.imread(image_path + '_color.png')[:, :, :3])
        ret_dict['nocs'] = torch.stack(all_nocs)
        ret_dict['choose'] = torch.stack(all_choose)
        ret_dict['model'] = torch.stack(all_models)
        ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
        ret_dict['text_texture_feat'] = torch.stack(all_text_texture_feat)
        ret_dict['text_shape_feat'] = torch.stack(all_text_shape_feat)

        ret_dict['gt_class_ids'] = torch.tensor(data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(data['gt_handle_visibility'])

        ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
        ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
        ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]
        ret_dict['index'] = torch.IntTensor([index])
        return ret_dict
