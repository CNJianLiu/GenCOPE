import os
import time
import logging
from tqdm import tqdm
import pickle as cPickle
import numpy as np
import torch
import torch.optim as optim
import numpy as np
import gorilla
from tensorboardX import SummaryWriter
from common_utils import write_obj
from scheduler import BNMomentumScheduler #, CyclicLR
import torch.nn.functional as F
from evaluation_utils import compute_3d_matches_for_each_gt
from vis_utils import draw_detections

class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, data_mode, loss, dataloaders, logger, cfg, start_epoch=1, start_iter=0):
        super(Solver, self).__init__(
            model=model,
            dataloaders=dataloaders,
            cfg=cfg,
            logger=logger,
        )
        self.loss = loss
        self.data_mode = data_mode
        self.logger.propagate = 0

        tb_writer_ = tools_writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer = tb_writer_

        self.per_val = cfg.per_val
        self.per_write = cfg.per_write
        self.epoch = start_epoch
        self.iter = start_iter
        if cfg.get("freeze_world_enhancer", False):
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        
        self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-6, max_lr=1e-4,
                                            step_size_up=cfg.max_epoch * cfg.num_mini_batch_per_epoch // 200, mode='triangular', cycle_momentum=False)

        bnm_lmbd = lambda it: max(cfg.bn.bn_momentum*cfg.bn.bn_decay**(int(it / cfg.bn.decay_step)), cfg.bn.bnm_clip)
        self.bnm_scheduler = BNMomentumScheduler(self.model, bn_lambda=bnm_lmbd, last_epoch=self.iter)

    def solve(self):
        while self.epoch <= self.cfg.max_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value

            if self.epoch % 10 == 0:
                ckpt_path = os.path.join(
                    self.cfg.log_dir, 'epoch_' + str(self.epoch) + '.pth')
                gorilla.solver.save_checkpoint(
                    model=self.model, filename=ckpt_path, optimizer=self.optimizer, meta={'iter': self.iter, "epoch": self.epoch})

            if self.epoch % 100 == 0:
                self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.8 ** (self.epoch/100) * 1e-6, max_lr=0.8 ** (self.epoch/100) * 1e-4, step_size_up=10000, mode='triangular', cycle_momentum=False)

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1

    # sys
    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()

        self.dataloaders["syn"].dataset.reset()

        i=0

        if not hasattr(self, 'log_buffer'):
            class SimpleLogBuffer:
                def __init__(self):
                    self._output = {}
                    self.avg = {}
                def update(self, d):
                    for k, v in d.items():
                        if k not in self._output:
                            self._output[k] = []
                        self._output[k].append(v)
                def average(self, n):
                    self.avg = {}
                    out = {}
                    for k, v_list in self._output.items():
                        if len(v_list) > 0:
                            val = sum(v_list[-n:]) / min(n, len(v_list[-n:]))
                            self.avg[k] = val
                            out[k] = float(val)
                    self._output = out
                    return out
                def clear(self):
                    self._output = {}
                    self.avg = {}
            self.log_buffer = SimpleLogBuffer()

        for syn_data in self.dataloaders["syn"]:
            data_time = time.time()-end

            if self.bnm_scheduler is not None:
                self.bnm_scheduler.step(self.iter)

            self.optimizer.zero_grad()
            loss, dict_info_step = self.step(syn_data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except TypeError:
                    self.lr_scheduler.step(self.iter)
            backward_time = time.time() - end - forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                self.log_buffer.average(self.per_write)
                prefix = '[{}/{}][{}/{}][{}] Train - '.format(
                    self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["syn"]), self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                try:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), write_info, flush=True)
                except Exception:
                    pass
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()

            self.iter += 1
            i+=1

        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch


    def evaluate(self):
        mode = 'eval'
        self.model.eval()

        for i, data in enumerate(self.dataloaders["eval"]):
            with torch.no_grad():
                _, dict_info_step = self.step(data, mode)
                self.log_buffer.update(dict_info_step)
                if i % self.per_write == 0:
                    self.log_buffer.average(self.per_write)
                    prefix = '[{}/{}][{}/{}] Test - '.format(
                        self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["eval"]))
                    write_info = self.get_logger_info(
                        prefix, dict_info=self.log_buffer._output)
                    self.logger.info(write_info)
                    self.write_summary(self.log_buffer._output, mode)
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch


    # sys
    def step(self, syn_data, mode):
        torch.cuda.synchronize()
        b1 = syn_data['rgb'].size(0)

        for key in syn_data:
            if torch.is_tensor(syn_data[key]):
                syn_data[key] = syn_data[key].cuda()

        data = {
            'rgb': syn_data['rgb'],
            'pts': syn_data['pts'],
            'choose': syn_data['choose'],
            'category_label': syn_data['category_label'],
            'model': syn_data['model'],
            'sym_info': syn_data['sym_info'],
            'gt_R': syn_data['rotation_label'],
            'gt_t': syn_data['translation_label'],
            'gt_s': syn_data['size_label'],
            'qo': syn_data['qo'],
            'text_texture_feat': syn_data['text_texture_feat'],
            'text_shape_feat': syn_data['text_shape_feat'],
        }
        end_points = self.model(data)

        for key in end_points:
            syn_data[key] = end_points[key][0:b1]

        loss_syn = self.loss['syn'](syn_data)

        loss_all = loss_syn

        dict_info = {
            'loss_all': float(loss_all.item()),
            'loss_syn': float(loss_syn.item()),
        }

        try:
            loss_module = self.loss.get('syn', None)
            if loss_module is not None and hasattr(loss_module, 'last_loss_dict') and loss_module.last_loss_dict is not None:
                for k, v in loss_module.last_loss_dict.items():
                    if k not in dict_info:
                        dict_info[k] = float(v)
        except Exception:
            pass

        if mode == 'train':
            if self.lr_scheduler is not None:
                try:
                    dict_info['lr'] = self.lr_scheduler.get_last_lr()[0]
                except Exception:
                    try:
                        dict_info['lr'] = self.lr_scheduler.get_lr()[0]
                    except Exception:
                        dict_info['lr'] = 0.0

        return loss_all, dict_info


    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False


def test_func(model, dataloder, save_path):
    model.eval()
    total_time = 0
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            path = dataloder.dataset.result_pkl_list[i]
            result = {}
            # save
            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()

            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()

            result['gt_scales'] = data['gt_scales'][0].numpy()
            result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()
            
            result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            result['pred_scores'] = data['pred_scores'][0].numpy()

            if 'rgb' in  data.keys():

                inputs = {
                    'rgb': data['rgb'][0].cuda(),
                    'pts': data['pts'][0].cuda(),
                    'category_label': data['category_label'][0].cuda(),
                    'text_texture_feat': data['text_texture_feat'][0].cuda(),
                    'text_shape_feat': data['text_shape_feat'][0].cuda(),
                }
                start_time1 = time.time()
                end_points = model(inputs)
                end_time1 = time.time()
                total_time += (end_time1 - start_time1)

                pred_translation = end_points['pred_translation']
                pred_size = end_points['pred_size']
                pred_scale = torch.norm(pred_size, dim=1, keepdim=True)
                pred_size = pred_size / pred_scale
                pred_rotation = end_points['pred_rotation']

                num_instance = pred_rotation.size(0)
                pred_RTs =torch.eye(4).unsqueeze(0).repeat(num_instance, 1, 1).float().to(pred_rotation.device)
                pred_RTs[:, :3, 3] = pred_translation
                pred_RTs[:, :3, :3] = pred_rotation * pred_scale.unsqueeze(2)
                pred_scales = pred_size

                result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
                result['pred_scales'] = pred_scales.detach().cpu().numpy()
                
                with open(os.path.join(save_path, path.split('/')[-1]), 'wb') as f:
                    cPickle.dump(result, f)
                draw_box_to_image(data, result, dataloder.dataset.data_name, i, save_path)

            else:
                import numpy as np
                ninstance = data['pred_class_ids'][0].numpy().shape[0]
                result['pred_RTs'] = np.zeros((ninstance, 4, 4))
                result['pred_RTs'][:, :3, :3] = np.diag(np.ones(3))
                result['pred_scales'] = np.ones((ninstance, 3))

            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, len(dataloder), num_instance)
            )

            t.update(1)
    print('Average inference time per sample: {:.4f} seconds'.format(total_time / len(dataloder.dataset)))        


def draw_box_to_image(data, result,  data_name, img_id, save_path=None):
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug']  # 6
    if data_name == 'real':
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else:
        intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])

    out_dir = save_path + "/vis_box"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = data['ori_img'][0].numpy()
    gt_class_ids = result['gt_class_ids']
    gt_bboxes = result['gt_bboxes']
    gt_RTs = result['gt_RTs']
    gt_scales = result['gt_scales']
    gt_handle_visibility = result['gt_handle_visibility']
    pred_class_ids = result['pred_class_ids']
    pred_bboxes = result['pred_bboxes']
    pred_scores = result['pred_scores']

    pred_RTs = result['pred_RTs']
    pred_scales = result['pred_scales']

    ########### vis box ###############
    iou_cls_gt_match, iou_pred_indices = compute_3d_matches_for_each_gt(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                                                                        pred_bboxes, pred_class_ids, pred_scores, pred_RTs, pred_scales)

    pred_class_ids = pred_class_ids[iou_pred_indices]
    pred_RTs = pred_RTs[iou_pred_indices]
    pred_scores = pred_scores[iou_pred_indices]
    pred_bboxes = pred_bboxes[iou_pred_indices]

    pred_RTs = pred_RTs[iou_cls_gt_match]
    pred_scales = pred_scales[iou_cls_gt_match]
    pred_class_ids = pred_class_ids[iou_cls_gt_match]

    draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_RTs, pred_scales, pred_class_ids,
                    gt_RTs, gt_scales, gt_class_ids, None, None, None, draw_gt=True, draw_nocs=False)



class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger
