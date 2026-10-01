import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from solver import test_func, get_logger
from dataset import TestDataset
from evaluation_utils import evaluate

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="1",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/GenCOPE.yaml",
                        help="path to config file")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=1000,
                        help="test epoch")
    parser.add_argument('--mask_label', action='store_true', default=False,
                        help='whether having mask labels of real data')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='whether directly evaluating the results')
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    log_dir = "log/backbone/GenCOPE"

    cfg = gorilla.Config.fromfile(args.config)

    cfg.log_dir = log_dir
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.mask_label = args.mask_label
    cfg.only_eval = args.only_eval

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch" + str(cfg.test_epoch)  + "_logger.log")

    return logger, cfg

def count_parameters_per_module(model):
    print(f"{'Module Name':<25} | {'Params (M)':<10} | {'Ratio (%)':<10}")
    print("-" * 50)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        
        ratio = (module_params / total_params) * 100
        print(f"{name:<25} | {module_params/1e6:<10.2f} | {ratio:<10.1f}")
        
    print("-" * 50)
    print(f"Total Model Params: {total_params/1e6:.2f} M")

if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = os.path.join(cfg.log_dir, 'eval_epoch' + str(cfg.test_epoch))
    if not cfg.only_eval:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # model
        logger.info("=> creating model ...")
        if cfg.model_arch == "GenCOPE":
            from model.GenCOPE import GenCOPE
            print("Loading GenCOPE model")
            model = GenCOPE()

        if len(cfg.gpus)>1:
            model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
        model = model.cuda()
        model.eval()

        checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
        logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

        # data loader
        TestingDataset = TestDataset
        data_dir = r'/mnt/HDD4/dzq/nocs_istnet'
        dataset = TestingDataset(cfg.test, data_dir, text_features_file='text_features.npz')

        dataloder = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=8,
                shuffle=False,
                drop_last=False
            )

        test_func(model, dataloder, save_path)
        

    evaluate(save_path, logger)

