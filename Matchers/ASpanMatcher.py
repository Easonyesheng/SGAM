
# NOTE use the conda activate sgam_aspan
import sys
sys.path.append("../")

import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from loguru import logger
import random
import torch.nn.functional as F

from Matchers.ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer 
from Matchers.ASpanFormer.src.config.default import get_cfg_defaults
from Matchers.ASpanFormer.src.utils.misc import lower_config
from Matchers.ASpanFormer.demo import demo_utils
# get current file path
cur_path = os.path.dirname(os.path.abspath(__file__))
ASpan_config_path = f"{cur_path}/ASpanFormer/configs/aspan/indoor/aspan_test.py"

from Matchers.BasicGeoLoadMatcher import BasicMatcher

class ASpanMatcher(BasicMatcher):
    """
    Specific:
        configs["cuda_idx"]
        configs["weights"]
    """
    def __init__(self, configs={}, dataset="ScanNet", mode="test", resize=False):
        super().__init__(configs, dataset, mode)

        """specific"""
        self.matcherName = "ASpan"
        self.resize = resize
        torch.cuda.set_device(0)

        _default_cfg = get_cfg_defaults()
        if dataset == "ScanNet" or dataset == "Matterport3D" or dataset == "KITTI":
            main_cfg_path = f"{cur_path}/ASpanFormer/configs/aspan/indoor/aspan_test.py"
            data_config = f"{cur_path}/ASpanFormer/configs/data/scannet_test_1500.py"
        elif dataset == "MegaDepth" or dataset == "YFCC":
            main_cfg_path = f"{cur_path}/ASpanFormer/configs/aspan/outdoor/aspan_test.py"
            data_config = f"{cur_path}/ASpanFormer/configs/data/megadepth_test_1500.py"
        else:
            raise NotImplementedError(f"dataset {dataset} not implemented")

        _default_cfg.merge_from_file(main_cfg_path)
        _default_cfg.merge_from_file(data_config)

        _default_cfg = lower_config(_default_cfg)
        matcher = ASpanFormer(config=_default_cfg['aspan'])
        # weights = f"{cur_path}/{configs['weights']}"
        weights = configs['weights']
        matcher.load_state_dict(torch.load(weights)["state_dict"], strict=False)

        self.matcher = matcher.eval().cuda()


    def match_ori_img(self):
        """ For Related work
        """
        self.color0_input = cv2.imread(self.color_path0, cv2.IMREAD_GRAYSCALE)
        self.color1_input = cv2.imread(self.color_path1, cv2.IMREAD_GRAYSCALE)
        
        self.color0_input = cv2.resize(self.color0_input, (self.W, self.H))
        self.color1_input = cv2.resize(self.color1_input, (self.W, self.H))
        logger.info(f"match ori images with size: {self.W} x {self.H}")

        img_tensor0 = torch.from_numpy(self.color0_input/255.)[None][None].cuda().float()
        img_tensor1 = torch.from_numpy(self.color1_input/255.)[None][None].cuda().float()

        batch = {"image0": img_tensor0, "image1": img_tensor1}

        with torch.no_grad():
            self.matcher(batch, online_resize=True)
            mkpts0 = batch['mkpts0_f'].cpu().numpy() # Nx2
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
        
        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)
        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        self.recover_ori_img_corrs()
        
    def match(self, img0, img1, mask0=None, mask1=None):
        """for SGAMer"""
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        if self.resize:
            img0 = cv2.resize(img0, (640, 480))
            img1 = cv2.resize(img1, (640, 480))

        logger.info(f"img shape is {img0.shape}")

        img_tensor0 = torch.from_numpy(img0 / 255.)[None][None].cuda().float()
        img_tensor1 = torch.from_numpy(img1 / 255.)[None][None].cuda().float()

        batch = {"image0": img_tensor0, "image1": img_tensor1}

        if mask0 is not None and mask1 is not None:
            mask0 = torch.from_numpy(mask0).cuda()
            mask1 = torch.from_numpy(mask1).cuda()
            [ts_mask_0, ts_mask_1] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=0.125,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool().to("cuda")
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})


        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy() # Nx2
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            m_bids = batch['m_bids'].cpu().numpy()

        
        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        
        self.corrs = self.matched_corrs # used in SGAM

        return self.matched_corrs
