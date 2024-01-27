'''
Author: your name
Date: 2022-10-18 17:21:28
LastEditTime: 2024-01-17 19:06:11
LastEditors: EasonZhang
Description: Matcher used for GAMatcher
FilePath: /SGAM/Matchers/BasicGeoLoadMatcher.py
'''

""" Fixed 
Funcs
    - init by configs={}
        - load imgs/depths/... for eval
    - self.update_img
    - self.set_corr_num_init
    - self.match(img0, img1)
    - self.return_matches (ori-corrs, no need to recover)
Params
    Match
        - self.matched_corrs [[corr]s]
    Save Params:
        - self.ratios_folder
        - self.ratios_file
        - self.ratios_name_file
"""

import sys 
sys.path.append("..")

import os
from loguru import logger

from utils.load import *
from utils.geo import assert_match_reproj
from utils.vis import plot_matches_with_mask_ud


class BasicMatcher(object):
    """
    """

    def __init__(self, configs={}, dataset="ScanNet", mode="test"):
        """
        Args:
            configs are updated with pairs and scene
        """
        if mode == "test":
            if dataset == "ScanNet":
                self.ScanNet_init(configs)
            else:
                raise NotImplementedError

            self.scene_pairs_name = configs["scene_name"]+"_"+configs["pair0"]+"_"+configs["pair1"]

            self.match_num = 500

            self.size = [configs["W"], configs["H"]]
            self.W = configs["W"]
            self.H = configs["H"]

            self.color0, self.scale0 = load_cv_img_resize(self.color_path0, self.std_W, self.std_H, 1)
            self.color1, self.scale1 = load_cv_img_resize(self.color_path1, self.std_W, self.std_H, 1)

            self.depth0 = load_cv_depth(self.depth_path0)
            self.depth1 = load_cv_depth(self.depth_path1)
            self.depth_factor = configs["depth_factor"]

            assert self.std_H, self.std_W == self.depth0.shape
            logger.info(f"standard shape = {self.depth0.shape}")
            
            self.K0 = load_K_txt(self.K_path, self.scale0)
            self.K1 = load_K_txt(self.K_path, self.scale1)
            self.pose0 = load_pose_txt(self.pose_path0)
            self.pose1 = load_pose_txt(self.pose_path1)
            
            self.out_path = configs["out_path"]
            test_dir_if_not_create(self.out_path)

            self.eval_MMA_thds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

            self.matched_corrs = []
            
            # specific
            self.matcherName = ""
            self.ratios_folder = configs["ratios_folder"]
            self.ratios_file = configs["ratios_file"]
            self.ratios_name_file = configs["ratios_name_file"]
        elif mode == "tool":
            # self.W = configs["W"]
            # self.H = configs["H"]
            self.dataset = dataset
            pass
        else:
            raise NotImplementedError

    def ScanNet_init(self, configs={}):
        """
        """
        self.sem_path = "label-filt"
        self.sem_post = ".png"
        self.color_path = "color"
        self.color_post = ".jpg"
        self.depth_path = "depth"
        self.depth_post = ".png"
        self.color_K_path = "intrinsic/intrinsic_color.txt"
        self.pose_path = "pose"
        self.pose_post = ".txt"
        self.std_W = 640 # same as depth
        self.std_H = 480

        scene = configs["scene_name"]
        pair0 = configs["pair0"]
        pair1 = configs["pair1"]

        self.color_path0 = os.path.join(configs["root_path"], scene, self.color_path, pair0+self.color_post)
        self.color_path1 = os.path.join(configs["root_path"], scene, self.color_path, pair1+self.color_post)

        self.depth_path0 = os.path.join(configs["root_path"], scene, self.depth_path, pair0+self.depth_post)
        self.depth_path1 = os.path.join(configs["root_path"], scene, self.depth_path, pair1+self.depth_post)

        self.K_path = os.path.join(configs["root_path"], scene, self.color_K_path)

        self.pose_path0 = os.path.join(configs["root_path"], scene, self.pose_path, pair0+self.pose_post)
        self.pose_path1 = os.path.join(configs["root_path"], scene, self.pose_path, pair1+self.pose_post)

        self.out_path = os.path.join(configs["out_path"], scene+"_"+pair0+"_"+pair1)

    def recover_ori_img_corrs(self):
        """
        """
        rt_matches = []
        W_ratio = self.std_W / self.W
        H_ratio = self.std_H / self.H

        for corrs in self.matched_corrs:
            u0, v0, u1, v1 = corrs
            u0 = W_ratio * u0
            v0 = H_ratio * v0
            u1 = W_ratio * u1
            v1 = H_ratio * v1

            rt_matches.append([u0, v0, u1, v1])
        
        self.matched_corrs = rt_matches

    def update_img(self, img0, img1):
        """
        """
        self.color0 = img0
        self.color1 = img1
    
    def set_corr_num_init(self, num):
        """
        """
        self.match_num = num
    
    def match_ori_img(self):
        """ Specific for Related work
        """
        raise NotImplementedError

    def match(self, img0, img1):
        """
        """
        raise NotImplementedError
    
    def return_matches(self):
        """
        Returns:
            matched_corrs: list [[corr]s]
        """
        return self.matched_corrs

    def convert_matches2list(self, mkpts0, mkpts1):
        """
        Args:
            mkpts0/1: np.ndarray Nx2
        """
        matches = []

        assert mkpts0.shape == mkpts1.shape, f"different shape: {mkpts0.shape} != {mkpts1.shape}"

        for i in range(mkpts0.shape[0]):
            u0, v0 = mkpts0[i,0], mkpts0[i,1]
            u1, v1 = mkpts1[i,0], mkpts1[i,1]

            matches.append([u0, v0, u1, v1])
        
        return matches