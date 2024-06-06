'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2024-06-05 11:20:26
LastEditors: EasonZhang
LastEditTime: 2024-06-06 23:16:08
FilePath: /SGAM/demo/sgam_demo.py
Description: demo of using semantic and geometry area matcher
'''

import sys
sys.path.append('../')

import cv2
import numpy as np
import os
from loguru import logger
import random
random.seed(1)

from models.SemAreaMatcher import RawSemAreaMatcher 
from models.GeoAreaMatcher import PRGeoAreaMatcher

from configs.CommonConfigWriter import args4SGAM, RawSemAMConfWriter, PRGeoAMConfWriter
from utils.load import (
    load_config_yaml,
    assemble_config_with_scene_pair,
    load_cv_depth,
    load_K_txt,
    load_pose_txt,
)

from utils.geo import compute_pose_error_simp


scene = "scene0002_00"
pair0 = "0"
pair1 = "5"

def _list_of_corrs2corr_list(list_of_corrs, fuse=0):
    """
        [[corrs],...] -> [[corr], ...]
    Args:
        fuse: if fuse the corrs, if True, fuse the close corrs into one corr
            if two corrs are close enough (both of the coordinate pairs distance < 1px), fuse them into one corr
        fuse_mode: rough, average
    Returns:
    """
    rt_corrs = []
    total_num = 0
    if fuse == 0:
        for corrs in list_of_corrs:
            rt_corrs += corrs
            total_num += len(corrs)
    else:
        raise NotImplementedError
    logger.info(f"fuse {total_num} corrs into {len(rt_corrs)} corrs")
    return rt_corrs

def _update_matched_areas(alpha_list, phi_inlier_idxs_dict, matched_area0s, matched_area1s):
    """ use self.phi_inlier_idx_dict to update self.matched_area0/1s into self.matched_area0/1s_dict
    """
    alpha_matched_area0s_dict = {}
    alpha_matched_area1s_dict = {}

    for alpha in alpha_list:
        temp_matched_area0s = []
        temp_matched_area1s = []
        if len(phi_inlier_idxs_dict[alpha]) == 0:
            logger.info("No inliers, clear matched pairs")
        else:
            for idx in phi_inlier_idxs_dict[alpha]:
                temp_matched_area0s.append(matched_area0s[idx])
                temp_matched_area1s.append(matched_area1s[idx])
    
        alpha_matched_area0s_dict[alpha] = temp_matched_area0s
        alpha_matched_area1s_dict[alpha] = temp_matched_area1s

    return alpha_matched_area0s_dict, alpha_matched_area1s_dict

def pose_eval(corrs, pose0, pose1, K0, K1):
    """ Evaluation the matching performance through relative pose
    Returns:
        errs: [R_err, t_err]
    """
    corrs_num = 1000

    if len(corrs) > corrs_num:
        corrs = random.sample(corrs, corrs_num)


    gt_pose = pose1.I @ pose0

    logger.success(f'calc pose with {len(corrs)} corrs')

    errs = compute_pose_error_simp(corrs, K0, K1, gt_pose)
    
    return errs


# write config file
args = args4SGAM()

args.out_SAM_conf_name = "demo_sam_config"
args.out_SAM_conf_path = "./demo_sam_res"

# set up args
args.datasetName = "ScanNet"
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
args.root_path = os.path.join(current_file_path, "demo_data", "ScanData")
args.out_path = os.path.join(current_file_path, "demo_sam_res")

# SAMer part, most parameters are set default in args4SGAM
args.out_SAM_conf_path = os.path.join(current_file_path, "demo_sam_cfg")
args.out_SAM_conf_name = "demo_sam_config"

# GAMer part
args.out_GAM_conf_path = os.path.join(current_file_path, "demo_gam_cfg")
args.out_GAM_conf_name = "demo_gam_config"
args.out_path = os.path.join(current_file_path, "demo_sgam_res")
args.alpha_list = [5.0]
args.adaptive_size_thd=0.6

args.crop_from_size_W = 1296
args.crop_from_size_H = 968
args.crop_size_W = 640
args.crop_size_H = 480
args.eval_from_size_W = 640
args.eval_from_size_H = 480

args.SAMerName = "RSAM"
args.semantic_mode = "ScanNetGT"

# write down config file for SAMer FIXME: need to be refactored
writeSAM = RawSemAMConfWriter(args, dataset="ScanNet")
writeSAM.write_config_file(args)

args.GAMerName = "PRGAM"

# write down config file for GAMer FIXME: need to be refactored
writeGAM = PRGeoAMConfWriter(args, dataset="ScanNet")
writeGAM.write_config_file(args)

# read config file to initialize SAMer
SAMer_cfg = load_config_yaml(os.path.join(args.out_SAM_conf_path, args.out_SAM_conf_name + ".yaml"))
SAMer_cfg = assemble_config_with_scene_pair(SAMer_cfg, scene=scene, pair0=pair0, pair1=pair1, out_path="")
SAMer = RawSemAreaMatcher(SAMer_cfg, datasetName="ScanNet", draw_verbose=1)

# area matching using SAMer
matched_area0s, matched_area1s, doubt_match_pairs, total_crops = SAMer.FindMatchArea(1)

# read config file to initialize GAMer
GAMer_cfg = load_config_yaml(os.path.join(args.out_GAM_conf_path, args.out_GAM_conf_name + ".yaml"))
GAMer_cfg = assemble_config_with_scene_pair(GAMer_cfg, scene=scene, pair0=pair0, pair1=pair1, out_path="")
GAMer = PRGeoAreaMatcher(GAMer_cfg, datasetName="ScanNet", draw_verbose=1)

# geometry matching using GAMer

## load point matcher
from Matchers.ASpanMatcher import ASpanMatcher
# NOTE: the weight path in the config file need to be modified
PMconf = load_config_yaml(os.path.join(current_file_path, "..", "configs", "ASpanConfig.yaml")) 
PMer = ASpanMatcher(PMconf, "ScanNet", "tool")

# PMer matches ori images
PMer.set_corr_num_init(1000)

# load ori imgs
ori_img0, ori_img1 = SAMer.color0, SAMer.color1

PMer.match(ori_img0, ori_img1)
ori_corrs = PMer.corrs

GAMer.load_matcher(PMer)
GAMer.ori_img_corrs = ori_corrs

# geometry area prediction
if len(doubt_match_pairs) > 0:
    predicted_match_area0s, predicted_match_area1s = GAMer.predict_area_match_main_flow(doubt_match_pairs)

# geometry area rejection
matched_area0s += predicted_match_area0s
matched_area1s += predicted_match_area1s

[phi_inlier_corrs_dict, inlier_F, phi_inlier_idxs_dict] = GAMer.rejection_by_samp_dist_flow(matched_area0s, matched_area1s)

if phi_inlier_corrs_dict is not None:
    for alpha_temp in args.alpha_list:
        phi_inlier_corrs_dict[alpha_temp] = _list_of_corrs2corr_list(phi_inlier_corrs_dict[alpha_temp], fuse=0)


matched_area0s_dict, matched_area1s_dict = _update_matched_areas(args.alpha_list, phi_inlier_idxs_dict, matched_area0s, matched_area1s)


# load geo info 
data_path = args.root_path
K_txt = os.path.join(data_path, scene, "intrinsic", "intrinsic_color.txt")
K0 = load_K_txt(K_txt, [640/1296, 480/968])
pose_txt0 = os.path.join(data_path, scene, "pose", f"{pair0}.txt")
pose_txt1 = os.path.join(data_path, scene, "pose", f"{pair1}.txt")
pose0 = load_pose_txt(pose_txt0)
pose1 = load_pose_txt(pose_txt1)

# evaluate the pose
errs_ori = pose_eval(ori_corrs, pose0, pose1, K0, K0)

logger.success(f"ori pose error: {errs_ori}")

# evaluate the pose for SGAM
for alpha in args.alpha_list:
    errs = pose_eval(phi_inlier_corrs_dict[alpha], pose0, pose1, K0, K0)

    logger.success(f"SGAM-{alpha} pose error: {errs}")