'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2024-06-05 11:20:26
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2024-06-05 14:50:10
FilePath: /SA2M/demo/semantic_area_match_demo.py
Description: demo of using semantic area matcher
'''

import sys
sys.path.append('../')

import cv2
import numpy as np
import os
from loguru import logger

from models.SemAreaMatcher import RawSemAreaMatcher 

from configs.CommonConfigWriter import args4SGAM, RawSemAMConfWriter
from utils.load import (
    load_config_yaml,
    assemble_config_with_scene_pair,
)

# write config file
args = args4SGAM()

args.out_SAM_conf_name = "demo_sam_config"
args.out_SAM_conf_path = "./demo_sam_res"

# only area matching cfg need to be written
# dataset path
# get the path of the file
args.datasetName = "ScanNet"
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
args.root_path = os.path.join(current_file_path, "demo_data", "ScanData")
args.out_path = os.path.join(current_file_path, "demo_sam_res")
args.out_SAM_conf_path = os.path.join(current_file_path, "demo_sam_cfg")
args.out_SAM_conf_name = "demo_sam_config"
args.out_path = os.path.join(current_file_path, "demo_sam_res")

args.crop_from_size_W = 1296
args.crop_from_size_H = 968
args.crop_size_W = 480
args.crop_size_H = 480

args.SAMerName = "RSAM"
args.semantic_mode = "ScanNetGT"

writeSAM = RawSemAMConfWriter(args, dataset="ScanNet")
writeSAM.write_config_file(args)

SAMer_cfg = load_config_yaml(os.path.join(args.out_SAM_conf_path, args.out_SAM_conf_name + ".yaml"))
SAMer_cfg = assemble_config_with_scene_pair(SAMer_cfg, scene="scene0002_00", pair0="0", pair1="5", out_path="")
SAMer = RawSemAreaMatcher(SAMer_cfg, datasetName="ScanNet", draw_verbose=1)


matched_area0s, matched_area1s, doubt_match_pairs, total_crops = SAMer.FindMatchArea(1)

