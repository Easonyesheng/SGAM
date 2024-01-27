'''
Author: your name
Date: 2022-10-22 12:14:21
LastEditTime: 2024-01-17 20:53:44
LastEditors: EasonZhang
Description: Write configs for SAMer, GAMer, PMer and SGAMer, not scene specific
FilePath: /SGAM/configs/CommonConfigWriter.py
'''

import sys
sys.path.append("..")

import os
import yaml
import argparse
from loguru import logger

from utils.comm import test_dir_if_not_create

def args4SGAM():
    """# Complete Area to Point Matcher V0
        - SemAMer: RawSemAMer
        - GeoAMer: PRGeoAMer
        - PtsMatcher: COTR/LoFTR
    """
    parser = argparse.ArgumentParser()
    
    # common
    parser.add_argument("--root-path", default="", type=str)
    parser.add_argument("--datasetName", type=str, default="ScanNet")
    # size
    parser.add_argument("--area-from-size-W", type=int, default=640, help="area matches are from which size") 
    parser.add_argument("--area-from-size-H", type=int, default=480, help="area matches are from which size")
    parser.add_argument("--eval-from-size-W", type=int, default=640, help="evaluation corrs from which size, same as the K & depth size")
    parser.add_argument("--eval-from-size-H", type=int, default=480, help="evaluation corrs from which size, same as the K & depth size")
    parser.add_argument("--crop-from-size-W", type=int, default=1296, help="crop for PMer from which size")
    parser.add_argument("--crop-from-size-H", type=int, default=968, help="crop for PMer from which size")
    parser.add_argument("--crop-size-W", type=int, default=256, help="crop area with which size fro PMer")
    parser.add_argument("--crop-size-H", type=int, default=256, help="crop area with which size fro PMer")
    parser.add_argument("--match-num", type=int, default=500, help="standard corrs number in SGAM")
    parser.add_argument("--fuse", type=int, default=0, help="fuse flag")
    parser.add_argument("--fuse-thd", type=float, default=0.5, help="fuse threshold")
    parser.add_argument("--fuse-mode", type=str, default="rough", help="fuse mode")
    parser.add_argument("--ablation", type=int, default=0, help="ablation flag")
    # out path
    parser.add_argument("--out-path", default="", type=str, help="res file out path / scenename")
    parser.add_argument("--out-ratio-filefolder", type=str, default="MMARatios")
    # conf path
    parser.add_argument("--out-SAM-conf-path", default="", type=str, help="output config file path")
    parser.add_argument("--out-SAM-conf-name", default="ScanNet_SAM_conf", type=str)
    parser.add_argument("--out-GAM-conf-path", default="", help="GeoAM config path")
    parser.add_argument("--out-GAM-conf-name", default="ScanNet_GAM_conf", type=str)
    parser.add_argument("--out-SGAM-conf-path", type=str, default="")
    parser.add_argument("--out-SGAM-conf-name", type=str, default="ScanNet_SGAM_conf")
    parser.add_argument("--PM-conf-path", type=str, default="", help="point matcher config path")
    parser.add_argument("--resize", type=int, default=0, help="resize flag")

    # for KITTI
    parser.add_argument("--sequence_id", type=str, default="0", help="KITTI sequence id")

    # for RawSemAMer
    parser.add_argument("--connected-radius", default=10, type=int, help="connected area stastic radius")
    parser.add_argument("--connected-thd", default=3600, type=int, help="connected area size threshold")
    parser.add_argument("--radius-thd-down", default=100, type=int, help="achieve patch radius down limits")
    parser.add_argument("--radius-thd-up", default=128, type=int, help="achieve patch radius up limits")
    parser.add_argument("--desc-type", default=2, type=int, help="descriptor type 1 is the binary descriptor")
    parser.add_argument("--small-label-filted-thd-on-bound", default=20, type=int, help="small label filted on bound with this size")
    parser.add_argument("--small-label-filted-thd-in-area", default=900, type=int, help="small label filted in area with this size")

    parser.add_argument("--combined-obj-dist-thd", default=200, type=int, help="overlap combination distance threshold")
    parser.add_argument("--obj-desc-match-thd", default=0.5, type=float, help="object descriptor match threshold")
    parser.add_argument("--leave-multi-obj-match", default=0, type=int, help="1 is to leave multi-match to geoAM")

    parser.add_argument("--same-overlap-dist", default=100, type=int, help="points distance thd belong to the same overlap when stastic overlap area")
    parser.add_argument("--inv-overlap-pyramid-ratio", default=8, type=int, help="the inverse value of overlap stastic pyramid ratio")
    parser.add_argument("--label-list-area-thd", default=400, type=int, help="stastic area on label list with this size threshold")
    parser.add_argument("--overlap-radius", default=128, type=int, help="the semantic overlap stastic radius < 240")
    parser.add_argument("--overlap-desc-ll-match-thd", default=0.7, type=float, help="overlap label list soft match threshold")
    parser.add_argument("--overlap-desc-dist-thd", default=0.25, type=float, help="overlap area desc eu-dist thd")
    parser.add_argument("--output-patch-size", default=256, type=int, help="the output patch size, always square")
    parser.add_argument("--semantic-mode", default='ScanNetGT', type=str, help="How to get the semantic segmentation")
    parser.add_argument("--sem-path-SEEM", default="/data2/zys/SEEMRes", type=str, help="SEEM semantic segmentation path")
    parser.add_argument("--sem-post-SEEM", default=".png", type=str, help="SEEM semantic segmentation post")

    # for PRGeoAM
    parser.add_argument("--standard-area-size", default=256, help="default area size used in GeoAM") # old version
    parser.add_argument("--area-match-num", type=int, default=500, help="standard corrs number in GeoAM")
    parser.add_argument("--rejection-mode", default=1, type=int, help="choice: 0. non-rejection; 1. geo_consist; 2. RANSAC;")
    parser.add_argument("--inlier-thd-mode", type=int, default=0, help="0. average;")
    parser.add_argument('--alpha_list', type=float, nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--filter_area_num", type=int, default=1, help="area num thd for filter")
    parser.add_argument("--adaptive-size-thd", type=float, default=0.2, help="adaptive size thd for adaptive A2PM point collection")
    parser.add_argument("--draw-adap", type=int, default=0, help="draw adaptive A2PM point collection")

    # for A2PMer
    parser.add_argument("--cuda-idx", type=str, default="5")
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--SAMerName", type=str, default="RSAM", help="The name of SAMer")
    parser.add_argument("--GAMerName", type=str, default="PRGAM", help="the name of GeoAMer")
    parser.add_argument("--PMerName", type=str, default="COTR", help="The name of Point Matcher")
    parser.add_argument("--draw-verbose-flag", type=int, default=0, help="draw flag 1 is true")
    parser.add_argument("--testAM", type=int, default=0, help="test area matcher flag")

    args = parser.parse_args()

    return args

class YFCCCommonPartConfWriter(object):
    """ Common Part Config Writer for YFCC"""
    def __init__(self, args) -> None:
        """ at config write part, we don't know the scene name and pair name
            so only write the common part
        """
        self.data = {
            "root_path": args.root_path,
            "sem_mode": "SEEMRes",
            "sem_model_name": "SAMViTB",
            "sem_post": ".png",
            "pair_folder": "raw_data/pairs",
            "pair_post": ".txt",
            "geo_folder": "data_dump_rPose",
            "geo_mid_path": "sift-2000/test",
            "out_path": args.out_path,
        }

    def get_basic_conf_dict(self): 
        """
        """
        return self.data

class ScanNetCommonPartConfWriter(object):
    """ Common Part Config Writer for ScanNet
    """

    def __init__(self, args) -> None:
        self.data = {
            "root_path": args.root_path,
            "sem_path": "label-filt",
            "sem_post": ".png",
            "color_path": "color",
            "color_post": ".jpg",
            "depth_path": "depth",
            "depth_post": ".png",
            "color_K_path": "intrinsic/intrinsic_color.txt",
            "pose_path": "pose",
            "pose_post": ".txt",
            "depth_factor": 1000,
            "out_path": args.out_path
        }
    
    def get_basic_conf_dict(self):
        """
        """
        return self.data

class MP3DCommonPartConfWriter(object):
    """
    NOTE:
        1. pose is camera to world
        2. depth factor is 4000 (divide 4000 to get meter per value)
        3. args input scene and pair folder
    """

    def __init__(self, args) -> None:
        self.data = {
            "root_path": args.root_path,
            "sem_path": "sems",
            "sem_post": ".png",
            "color_path": "imgs",
            "color_post": ".jpg",
            "depth_path": "depths",
            "depth_post": ".png",
            "K_path": "cam_params",
            "pose_path": "cam_params",
            "pose_post": ".txt",
            "depth_factor": 4000,
            "out_path": args.out_path
        }

    def get_basic_conf_dict(self):
        """
        """
        return self.data

class MegaDepthCommonPartConfWriter(object):
    """
    """
    def __init__(self, args) -> None:
        """
        """
        self.data = {
            "root_path": args.root_path,
            "out_path": args.out_path,
            "sam_path": args.sam_path,
            "sam_post": args.sam_post,
        }
    
    def get_basic_conf_dict(self):
        """
        """
        return self.data

class KITTICommonPartConfWriter(object):
    """
    NOTE:
        1. pose is camera to world
        2. no depth is offered, only pose estimation, without MMA
        3. sequence id is provided in args
    Args:
        args: args from argparse
            - args.root_path: root path of KITTI dataset
            - args.sequence_id: sequence id of KITTI dataset, no need to fill 0
            - args.out_path: output path of KITTI dataset
    """
    def __init__(self, args) -> None:
        """
        """
        sequence_id = str(args.sequence_id).zfill(4)
        self.data = {
            "root_path": args.root_path,
            "sequence_id": sequence_id,
            "sem_path": f"data_2d_semantics/train/2013_05_28_drive_{sequence_id}_sync/image_00/semantic",
            "sem_post": ".png",
            "color_path": f"2013_05_28_drive_{sequence_id}_sync/image_00/data_rect",
            "color_post": ".png",
            "pose_path": f"data_poses/2013_05_28_drive_{sequence_id}_sync/cam0_to_world.txt",
            "out_path": args.out_path,
        }
    
    def get_basic_conf_dict(self):
        """
        """
        return self.data

class RawSemAMConfWriter(object):
    """
    Returns:
        config file is saved as config_root_path/{scene_name}_{pair0}_{pair1}.yaml
    """
    def __init__(self, args, dataset="ScanNet") -> None:
        """
        """
        self.datasetName = dataset

        if dataset == "ScanNet":
            self.dataset_conf = ScanNetCommonPartConfWriter(args)
        elif dataset == "MatterPort3D":
            self.dataset_conf = MP3DCommonPartConfWriter(args)
        elif dataset == "KITTI":
            self.dataset_conf = KITTICommonPartConfWriter(args)
        elif dataset == "YFCC":
            self.dataset_conf = YFCCCommonPartConfWriter(args)
        else:
            raise NotImplementedError
        
        dataset_conf = self.dataset_conf.get_basic_conf_dict()

        specific_conf = {
            "W": args.area_from_size_W,
            "H": args.area_from_size_H,
            "connected_thd": args.connected_thd,
            "radius_thd_down": args.radius_thd_down,
            "radius_thd_up": args.radius_thd_up,
            "desc_type": args.desc_type, 
            "small_label_filted_thd_on_bound": args.small_label_filted_thd_on_bound,
            "small_label_filted_thd_inside_area": args.small_label_filted_thd_in_area,
            "combined_obj_dist_thd": args.combined_obj_dist_thd,
            "obj_desc_match_thd": args.obj_desc_match_thd,
            "same_overlap_dist": args.same_overlap_dist,
            "overlap_desc_dist_thd": args.overlap_desc_dist_thd,
            "label_list_area_thd": args.label_list_area_thd,
            "overlap_radius": args.overlap_radius,
            "output_patch_size": args.output_patch_size,
            "leave_multi_obj_match": args.leave_multi_obj_match,
            "inv_overlap_pyramid_ratio": args.inv_overlap_pyramid_ratio,
            "semantic_mode": args.semantic_mode,
            "sem_path_SEEM": args.sem_path_SEEM,
            "sem_post_SEEM": args.sem_post_SEEM,
        }

        self.data = {**dataset_conf, **specific_conf}
        

    
    def write_config_file(self, args):
        """
        """
        logger.info(f"config file for RawSemAMer is written and saved in {args.out_SAM_conf_path}")

        test_dir_if_not_create(args.out_SAM_conf_path)        

        yaml_path = os.path.join(args.out_SAM_conf_path, args.out_SAM_conf_name+".yaml")

        with open(yaml_path, "w") as f:
            yaml.dump(self.data, f)
        
        return 
    
class PRGeoAMConfWriter(object):
    """
    """
    def __init__(self, args, dataset="ScanNet") -> None:
        """
        """
        self.datasetName = dataset

        if dataset == "ScanNet":
            self.dataset_conf = ScanNetCommonPartConfWriter(args)
        elif dataset == "MatterPort3D":
            self.dataset_conf = MP3DCommonPartConfWriter(args)
        elif dataset == "KITTI":
            self.dataset_conf = KITTICommonPartConfWriter(args)
        elif dataset == "MegaDepth":
            self.dataset_conf = MegaDepthCommonPartConfWriter(args)
        elif dataset == "YFCC":
            self.dataset_conf = YFCCCommonPartConfWriter(args)
        else:
            raise NotImplementedError
        
        dataset_cof = self.dataset_conf.get_basic_conf_dict()

        specific_conf = {
            "crop_from_size_W": args.crop_from_size_W,
            "crop_from_size_H": args.crop_from_size_H,
            "area_from_size_W": args.area_from_size_W,
            "area_from_size_H": args.area_from_size_H,
            "eval_from_size_W": args.eval_from_size_W,
            "eval_from_size_H": args.eval_from_size_H,
            "crop_size_W": args.crop_size_W,
            "crop_size_H": args.crop_size_H,
            "standard_area_size": args.standard_area_size,
            "match_num": args.area_match_num,
            "rejection_mode": args.rejection_mode,
            "inlier_thd_mode": args.inlier_thd_mode,
            "alpha_list": args.alpha_list,
            "filter_area_num": args.filter_area_num,
            "adaptive_size_thd": args.adaptive_size_thd,
            "draw_adap": args.draw_adap,
        }

        self.data = {**dataset_cof, **specific_conf}

    def write_config_file(self, args):
        """
        """
        yaml_path = os.path.join(args.out_GAM_conf_path, args.out_GAM_conf_name+".yaml")

        test_dir_if_not_create(args.out_GAM_conf_path)

        with open(yaml_path, "w") as f:
            yaml.dump(self.data, f)

        logger.info(f"config file for PRGeoAMer is written and saved in {args.out_GAM_conf_path}")
        return

class SGAMConfWriter(object):
    """
    """
    def __init__(self, args, dataset="ScanNet") -> None:
        """
        """
        self.datasetName = dataset

        if dataset == "ScanNet":
            self.dataset_conf = ScanNetCommonPartConfWriter(args)
        elif dataset == "MatterPort3D":
            self.dataset_conf = MP3DCommonPartConfWriter(args)
        elif dataset == "KITTI":
            self.dataset_conf = KITTICommonPartConfWriter(args)
        elif dataset == "MegaDepth":
            self.dataset_conf = MegaDepthCommonPartConfWriter(args)
        elif dataset == "YFCC":
            self.dataset_conf = YFCCCommonPartConfWriter(args)
        else:
            raise NotImplementedError
        
        dataset_cof = self.dataset_conf.get_basic_conf_dict()

        self.SAM_conf_path = os.path.join(args.out_SAM_conf_path, args.out_SAM_conf_name+".yaml")
        self.GAM_conf_path = os.path.join(args.out_GAM_conf_path, args.out_GAM_conf_name+".yaml")

        specific_conf = {
            "crop_W": args.crop_size_W,
            "crop_H": args.crop_size_H,
            "eval_W": args.eval_from_size_W,
            "eval_H": args.eval_from_size_H,
            "cuda_idx": args.cuda_idx,
            "random_seed": args.random_seed,
            "SAMerName": args.SAMerName,
            "SAM_conf_path": self.SAM_conf_path,
            "GAMerName": args.GAMerName,
            "GAM_conf_path": self.GAM_conf_path,
            "PMerName": args.PMerName,
            "PM_conf_path": args.PM_conf_path,
            "draw_verbose_flag": args.draw_verbose_flag,
            "out_ratio_filefolder": args.out_ratio_filefolder,
            "alpha_list": args.alpha_list,
            "semantic_mode": args.semantic_mode,
            "match_num": args.match_num,
            "fuse": args.fuse,
            "fuse_thd": args.fuse_thd,
            "fuse_mode": args.fuse_mode,
            "ablation": args.ablation,
            "resize": args.resize,
            "testAM": args.testAM,
        }

        self.data = {**dataset_cof, **specific_conf}

    def write_config_file(self, args):
        """
        """

        yaml_path = os.path.join(args.out_SGAM_conf_path, args.out_SGAM_conf_name + ".yaml")

        test_dir_if_not_create(args.out_SGAM_conf_path)

        with open(yaml_path, "w") as f:
            yaml.dump(self.data, f)

        logger.info(f"config file for PRGeoAMer is written and saved in {args.out_SGAM_conf_path}")
        return

