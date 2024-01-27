'''
Author: Eason
Date: 2022-07-08 15:18:44
LastEditTime: 2024-01-17 20:32:58
LastEditors: EasonZhang
Description: load utils
FilePath: /SGAM/utils/load.py
'''


import numpy as np
import yaml
import cv2
from loguru import logger
import os
import glob
from itertools import combinations

from .geo import assembel_pose, assembel_K

def assemble_config_with_scene_pair(configs, scene, pair0, pair1, out_path):
    """
    """
    update_dict = {
        "scene_name": scene,
        "pair0": pair0,
        "pair1": pair1
    }

    rt_configs = {**configs, **update_dict}

    if out_path != "":
        rt_configs["out_path"] = out_path

    return rt_configs

def assemble_config_with_scene_pair_KITTI(configs, pair0, pair1, out_path):
    """
    """
    update_dict = {
        "pair0": pair0,
        "pair1": pair1
    }

    rt_configs = {**configs, **update_dict}

    if out_path != "":
        rt_configs["out_path"] = out_path

    return rt_configs

def assemble_config_with_pair_MegaDepth(configs, pair0, pair1, out_path):
    """
    """
    update_dict = {
        "pair0": pair0,
        "pair1": pair1
    }

    rt_configs = {**configs, **update_dict}

    if out_path != "":
        rt_configs["out_path"] = out_path

    return rt_configs

def assemble_config_with_scene_pair_MP3D(configs, scene, pairs_name, out_path):
    """ output a list of configs with all pairs in pairs folder for MatterPort3D Dataset
    TODO
    Args:
        update:
            scene: scenen/pair
            pair0: pair0_name
            pair1: pair1_name
    Returns:
        config_list
    """
    config_list = []
    img_folder = os.path.join(configs["root_path"], scene, pairs_name, "imgs")
    logger.info(f"img folder is {img_folder}")

    img_name_list = glob.glob(os.path.join(img_folder, "*.jpg"))
    img_pre_name_list = [x.split("/")[-1].split(".")[0] for x in img_name_list]

    img_num = len(img_name_list)
    candis_idx_list = [idx for idx in range(img_num)]
    candi_combinations = combinations(candis_idx_list, 2)

    for pair in candi_combinations:
        update_dict = {
            "scene_name": os.path.join(scene, pairs_name)
        }
        idx_pair0 = pair[0]
        idx_pair1 = pair[1]
        pair0_img_pre_name = img_pre_name_list[idx_pair0]
        pair1_img_pre_name = img_pre_name_list[idx_pair1]

        update_dict["pair0"] = pair0_img_pre_name
        update_dict["pair1"] = pair1_img_pre_name

        temp_rt_config = {**update_dict, **configs}
        if out_path != "":
            temp_rt_config["out_path"] = out_path
        
        config_list.append(temp_rt_config)
    
    logger.info(f"achieve {len(config_list)} pairs totally")
    return config_list

def update_MP3D_config(configs, semantic_mode="GT"):
    """ assemble all path
    TODO
    """
    if semantic_mode == "GT" or semantic_mode == "ScanNetGT":
        sem_path = configs["sem_path"]
        sem_post = configs["sem_post"]
    elif semantic_mode == "SEEM":
        sem_path = configs["sem_path_SEEM"]
        sem_post = configs["sem_post_SEEM"]
    elif semantic_mode == "SAM":
        sem_path = configs["sam_path"]
        sem_post = configs["sam_post"]
    elif semantic_mode == "SGAM":
        sem_path = ""
        sem_post = ""
    else:
        raise NotImplementedError

    color_path = configs["color_path"]
    color_post = configs["color_post"]
    depth_path = configs["depth_path"]
    depth_post = configs["depth_post"]
    K_path = configs["K_path"]
    pose_path = configs["pose_path"]
    pose_post = configs["pose_post"]

    scene = configs["scene_name"]
    pair0 = configs["pair0"]
    pair1 = configs["pair1"]

    color_path0 = os.path.join(configs["root_path"], scene, color_path, pair0+color_post)
    color_path1 = os.path.join(configs["root_path"], scene, color_path, pair1+color_post)

    depth_path0 = os.path.join(configs["root_path"], scene, depth_path, pair0+depth_post)
    depth_path1 = os.path.join(configs["root_path"], scene, depth_path, pair1+depth_post)
    
    K0_path = os.path.join(configs["root_path"], scene, K_path, "int_"+pair0+pose_post)
    K1_path = os.path.join(configs["root_path"], scene, K_path, "int_"+pair1+pose_post)

    pose_path0 = os.path.join(configs["root_path"], scene, pose_path, "ext_"+pair0+pose_post)
    pose_path1 = os.path.join(configs["root_path"], scene, pose_path, "ext_"+pair1+pose_post)

    out_path = os.path.join(configs["out_path"], scene.replace("/","_")+"_"+pair0+"_"+pair1)
    
    if semantic_mode == "GT" or semantic_mode == "ScanNetGT":
        sem_path0 = os.path.join(configs["root_path"], scene, sem_path, pair0+sem_post)
        sem_path1 = os.path.join(configs["root_path"], scene, sem_path, pair1+sem_post)
    elif semantic_mode == "SEEM":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    elif semantic_mode == "SAM":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    elif semantic_mode == "SGAM":
        sem_path0 = ""
        sem_path1 = ""
    else:
        raise NotImplementedError
    
    update_config = {
        "sem_path0": sem_path0,
        "sem_path1": sem_path1,
        "color_path0": color_path0,
        "color_path1": color_path1,
        "depth0": depth_path0,
        "depth1": depth_path1,
        "K0_path": K0_path,
        "K1_path": K1_path,
        "pose_path0": pose_path0,
        "pose_path1": pose_path1,
        "out_path": out_path
    }

    rt_configs = {**configs, **update_config}

    return rt_configs

def update_MegaDepth_config(configs, semantic_mode="SAM"):
    """ assembel paths for all pairs
    NOTE pairs for MegaDepth are {npz_file_name}_{id}
         id can be used to index image, pose and intrinsic
    Args:
        configs["root_path"]: root path of MegaDepth dataset
        configs[pair0]: {npz_name}_{id}, e.g. 0000_0.1_0.2_id
    """

    npz_folder = os.path.join(configs["root_path"], "scene_info_val_1500")
    pair0 = configs["pair0"] # npz_file_name_id, e.g. 0000_0.1_0.2_id 
    pair1 = configs["pair1"] # npz_file_name has "_id" in the end
    # get the npz file name
    pair0_npz_name = "_".join(pair0.split("_")[:-1])
    pair1_npz_name = "_".join(pair1.split("_")[:-1])
    assert pair0_npz_name == pair1_npz_name, "pair0 and pair1 should be in the same npz file"
    id0 = (pair0.split("_")[-1])
    id1 = (pair1.split("_")[-1])

    npz_path = os.path.join(npz_folder, pair0_npz_name+".npz")

    npz_data = np.load(npz_path, allow_pickle=True)

    # get the image path
    img_path0 = npz_data["image_paths"][int(id0)]
    img_path1 = npz_data["image_paths"][int(id1)]
    img_folder0 = img_path0.split("/")[1]
    img_folder1 = img_path1.split("/")[1]
    img_name0 = img_path0.split("/")[-1].split(".")[0]
    img_name1 = img_path1.split("/")[-1].split(".")[0]
    if semantic_mode == "SAM":
        sem_path = configs["sam_path"]
        sem_post = configs["sam_post"]
        sem_path0 = os.path.join(sem_path, "MegaDepth1500", img_folder0, img_name0+sem_post)
        sem_path1 = os.path.join(sem_path, "MegaDepth1500", img_folder1, img_name1+sem_post)
    elif semantic_mode == "SGAM":
        sem_path0 = ""
        sem_path1 = ""
    else:
        raise NotImplementedError

    out_path = os.path.join(configs["out_path"], f"MegaDepth_{id0}_{id1}")

    color_path0 = os.path.join(configs["root_path"], img_path0)
    color_path1 = os.path.join(configs["root_path"], img_path1)

    K0 = npz_data["intrinsics"][int(id0)].astype(np.float32)
    K1 = npz_data["intrinsics"][int(id1)].astype(np.float32)

    pose0 = npz_data["poses"][int(id0)]
    pose1 = npz_data["poses"][int(id1)]
    pose01 = np.matmul(pose1, np.linalg.inv(pose0))
    
    out_path = os.path.join(configs["out_path"], f"MegaDepth_{id0}_{id1}")

    update_config = {
        "sem_path0": sem_path0,
        "sem_path1": sem_path1,
        "color_path0": color_path0,
        "color_path1": color_path1,
        "K0": K0,
        "K1": K1,
        "pose0": pose0,
        "pose1": pose1,
        "out_path": out_path,
    }

    rt_configs = {**configs, **update_config}

    return rt_configs
    
def update_KITTI_config(configs, semantic_mode="GT"):
    """ assemble all path to file level
    Difference:
        - No depth
        - pose is in the same file
        - directly load pose and K

    Args:
        args.pair0/1
    """
    pair0 = configs["pair0"]
    pair0_non_fill = pair0.lstrip("0")
    pair0_fill = pair0.zfill(10)
    pair1 = configs["pair1"]
    pair1_non_fill = pair1.lstrip("0")
    pair1_fill = pair1.zfill(10)

    color_path0 = os.path.join(configs["root_path"], configs["color_path"], pair0_fill+configs["color_post"])
    color_path1 = os.path.join(configs["root_path"], configs["color_path"], pair1_fill+configs["color_post"])

    if semantic_mode == "GT" or semantic_mode == "ScanNetGT":
        sem_path0 = os.path.join(configs["root_path"], configs["sem_path"], pair0_fill+configs["sem_post"])
        sem_path1 = os.path.join(configs["root_path"], configs["sem_path"], pair1_fill+configs["sem_post"])
    elif semantic_mode == "SEEM":
        seq_id = configs["sequence_id"]
        sem_path0 = os.path.join(configs["root_path"], configs["sem_path_SEEM"], f"KITTI_{seq_id}", pair0_fill+configs["sem_post_SEEM"])
        sem_path1 = os.path.join(configs["root_path"], configs["sem_path_SEEM"], f"KITTI_{seq_id}", pair1_fill+configs["sem_post_SEEM"])
    elif semantic_mode == "SAM":
        seq_id = configs["sequence_id"]
        sem_path0 = os.path.join(configs["sam_path"], f"KITTI_{seq_id}", pair0_fill+configs["sam_post"])
        sem_path1 = os.path.join(configs["sam_path"], f"KITTI_{seq_id}", pair1_fill+configs["sam_post"])
    elif semantic_mode == "SGAM":
        sem_path0 = ""
        sem_path1 = ""
    else:
        raise NotImplementedError


    poses, frame_names = KITTI_load_pose(os.path.join(configs["root_path"], configs["pose_path"]))
    # get the idx of the pair in frame_names
    frame_names = frame_names.tolist()
    # turn to int, then to string
    frame_names = [str(int(name)) for name in frame_names]
    try:
        idx0 = frame_names.index(pair0_non_fill)
    except ValueError:
        logger.exception("?")
        
    idx1 = frame_names.index(pair1_non_fill)
    pose0 = poses[idx0]
    pose1 = poses[idx1]

    K0 = [552.554261, 0.000000, 682.049453, 0.000000, 552.554261, 238.769549, 0, 0, 1]
    # resize to 3x3
    K0 = np.array(K0).reshape(3,3)

    K1 = [552.554261, 0.000000, 682.049453, 0.000000, 552.554261, 238.769549, 0, 0, 1]
    # resize to 3x3
    K1 = np.array(K1).reshape(3,3)

    out_path = os.path.join(configs["out_path"], f"KITTI_{configs['sequence_id']}_{pair0}_{pair1}")

    update_config = {
        "sem_path0": sem_path0,
        "sem_path1": sem_path1,
        "color_path0": color_path0,
        "color_path1": color_path1,
        "K0": K0,
        "K1": K1,
        "pose0": pose0,
        "pose1": pose1,
        "out_path": out_path,
    }

    rt_configs = {**configs, **update_config}

    return rt_configs

def KITTI_load_pose(pose_txt):
    """ load camera-to-world pose from txt
    """
    cam2world = np.loadtxt(pose_txt)
    pose_cam2world = np.reshape(cam2world[:,1:],(-1,4,4))
    frame_names = cam2world[:,0]
    return pose_cam2world, frame_names

def update_YFCC_config(configs, semantic_mode="SAMViTB"):
    """ assemble path
    Func:
        About pair info:
            - common pair id
            - common scene name
            - common root path
        object:
            - get img path
            - get semantic path
            - get pose
            - get K
    """
    import pickle
    assert configs["pair0"] == configs["pair1"], "YFCC pair0 and pair1 should be the same as the pair id"
    pair_id = configs["pair0"]

    pairs_folder = os.path.join(configs["root_path"], configs["pair_folder"])
    pair_info_path = os.path.join(pairs_folder, configs["scene_name"]+configs["pair_post"])

    with open(pair_info_path, 'r') as f:
        pair_info = f.readlines()
        img_path0, img_path1 = pair_info[int(pair_id)].strip('\n').split(' ')

    img_name0 = img_path0.split("/")[-1].split(".")[0]
    img_name1 = img_path1.split("/")[-1].split(".")[0]

    if semantic_mode != "SGAM":
        assert semantic_mode in ["SAMViTB", "SAMViTL"], f"semantic mode should be one of {['SAMViTB', 'SAMViTL']}"
        sem_path = os.path.join(configs["root_path"], configs["sem_mode"], semantic_mode, configs["scene_name"])
        sem_path0 = os.path.join(sem_path, img_name0+configs["sem_post"])
        sem_path1 = os.path.join(sem_path, img_name1+configs["sem_post"])
    else:
        sem_path0 = ""
        sem_path1 = ""
    
    # get geo info
    geo_info_folder = os.path.join(configs["root_path"], configs["geo_folder"], configs["scene_name"], configs["geo_mid_path"])
    geo_names = ["Ris", "Rjs", "tis", "tjs", "cx1s", "cy1s", "cx2s", "cy2s", "f1s", "f2s"]

    geo_vals = {}
    for name in geo_names:
        # logger.critical(f"load {name}, geo_info_folder is {geo_info_folder}")
        geo_info_path = os.path.join(geo_info_folder, name+".pkl")
        if not os.path.exists(geo_info_path):
            logger.error(f"File: {geo_info_path} not exists!")
            return
        geo_info = pickle.load(open(geo_info_path, "rb"))
        geo_vals[name] = geo_info[int(pair_id)]
    
    pose0 = assembel_pose(geo_vals["Ris"], geo_vals["tis"]) # camera to world
    pose1 = assembel_pose(geo_vals["Rjs"], geo_vals["tjs"])
    K0 = assembel_K(geo_vals["f1s"], geo_vals["cx1s"], geo_vals["cy1s"])
    K1 = assembel_K(geo_vals["f2s"], geo_vals["cx2s"], geo_vals["cy2s"])

    out_path = os.path.join(configs["out_path"], configs["scene_name"]+"_"+pair_id)

    update_config = {
        "sem_path0": sem_path0,
        "sem_path1": sem_path1,
        "color_path0": img_path0,
        "color_path1": img_path1,
        "K0": K0,
        "K1": K1,
        "pose0": pose0,
        "pose1": pose1,
        "out_path": out_path,
    }

    return {**configs, **update_config}

def update_ScanNet_config(configs, semantic_mode="ScanNetGT"):
    """ assemble path
    Args:
        semantic_mode: ScanNetGT or SEEM or SAM, different semantic image path
        NOTE: at SAM mode, sem_path is the folder for SAM results and sem_post is the post of the result, i.e. .npy
    """
    if semantic_mode == "ScanNetGT":
        sem_path = configs["sem_path"]
        sem_post = configs["sem_post"]
    elif semantic_mode == "SEEM":
        sem_path = configs["sem_path_SEEM"]
        sem_post = configs["sem_post_SEEM"]
    elif semantic_mode == "SAM":
        sem_path = configs["sam_path"]
        sem_post = configs["sam_post"]
    elif semantic_mode == "SGAM":
        sem_path = ""
        sem_post = ""
    elif semantic_mode == "SAMViTB":
        sem_path = configs["sem_path_SEEM"]
        sem_post = configs["sem_post_SEEM"]
    elif semantic_mode == "SAMViTL":
        sem_path = configs["sem_path_SEEM"]
        sem_post = configs["sem_post_SEEM"]
    else:
        raise NotImplementedError

    color_path = configs["color_path"]
    color_post = configs["color_post"]
    depth_path = configs["depth_path"]
    depth_post = configs["depth_post"]
    color_K_path = configs["color_K_path"]
    pose_path = configs["pose_path"]
    pose_post = configs["pose_post"]

    scene = configs["scene_name"]
    pair0 = configs["pair0"]
    pair1 = configs["pair1"]

    color_path0 = os.path.join(configs["root_path"], scene, color_path, pair0+color_post)
    color_path1 = os.path.join(configs["root_path"], scene, color_path, pair1+color_post)

    depth_path0 = os.path.join(configs["root_path"], scene, depth_path, pair0+depth_post)
    depth_path1 = os.path.join(configs["root_path"], scene, depth_path, pair1+depth_post)

    K_path = os.path.join(configs["root_path"], scene, color_K_path)

    pose_path0 = os.path.join(configs["root_path"], scene, pose_path, pair0+pose_post)
    pose_path1 = os.path.join(configs["root_path"], scene, pose_path, pair1+pose_post)

    out_path = os.path.join(configs["out_path"], scene+"_"+pair0+"_"+pair1)

    if semantic_mode == "ScanNetGT":
        sem_path0 = os.path.join(configs["root_path"], scene, sem_path, pair0+sem_post)
        sem_path1 = os.path.join(configs["root_path"], scene, sem_path, pair1+sem_post)
    elif semantic_mode == "SEEM":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    elif semantic_mode == "SAM":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    elif semantic_mode == "SGAM":
        sem_path0 = ""
        sem_path1 = ""
    elif semantic_mode == "SAMViTB":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    elif semantic_mode == "SAMViTL":
        sem_path0 = os.path.join(sem_path, scene, pair0+sem_post)
        sem_path1 = os.path.join(sem_path, scene, pair1+sem_post)
    else:
        raise NotImplementedError

    update_config = {
        "sem_path0": sem_path0,
        "sem_path1": sem_path1,
        "color_path0": color_path0,
        "color_path1": color_path1,
        "depth0": depth_path0,
        "depth1": depth_path1,
        "K0_path": K_path,
        "K1_path": K_path,
        "pose_path0": pose_path0,
        "pose_path1": pose_path1,
        "out_path": out_path
    }

    rt_configs = {**configs, **update_config}

    return rt_configs

def load_config_yaml(config_path):
    """
    Returns:
        config dict
    """   
    assert('.yaml' in config_path)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config)
    return config

def load_txt_list_ret_split(path):
    """ split is \n
    """
    data = []
    with open(path, 'r') as f:
        for temp in f.readlines():
            temp.strip('\n')
            data.append(temp)
    return data
    
def load_cv_img(img_path, mode=0):
    """
    Args:
        mode: 0 = cv2.IMREAD_GRAYSCALE
        mode: 1 = cv2.IMREAD_COLOR
    """
    return cv2.imread(img_path, mode)

def load_cv_img_resize(img_path, W, H, mode=0):
    """
    """
    logger.info(f"load img from {img_path} resized to {W} x {H}")
    img = cv2.imread(img_path, mode)
    logger.info(f"raw img size is {img.shape}")
    if mode == 1:
        H_ori, W_ori, _ = img.shape
    else:
        H_ori, W_ori = img.shape
    scale_u = W / W_ori
    scale_v = H / H_ori
    # print(f"ori W, H: {W_ori} x {H_ori},  with scale: {scale_u} , {scale_v}")
    img = cv2.resize(img, (W, H), cv2.INTER_AREA)
    return img, [scale_u, scale_v]

def load_sem_pair(sem_path0, sem_path1, size_W, size_H):
    """ for MatterPort3D dataset
    NOTE:
        0 is background
    """
    color_sem0 = cv2.imread(sem_path0, 1)
    color_sem1 = cv2.imread(sem_path1, 1)

    raw_H, raw_W = color_sem0.shape[0], color_sem0.shape[1]

    scale_u = size_W / raw_W
    scale_v = size_H / raw_H
    logger.debug(f"resized by {scale_u}x{scale_v}")

    color_sem0 = cv2.resize(color_sem0, (size_W, size_H), cv2.INTER_NEAREST)
    color_sem1 = cv2.resize(color_sem1, (size_W, size_H), cv2.INTER_NEAREST)

    label_color_list = []
    color_list0 = np.unique(np.array(color_sem0).reshape(-1, 3), axis=0)
    color_list1 = np.unique(np.array(color_sem1).reshape(-1, 3), axis=0)
    
    for color0 in color_list0:
        if color0.tolist() not in label_color_list:
            label_color_list.append(color0.tolist())

    for color1 in color_list1:
        if color1.tolist() not in label_color_list:
            label_color_list.append(color1.tolist())
    
    per_color_factor = 10
    logger.debug(f"color list is {label_color_list}")

    rt_sem0 = np.zeros((size_H, size_W))
    rt_sem1 = np.zeros((size_H, size_W))

    for i, color in enumerate(label_color_list):
        if color == [0, 0, 0]:
            continue
        # sem0
        sem0_c0 = color_sem0[:,:,0] == color[0]
        sem0_c1 = color_sem0[:,:,1] == color[1]
        sem0_c2 = color_sem0[:,:,2] == color[2]

        sem0_where = sem0_c0 & sem0_c1
        sem0_where = sem0_where & sem0_c2
        
        rt_sem0[sem0_where] = (i+1) * per_color_factor

        # sem1
        sem1_c0 = color_sem1[:,:,0] == color[0]
        sem1_c1 = color_sem1[:,:,1] == color[1]
        sem1_c2 = color_sem1[:,:,2] == color[2]

        sem1_where = sem1_c0 & sem1_c1
        sem1_where = sem1_where & sem1_c2
        
        rt_sem1[sem1_where] = (i+1) * per_color_factor

    return rt_sem0, rt_sem1, [scale_u, scale_v]
    
def load_cv_depth(depth_path):
    """ for ScanNet Dataset
    """
    logger.info(f"load depth from {depth_path}")
    return cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

def load_pose_txt(pose_path):
    """For ScanNet pose: cam2world
        txt file with 
        P =
            |R t|
            |0 1|
            
    Returns:
        P : np.mat
    """
    P = np.loadtxt(pose_path)
    P = np.matrix(P)
    logger.info(f"load pose is \n{P}")
    return P

def load_K_txt(intri_path, scale=[1, 1]):
    """For ScanNet K
    Args:
        scale = [scale_u, scale_v]
    """
    K = np.loadtxt(intri_path)
    fu = K[0,0] * scale[0]
    fv = K[1,1] * scale[1]
    cu = K[0,2] * scale[0]
    cv = K[1,2] * scale[1]
    K_ = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

    logger.info(f"load K from {intri_path} with scale {scale} is \n {K_}")
    return np.matrix(K_)

#=========================================================== Normal file process
def get_filelist_of_folder(folder_path, post):
    """ get file with post
    """
    import glob
    import os

    return glob.glob(os.path.join(folder_path, "*."+post))

def sort_filelist2num(file_list):
    """
    """
    file_list.sort(key=lambda x: int(x.split('.')[0]))

def test_dir_if_not_create(path):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:
    """
    import os
    if os.path.isdir(path):
        return True
    else:
        print('Create New Folder:', path)
        os.makedirs(path)
        return True

