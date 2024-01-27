'''
Author: Eason
Date: 2022-07-09 15:35:50
LastEditTime: 2024-01-27 12:34:58
LastEditors: EasonZhang
Description: geometric tools
FilePath: /SGAM/utils/geo.py
'''

import numpy as np
import math
from loguru import logger
# import torch
import cv2
from collections import OrderedDict
import copy


def nms_for_corrs(corrs, r=3):
    """ perform Grid-based even sample for corrs
    Args:
        corrs: [[u0, v0, u1, v1], ...]
        r: radius
    """
    corrs_np = copy.copy(np.array(corrs)) # Nx4
    
    u0s = corrs_np[:, 0]
    v0s = corrs_np[:, 1]
    u0_min = u0s.min()
    u0_max = u0s.max()
    v0_min = v0s.min()
    v0_max = v0s.max()

    # generate grid
    u0_range = np.arange(u0_min, u0_max, r)
    v0_range = np.arange(v0_min, v0_max, r)
    u0_grid, v0_grid = np.meshgrid(u0_range, v0_range)
    u0_grid = u0_grid.flatten()
    v0_grid = v0_grid.flatten()

    # put corrs into grid
    corrs_after = []
    for i in range(len(u0_grid)):
        u0, v0 = u0_grid[i], v0_grid[i]
        corrs_in_grid = corrs_np[np.abs(u0s - u0) < r/2]
        v0s = corrs_in_grid[:, 1]
        corrs_in_grid = corrs_in_grid[np.abs(v0s - v0) < r/2]
        if len(corrs_in_grid) == 0:
            continue
        corrs_in_grid = corrs_in_grid[0].tolist()
        corrs_after.append(corrs_in_grid)
    
    return corrs_after

### detached from OANet
"""NOTE dR, dt here comes from world2cam pose 0to1"""
def np_skew_symmetric(v):
    """
    Args:
        v: Nx3
    """

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def np_unskew_symmetric(M):

    v = np.concatenate([
        0.5 * (M[:, 7] - M[:, 5])[None],
        0.5 * (M[:, 2] - M[:, 6])[None],
        0.5 * (M[:, 3] - M[:, 1])[None],
    ], axis=1)

    return v

def get_episqr(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()

    ys = x2Fx1**2

    return ys.flatten()

def get_episym(x1, x2, dR, dt):
    """
    Args:
        x1/2 [N, 2], normalized by K
    """
    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()

def get_sampsons(x1, x2, dR, dt):
    """ Compute Sampson error
    Args:
        x1: Nx2
        x2: Nx2
        dR: 3x3, rotation matrix from 0 to 1
        dt: 3
    """
    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 / (
        Fx1[..., 0]**2 + Fx1[..., 1]**2 + Ftx2[..., 0]**2 + Ftx2[..., 1]**2
    )

    return ys.flatten()
### detached from OANet DONE

def adopt_K(K, scale):
    """
    Args:
        scale: [scale_x, scale_y]
    """
    fu = K[0,0] * scale[0]
    fv = K[1,1] * scale[1]
    cu = K[0,2] * scale[0]
    cv = K[1,2] * scale[1]
    K_ = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    return K_

def assembel_K(f, cx, cy):
    # logger.critical(f"assembel K with f {f}, cx {cx}, cy {cy}")
    fx = f[0]
    fy = f[1]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).reshape(3, 3)

def assembel_pose(R, t):
    """ assembel pose
    Args:
        R: 3x3
        t: 3x1
    Returns:
        4x4
    """
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.reshape(3)
    return pose

def warp_area_by_MC(area0, depth0_, depth1_, K0, K1, pose0, pose1, sample_step=2, depth_factor=1):
    """
    Args:
        area0: [u_min, u_max, v_min, v_max]
        pose0/1: cam2world pose
    Returns:
        area1: [u_min, u_max, v_min, v_max]
    """
    try:
        depth0 = depth0_.copy().astype(np.float32)
        depth1 = depth1_.copy().astype(np.float32)
        depth0 /= depth_factor
        depth1 /= depth_factor
    except Exception as e:
        logger.exception(e)
        raise ValueError

    # sample points in area0
    u_min, u_max, v_min, v_max = area0
    u_min, u_max = int(u_min), int(u_max)
    v_min, v_max = int(v_min), int(v_max)
    u_range = np.arange(u_min, u_max, sample_step)
    v_range = np.arange(v_min, v_max, sample_step)
    u, v = np.meshgrid(u_range, v_range)
    # logger.success(f"sample {u.shape} points")

    # achieve depth0
    depth0 = depth0[v, u]
    # depth0 /= depth_factor
    depth0_mask = depth0 > 0
    depth0 = depth0[depth0_mask]
    u = u[depth0_mask]
    v = v[depth0_mask]
    # logger.success(f"after deptht0 mask get {u.shape} points")

    # achieve 3D points in cam0
    x = (u - K0[0, 2]) * depth0 / K0[0, 0]
    y = (v - K0[1, 2]) * depth0 / K0[1, 1]
    z = depth0

    # transform 3D points from cam0 to world
    pts0 = np.stack([x, y, z], axis=1) # Nx3
    pts0 = np.concatenate([pts0, np.ones_like(pts0[:, :1])], axis=1) # Nx4
    # logger.success(f"pts0 shape is {pts0.shape} ")
    pts0 = np.matmul(pose0, pts0.T).T # Nx4

    # transform 3D points from world to cam1
    pts1 = np.matmul(pose1.I, pts0.T).T # Nx4
    z1_compute = pts1[:, 2]

    # transform 3D points from cam1 to image1
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    u1 = (x1 * K1[0, 0] / z1_compute) + K1[0, 2]
    v1 = (y1 * K1[1, 1] / z1_compute) + K1[1, 2]
    u1 = u1.astype(int)
    v1 = v1.astype(int)
    
    # check if pts1 is in the image
    H, W = depth1.shape
    co_visible_mask = (u1 > 0) & (u1 < W-1) & (v1 > 0) & (v1 < H-1)
    u1 = u1[co_visible_mask]
    v1 = v1[co_visible_mask]
    z1_compute = z1_compute[co_visible_mask]
    # logger.success(f"after depth1 mask get {u1.shape} pts")

    # check depth consistency
    depth1 = depth1[v1, u1]
    # depth1 /= depth_factor
    depth_diff = np.abs(depth1 - z1_compute) / z1_compute
    depth_diff_mask = depth_diff < 0.2

    final_u1 = u1[depth_diff_mask].T
    final_v1 = v1[depth_diff_mask].T

    # logger.success(f"after depth consistency get {final_u1.shape} pts")
    if len(final_u1) < 10: raise ValueError

    # get the area1
    u1_min = final_u1.min()
    u1_max = final_u1.max()
    v1_min = final_v1.min()
    v1_max = final_v1.max()

    area1 = [u1_min, u1_max, v1_min, v1_max]

    # logger.critical(f"warp from \narea0 {area0} to \narea1 {area1} ")

    return area1, area0

def pose2F(pose0, pose1, K0, K1):
    """ calc F from pose
    Args:
        pose0/1: cam2world pose, np.matrix
    """
    K0 = np.matrix(K0)
    K1 = np.matrix(K1)


    # get world2cam pose
    pose0_w2c = pose0.I
    pose1_w2c = pose1.I
    R0 = pose0_w2c[:3, :3]
    R1 = pose1_w2c[:3, :3]
    t0 = pose0_w2c[:3, 3]
    t1 = pose1_w2c[:3, 3]

    # get dR, dt of T_0to1
    dR_0to1 = np.matmul(R1, R0.I)
    dt_0to1 = t1 - np.dot(dR_0to1, t0)
    dt_0to1 = np.array(dt_0to1).reshape(1,3)

    # normalize dt_0to1
    try:
        dt_norm = np.sqrt(np.sum(dt_0to1**2))
    except Exception as e:
        logger.exception(e)
        logger.critical(f"dt_0to1 is {dt_0to1}")
        raise ValueError

    dt_0to1 /= dt_norm

    # get F
    try:
        t_xR = np.matmul(
            np.reshape(np_skew_symmetric(dt_0to1), (3, 3)),
            dR_0to1
        ) # 3x3
    except Exception as e:
        logger.exception(e)
        logger.critical(f"dt_0to1 is {dt_0to1}, dR_0to1 is {dR_0to1}")
        raise ValueError

    F = np.matmul(K1.I.T, np.matmul(t_xR, K0.I))

    return F

def fuse_corr(corr_list, corr, fuse_mode="rough", fuse_thd=1):
    """ if corr is in corr_list, fuse them
        in_list_judge_metric: both uv0 and uv1 has distance<1 with corr in list
    """
    if len(corr_list) == 0:
        corr_list.append(corr)
        return
    # corrs shape is [[u0, v0, u1, v1], ...]
    # for corr_check in corr_list:
    #     assert len(corr_check) == 4, f"invalid corr {corr_check}"
    corrs_np = np.array(corr_list)
    corr_np = np.array(corr)
    # calc the distance between corr and corrs in list
    dists = np.sqrt(np.sum((corrs_np - corr_np)**2, axis=1))
    # find the min dist
    min_dist = np.min(dists)
    if min_dist <= fuse_thd:
        fuse_idx = np.argmin(dists)
        if fuse_mode == "rough":
            corr_list[fuse_idx] = corr
        elif fuse_mode == "average":
            ave_corr = np.array(corr_list[fuse_idx]) + np.array(corr)
            ave_corr /= 2
            corr_list[fuse_idx] = ave_corr
    else:
        corr_list.append(corr)

def calc_areas_iou(area0, area1):
    """ calc areas iou
    Args:
        area0: [u_min, u_max, v_min, v_max]
        area1: [u_min, u_max, v_min, v_max]
    """
    u_min0, u_max0, v_min0, v_max0 = area0
    u_min1, u_max1, v_min1, v_max1 = area1

    u_min = max(u_min0, u_min1)
    u_max = min(u_max0, u_max1)
    v_min = max(v_min0, v_min1)
    v_max = min(v_max0, v_max1)

    if u_min >= u_max or v_min >= v_max:
        return 0

    area0 = (u_max0 - u_min0) * (v_max0 - v_min0)
    area1 = (u_max1 - u_min1) * (v_max1 - v_min1)
    area = (u_max - u_min) * (v_max - v_min)

    return area / (area0 + area1 - area + 1e-5)

def tune_corrs_size(corrs, src_W, src_H, dst_W, dst_H):
    """
    Args:
        corrs: [[u0, v0, u1, v1], ...]
    Returns:
        rt_corrs: [[u0, v0, u1, v1], ...]
    """
    rt_corrs = []
    W_ratio = dst_W / src_W
    H_ratio = dst_H / src_H

    for corr in corrs:
        u0, v0, u1, v1 = corr
        u0_ = u0 * W_ratio
        v0_ = v0 * H_ratio
        u1_ = u1 * W_ratio
        v1_ = v1 * H_ratio

        rt_corrs.append([u0_, v0_, u1_, v1_])
    
    return rt_corrs

def tune_mkps_size(mkps, src_W, src_H, dst_W, dst_H):
    """ tune mkps size to dst size
    Args:
        mkps: np.array [[u, v], ...]
    Returns:
        rt_mkps: np.array [[u, v], ...]
    """
    rt_mkps = []
    W_ratio = dst_W / src_W
    H_ratio = dst_H / src_H

    for mkp in mkps:
        u, v = mkp
        u_ = u * W_ratio
        v_ = v * H_ratio

        rt_mkps.append([u_, v_])
    
    return np.array(rt_mkps)

def cal_corr_F_and_mean_sd(corrs_raw):
    """
    Args:
        corrs: [[u0, v0, u1, v1], ...]
    """
    corrs = copy.deepcopy(corrs_raw)
    if len(corrs) < 8:
        logger.error(f"too few corrs {len(corrs)}")
        return np.zeros((3,3)), 10000
    corrs = np.array(corrs)
    corrs_F0 = corrs[:, :2]
    corrs_F1 = corrs[:, 2:]
    logger.info(f"achieve corrs with shape {corrs_F0.shape} == {corrs_F1.shape} to calc F")
    F, mask = cv2.findFundamentalMat(corrs_F0, corrs_F1, method=cv2.FM_RANSAC,ransacReprojThreshold=1, confidence=0.99)
    # get corrs inliers from mask
    corrs_inliers = corrs[mask.ravel()==1]

    # calc mean sampson dist
    samp_dist = calc_sampson_dist(F, corrs_inliers)

    return F, samp_dist

def norm_pts_NT(pts):
    """ use 8-pts algorithm normalization
    """
    norm_pts = pts.copy()
    N = pts.shape[0]
    
    mean_pts = np.mean(pts, axis=0)
    
    mins_mean_pts = pts - mean_pts

    pts_temp = np.mean(np.abs(mins_mean_pts), axis=0)
    pts_temp+=1e-5

    norm_pts = mins_mean_pts / pts_temp

    # logger.info(f"after norm pts shape = {norm_pts.shape}")

    return norm_pts

def calc_sampson_dist(F, corrs):
    """
    Args:
        F: 3x3
        corrs: np.ndarray [[u0, v0, u1, v1], ...]
    """
    assert len(corrs.shape) == 2 and corrs.shape[1] == 4, f"invalid shape {corrs.shape}"
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    uv0_norm = norm_pts_NT(uv0)
    uv1_norm = norm_pts_NT(uv1)
    uv0_h, uv1_h = Homo_2d_pts(uv0_norm), Homo_2d_pts(uv1_norm) # N x 3
    samp_dist = 0

    for i in range(corrs.shape[0]):
        samp_dist += calc_sampson_1_pt(F, uv0_h[i,:], uv1_h[i,:])
    
    samp_dist /= corrs.shape[0]

    return samp_dist

def calc_pFq(corr, F):
    """ calc q^T F p
    Args:
        corr: [u0, v0, u1, v1]
    """
    assert F.shape == (3,3)


    kpts0 = np.array([corr[0], corr[1]])
    kpts1 = np.array([corr[2], corr[3]])

    uv0H = np.array([kpts0[0], kpts0[1], 1])
    uv1H = np.array([kpts1[0], kpts1[1], 1]) # 1x3

    try:
        up = np.matmul(uv1H, np.matmul(F, uv0H.reshape(3,1)))
    except Exception as e:
        logger.exception(e)


    return up

def assert_match_qFp(corrs, K0, K1, pose0, pose1, thd):
    """ assert the corrs with Fundamental Matrix
    Args:
        corrs: corrs list
        F: Fundamental Matrix
        thd: threshold
    Returns:
        mask: mask of corrs
        bad_ratio: ratio of bad corrs
    """
    mask = []
    bad_ratio = 0

    F = pose2F(pose0, pose1, K0, K1)

    for corr in corrs:
        pFq = calc_pFq(corr, F)
        if pFq < thd:
            mask.append(1)
        else:
            mask.append(0)
            bad_ratio += 1
    bad_ratio /= len(corrs)

    return mask, bad_ratio

def calc_sampson_1_pt(F, uv0H, uv1H):
    """
    Args:
        uviH: 1 x 3
        F: 3 x 3
    Returns:
        (uv1H^T * F * uv0H)^2 / [(F*uv0H)_0^2 + (F*uv0H)_1^2 + (F^T*uv1H)_0^2 + (F^T*uv1H)_1^2]
    """
    uv0H = uv0H.reshape((1,3))
    uv1H = uv1H.reshape((1,3))

    assert uv0H.shape[0] == 1 and uv0H.shape[1] == 3, f"invalid shape {uv0H.shape}"
    assert uv1H.shape[0] == 1 and uv1H.shape[1] == 3, f"invalid shape {uv1H.shape}"

    # logger.debug(f"calc sampson dist use:\n{uv0H}\n{uv1H}")

    up = np.matmul(uv1H, np.matmul(F, uv0H.reshape(3,1)))[0][0]
    up = up**2
    Fx0 = np.matmul(F, uv0H.T)
    FTx1 = np.matmul(F.T, uv1H.T)
    # logger.info(f"Fx1 = {Fx1}\nFTx0 = {FTx0}")
    down = Fx0[0,0]**2 + Fx0[1,0]**2 + FTx1[0,0]**2 + FTx1[1, 0]**2

    # logger.debug(f"calc sampson dist use {up} / {down}")
    
    dist = up / (down + 1e-5)


    return dist

def adjust_K(K, scale):
    """
    Args:
        scale: [scale_x, scale_y]
    Returns:
        np.ndarray
    """
    fu = K[0,0] * scale[0]
    fv = K[1,1] * scale[1]
    cu = K[0,2] * scale[0]
    cv = K[1,2] * scale[1]
    K_ = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    return K_

def img2cam(pt, K, depth, Z_neg=0):
    """
    Args:
        pt [2,] np.array
    Returns:
        3 x 1 mat
    """
    # logger.info(f"K is \n{K}\nd is {depth}, uv is {pt}")
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    [u, v] = pt
    X = (u - cx)*depth / fx
    Y = (v - cy)*depth / fy
    if Z_neg == 0:
        Z = depth
    elif Z_neg == 1:
        Z = -depth
    else:
        raise KeyError
    Pt = np.matrix([[X, Y, Z]]).T
    return Pt

def cam2img(Pt, K, Z_neg=0):
    """
    Args:
        Pt [3x1] XYZ
        Z_neg is for MatterPort3D
    Returns:
        pt [u, v] np.array[2,]
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    X = Pt[0,0]
    Y = Pt[1,0]
    if Z_neg == 0:
        Z = Pt[2,0]
    elif Z_neg == 1:
        Z = -Pt[2,0]
    else:
        raise KeyError

    u = X/Z*fx + cx
    v = Y/Z*fy + cy

    return np.array([u,v])

def Homo_2d(Pt):
    """
    Args:
        Pt 2x1
    """
    return np.row_stack((Pt, np.array([[1]])))

def Homo_2d_pts(pts):
    """ 
    Args:
        pts: Nx2 
    """
    N = pts.shape[0]
    homo = np.ones((N, 1))

    pts_homo = np.hstack((pts, homo))
    # logger.info(f"homo pts to {pts_homo.shape}")
    return pts_homo

def Homo_3d(Pt):
    """ make coordinate homogeneous
    Args:
        Pt: 3x1
    """
    return np.row_stack((Pt, np.array([[1]])))
    
def achieve_depth(pt, depth_map):
    """
    """
    [u, v] = pt
    # u_ = int(math.floor(u))
    # v_ = int(math.floor(v))
    u_ = int(round(u))
    v_ = int(round(v))
    W, H = depth_map.shape[1], depth_map.shape[0]
    if u_ < 0 or u_ >= W or v_ < 0 or v_ >= H:
        return -1

    return depth_map[v_, u_] if depth_map[v_, u_] != 0 else -1

def achieve_depth_by_K(pt, depth_map, K, K_depth, mode="Bilinear"):
    """For ScanNet depth achieve
    """
    pt = np.round(pt).astype(int)
    depth = -1

    [u, v] = pt
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    fx_d = K_depth[0,0]
    fy_d = K_depth[1,1]
    cx_d = K_depth[0,2]
    cy_d = K_depth[1,2]

    u_d = (fx_d / fx) * (u - cx) + cx_d
    v_d = (fy_d / fy) * (v - cy) + cy_d

    [W, H] = depth_map.shape

    if u_d <= 0 or u_d >= W or v_d <= 0 or v_d >= H:
        logger.info(f"Unvalid depth in ({u_d:.2f}, {v_d:.2f})")
        return depth

    if mode == "Bilinear":
        u_raw, v_raw = u_d, v_d
        u_f = math.floor(u_raw)
        v_f = math.floor(v_raw)
        u_ = u_raw - u_f
        v_ = v_raw - v_f
        depth = (1 - u_)*(1 - v_)*depth_map[v_f, u_f] + (u_)*(1 - v_)*depth_map[v_f+1, u_f] + (1 - u_)*v*depth_map[v_f, u_f+1] + u*v*depth_map[v_f+1, u_f+1]
    else:
        raise NotImplementedError

    return depth

def re_projection(pt, depth, K, T):
    """
    Args:
        pt [2,]
        T01: np.matrix
    Returns:
        pt_target: [u, v] [2,]
    """
    # img to camera
    Pt_cam = img2cam(pt, K, depth)
    Pt_cam = Homo_3d(Pt_cam)

    Pt_cam_target = T @ Pt_cam
    # logger.info(f"get P_targrt is \n{Pt_cam_target}")
    Pt_cam_target = Pt_cam_target[:3]
    
    p1 = cam2img(Pt_cam_target, K)

    # logger.info(f"get target img coor is \n{p1}")

    return p1

def inv_proj(pt, depth, K, pose, Z_neg=0):
    """img to world
    """
    # print(f"depth is {depth}")
    Pt_cam = img2cam(pt, K, depth, Z_neg)
    Pt_cam = Homo_3d(Pt_cam)

    Pt_w = pose @ Pt_cam

    return Pt_w[:3]

def calc_euc_dist_2d(pt0, pt1):
    """
    Args:
        pt [u, v]
    """
    # logger.info(f"calc 2d euc dist between {pt0} and {pt1}")
    (u0, v0) = pt0
    (u1, v1) = pt1
    return math.sqrt((u0-u1)**2 + (v0 -v1)**2)

def calc_euc_dist_3d(P0, P1):
    """
    """
    [X0, Y0, Z0] = P0
    [X1, Y1, Z1] = P1

    return math.sqrt((X0-X1)**2 + (Y0-Y1)**2 + (Z0-Z1)**2)

def recover_corrs_offset(corrs, offsets):
    """
    Args:
        corrs: [u0, v0, u1, v1]
    """
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    # print("uv0 size:", uv0.shape)
    # print("uv1 size:", uv1.shape)

    u0_offset, v0_offset, u1_offset, v1_offset = offsets
    # print("offsets: ", u0_offset, v0_offset, u1_offset, v1_offset)

    ori_matches = []
    N = uv0.shape[0]
    assert N == uv1.shape[0]
    for i in range(N):
        u0, v0 = uv0[i]
        # print("u0:", u0 )
        # print("v0:", v0 )
        u1, v1 = uv1[i]
        ori_matches.append([u0+u0_offset, v0+v0_offset, u1+u1_offset, v1+v1_offset])
    
    logger.info(f"fuse {len(ori_matches)} matches")
    # print("ori match\n", ori_matches)
    return ori_matches

def recover_corrs_offset_scales(corrs, offsets, scales):
    """
        Args:
            corrs: [[u0, v0, u1 v1]s]: NOTE np.ndarray Nx4
            offsets: [u0_offset, v0_offset, u1_offset, v1_offset]
            scales: [u0_scale, v0_scale, u1_scale, v1_scale]
        Returns:
            ori_matches: [[u0, v0, u1, v1]s] NOTE list
    """
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    logger.info(f"fuse with offset: {offsets} and scales: {scales}")
    # print("uv0 size:", uv0.shape)
    # print("uv1 size:", uv1.shape)

    u0_offset, v0_offset, u1_offset, v1_offset = offsets
    u0_scale, v0_scale, u1_scale, v1_scale = scales
    # print("offsets: ", u0_offset, v0_offset, u1_offset, v1_offset)

    ori_matches = []
    N = uv0.shape[0]
    assert N == uv1.shape[0]
    for i in range(N):
        u0, v0 = uv0[i]
        u1, v1 = uv1[i]
        u0_ = u0 * u0_scale
        v0_ = v0 * v0_scale
        u1_ = u1 * u1_scale
        v1_ = v1 * v1_scale
        ori_matches.append([u0_+u0_offset, v0_+v0_offset, u1_+u1_offset, v1_+v1_offset])
    
    logger.info(f"fuse {len(ori_matches)} matches")
    # print("ori match\n", ori_matches)
    return ori_matches

def assert_match_reproj(matches, depth0, depth1, depth_factor, K0, K1, pose0, pose1, thred=0.2, Z_neg=0):
    """ NOTE: For ScanNet whose pose is cam to world & depth factor is 1000
    Args:
        matches: [u0, v0, u1, v1]s
    Returns:
        mask: [0/1/-1]:
            0 - false match
            1 - true match
           -1 - invalid depth
    """
    mask = []
    gt_pts = []
    bad_count = 0
    l = len(matches)

    if len(matches) == 0:
        return [], 100, []
    
    for match in matches:
        u0, v0, u1, v1 = match
        d0 = achieve_depth([u0, v0], depth0)
        d1 = achieve_depth([u1, v1], depth1)

        if d0 == -1 or d1 == -1:
            mask.append(-1)
            gt_pts.append([0,0])
            l-=1
            continue

        d0 /= depth_factor
        d1 /= depth_factor
        
        P_w = inv_proj([u0, v0], d0, K0, pose0, Z_neg)
        P_w = Homo_3d(P_w)
        P0_C1 = np.matmul(pose1.I, P_w)
        p1 = cam2img(P0_C1, K1, Z_neg)
        gt_pts.append(p1)

        dist = calc_euc_dist_2d(p1, np.array([u1, v1]))

        if dist > thred:
            # logger.info(f"bad match with dist = {dist}")
            mask.append(0)
            bad_count += 1
        else:
            mask.append(1)
    
    bad_ratio = bad_count / (l + 1e-5) * 100
    # logger.info(f"assert match bad ratio is {bad_ratio}")
    if l == 0:
        bad_ratio = 100

    return mask, bad_ratio, gt_pts
    
# --- METRICS ---

def R_t_err_calc(T_0to1, R, t, ignore_gt_t_thr=0.0):
    """ same as metric in paper Sparse-to-Local_Dense
        t_err = normed(t_gt)^T * t_est
        R_err = 100/scale_t * angle_err(R)
    """
    t_gt = T_0to1[:3, 3]

    # calc the absolute error of t
    scale_t = np.linalg.norm(t_gt)
    t_err = np.linalg.norm(t_gt) * np.dot(t_gt, t) # / (np.linalg.norm(t_gt) * np.linalg.norm(t))

    # calc the angle error of R
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    angle_err_R = np.rad2deg(np.abs(np.arccos(cos)))

    R_err = 100 / scale_t * angle_err_R

    return float(t_err), float(R_err)

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    try:
        t_err = np.minimum(t_err, 180 - t_err)[0][0]  # handle E ambiguity
    except IndexError:
        t_err = min(t_err, 180 - t_err)

        
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return float(t_err), float(R_err)

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    """
    Returns:
        ret = (R, t, mask)
    """
    if len(kpts0) < 5:
        return None

    K0 = np.array(K0)
    K1 = np.array(K1)
    
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    logger.info(f"get {errors.shape} errors" )
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def pose_auc(errors, thresholds):
    logger.info(f"get {errors.shape} errors" )
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    # return aucs
    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates based on pair name
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    return {**aucs, **precs}

def aggregate_pose_auc_simp(rt_errors, mode="ScanNet"):
    """
    Args:
        rt_errors: np.array Nx2
    """
    thds = [5, 10, 20]

    pose_errs = np.max(rt_errors, axis=1)
    logger.info(f"pose errs max  = {pose_errs.shape}")
    # aucs = error_auc(pose_errs, thds)
    aucs = pose_auc(pose_errs, thds) # superglue

    return aucs

def aggregate_pose_auc(rt_errors, mode="ScanNet"):
    """
    Args:
        rt_errors: [R_errs, t_errs]
            R_errs = [err, ...] list of float
                err: float
    Returns:
        aucs = {
            thd: auc,
        }
    """
    thds = [5, 10, 20]

    pose_errs = np.max(np.stack([rt_errors[0], rt_errors[1]]), axis=0)
    logger.info(f"pose errs max  = {pose_errs.shape}")
    aucs = pose_auc(pose_errs, thds) # superglue

    return aucs

def compute_pose_only(corrs, K0, K1):
    """
    Returns:
        ret = (R, t)
    """
    corrs = np.array(corrs) # N x 4
    pts0, pts1 = corrs[:, :2], corrs[:, 2:]
    pose_ret = estimate_pose(pts0, pts1, K0, K1, 0.5, 0.9999)

    if pose_ret is None:
        logger.warning(f"use len={corrs.shape[0]} corrs to compute failed")
        return None
    else:
        R, t, inlier = pose_ret
        logger.info(f"use len={corrs.shape[0]} corrs to compute pose = t-{t}, R-{R}")
        return R, t

def get_project_map(depth0, depth1, K0, K1, pose0, pose1, depth_factor):
    """ project each point in depth0 to depth1 and record them in a map
        project points in co-vis area from depth1 to depth0 and record them in a map
    Returns:
        proj_map01 = [pt0s,pt1s]
        pt0s = [[u0, v0]s]
    """
    W_d0, H_d0 = depth0.shape[1], depth0.shape[0]
    W_d1, H_d1 = depth1.shape[1], depth1.shape[0]

    assert W_d0 == W_d1 and H_d0 == H_d1, "depth0 and depth1 should have same size"

    proj_map01 = [[], []]

    for u0 in range(W_d0):
        for v0 in range(H_d0):
            d0 = achieve_depth([u0,v0], depth0)
            if d0 <= 0: continue

            d0 /= depth_factor
            
            P_w = inv_proj([u0, v0], d0, K0, pose0)
            P_w = Homo_3d(P_w)
            P0_C1 = np.matmul(pose1.I, P_w)
            p1 = cam2img(P0_C1, K1)

            u1, v1 = p1
            u1 = int(u1)
            v1 = int(v1)
            
            # eval u1, v1 inside image1
            if u1 >= 0 and u1 < W_d1 and v1 >= 0 and v1 < H_d1:
                proj_map01[0].append([u0, v0])
                proj_map01[1].append([u1, v1])

    # calc for image1 to image0
    co_area = get_area_from_pts(proj_map01[1])
    u1_min, u1_max, v1_min, v1_max = co_area
    # spread co-area
    u1_min = max(0, u1_min - 20)
    u1_max = min(W_d1, u1_max + 20)
    v1_min = max(0, v1_min - 20)
    v1_max = min(v1_max + 20, H_d1)

    for u1_ in range(u1_min, u1_max):
        for v1_ in range(v1_min, v1_max):
            if [u1_, v1_] in proj_map01[1]: continue
            d1 = achieve_depth([u1_, v1_], depth0)
            if d1 <= 0: continue

            d1 /= depth_factor

            P_w = inv_proj([u1_, v1_], d1, K1, pose1)
            P_w = Homo_3d(P_w)
            P1_C0 = np.matmul(pose0.I, P_w)
            p0 = cam2img(P1_C0, K0)

            u0_i, v0_i = p0
            u0_i, v0_i = int(u0_i), int(v0_i)

            if u0_i >= 0 and u0_i < W_d0 and v0_i >= 0 and v0_i < H_d0:
                proj_map01[0].append([u0_i, v0_i])
                proj_map01[1].append([u1_, v1_])

    return proj_map01

def stastic_area_size(area, depth):
    """ get pts num inside area with depth > 0
    """
    area_int = [int(num) for num in area]
    umin, umax, vmin, vmax = area_int

    size = 0
    for u in range(umin, umax):
        for v in range(vmin, vmax):
            if achieve_depth([u,v], depth) > 0:
                size += 1

    return size

def eval_area_match_performence_from_proj_map(proj_map, matched_area0s, matched_area1s, depth0, depth1, aor=False):
    """ for each point pair in proj_map, stastic its occurance in every matched_area_pair
        get area_overlap_ratio and area_cover_ratio
    Args:
        within_nums = [[nums_in_area0, nums_in_area1]s]
        cover_num = int    
    """
    cover_num = 0
    within_nums = [[[], []] for _ in range(len(matched_area0s))]

    area0_sizes = [stastic_area_size(area, depth0) for area in matched_area0s]
    area1_sizes = [stastic_area_size(area, depth1) for area in matched_area1s]

    # area0_sizes = [(area[1] - area[0])*(area[3] - area[2]) for area in matched_area0s]
    # area1_sizes = [(area[1] - area[0])*(area[3] - area[2]) for area in matched_area1s]
    
    total_matched_pt_num = len(proj_map[0])
    area_pair_num = len(matched_area0s)
    assert len(matched_area0s) == len(matched_area1s)


    for pt0, pt1 in zip(proj_map[0], proj_map[1]):
        count_flag = False
        for i, (area0, area1) in enumerate(zip(matched_area0s, matched_area1s)):
            if eval_pt_in_area(pt0, area0) and eval_pt_in_area(pt1, area1):
                if not count_flag:
                    cover_num += 1
                    count_flag = True
                if aor:
                    within_nums[i][0].append(pt0)
                    within_nums[i][1].append(pt1)

            
            # elif eval_pt_in_area(pt0, area0):
            #     area0_sizes[i] += 1
            # elif eval_pt_in_area(pt1, area1):
            #     area1_sizes[i] += 1
            
    
    area_cover_ratio = cover_num / (total_matched_pt_num + 1e-5)

    area_overlap_ratios = []
    if aor:
        for i, (area0_size, area1_size) in enumerate(zip(area0_sizes, area1_sizes)):
            covered_area0 = len(within_nums[i][0])
            covered_area1 = len(within_nums[i][1])
            cover_size0 = (covered_area0[1] - covered_area0[0]) * (covered_area0[3] - covered_area0[2])
            cover_size1 = (covered_area1[1] - covered_area1[0]) * (covered_area1[3] - covered_area1[2])
            area_overlap_ratio = (cover_size0 / (area0_size + 1e-5) + cover_size1 / (area1_size + 1e-5)) / 2
            area_overlap_ratios.append(area_overlap_ratio)

    return area_cover_ratio, area_overlap_ratios 

def get_area_from_pts(pts):
    """ get bbox from a list of points
    Args:
        pts: [[u,v]s]
    """
    pts = np.array(pts)
    umin = np.min(pts[:, 0])
    umax = np.max(pts[:, 0])
    vmin = np.min(pts[:, 1])
    vmax = np.max(pts[:, 1])

    return [umin, umax, vmin, vmax]

def compute_pose_error_simp(corrs, K0, K1, gt_pose, pix_thd=0.5, conf=0.9999):
    """
    Args:
        corrs: [corrs]
    Returns:
        error: [R_err, t_err]
    """

    rt_errs = [180, 180]
    corrs = np.array(corrs) # N x 4
    pts0, pts1 = corrs[:, :2], corrs[:, 2:]
    pose_ret = estimate_pose(pts0, pts1, K0, K1, pix_thd, conf)

    if pose_ret is None:
        logger.success(f"use len={corrs.shape[0]} corrs to eval pose err failed")
        return rt_errs
    else:
        R, t, inlier = pose_ret
        # logger.info(f"achieve R:\n{R}\n t:{t} \n gt pose is \n{gt_pose}")
        t_err, R_err = relative_pose_error(gt_pose, R, t, 0.0)
        logger.success(f"use len={corrs.shape[0]} corrs to eval pose err = t-{t_err:.4f}, R-{R_err:.4f}")
        rt_errs[0] = R_err
        rt_errs[1] = t_err
    
    return rt_errs

def area_match_aor_MC(area0, area1, K0, K1, pose0, pose1, depth0, depth1, depth_factor):
    """
    """
    area0_1, _ = warp_area_by_MC(area0, depth0, depth1, K0, K1, pose0, pose1, depth_factor=depth_factor)
    aor_0to1_1 = calc_areas_iou(area0_1, area1)
    area1_0, _ = warp_area_by_MC(area1, depth1, depth0, K1, K0, pose1, pose0, depth_factor=depth_factor)
    aor_1to0_1 = calc_areas_iou(area1_0, area0)
    aor = (aor_0to1_1 + aor_1to0_1) / 2

    return aor

def assert_inside_area(pt, area):
    """
    Returns:
        True - pt in area
    """
    umin, umax, vmin, vmax = area
    u, v = pt

    if u <= umax and u >= umin and v <= vmax and v >= vmin:
        return True
    
    return False

def eval_pt_in_area(pt, area):
    """ 
    Args:
        pt: [u,v]
        area: [u_min, u_max, v_min, v_max]
    """
    u, v = pt
    u_min, u_max, v_min, v_max = area
    if u < u_min or u > u_max or v < v_min or v > v_max:
        return False
    
    return True


""" calc the area matching performance"""

def fuse_areas(areas):
    """ fuse a list of areas
    """
    areas_np = np.array(areas)
    u_min = np.min(areas_np[:, 0])
    u_max = np.max(areas_np[:, 1])
    v_min = np.min(areas_np[:, 2])
    v_max = np.max(areas_np[:, 3])

    return [u_min, u_max, v_min, v_max]

def project_point(pt, depth0, K0, K1, pose0, pose1, depth_factor, search_direction):
    """ project pt from image0 to image1
        achieve the depth of pt in image0
        if no valid depth, search around pt to get a valid depth and change pt
    Returns:
        pt1: [u1, v1]
        pt0: [u0, v0]
        changed: bool
    """
    changed = False

    depth, pt, changed = achieve_depth_search_around(pt, depth0, search_direction)

    if pt is None:
        return None, None, changed

    u0, v0 = pt
    d0 = depth / depth_factor
    # project
    P_w = inv_proj([u0, v0], d0, K0, pose0)
    P_w = Homo_3d(P_w)
    P0_C1 = np.matmul(pose1.I, P_w)
    p1 = cam2img(P0_C1, K1)

    return p1, pt, changed

def achieve_depth_search_around(pt, depth_map, direction):
    """ search around pt to get a valid depth
        as pt is always the vertices of area
        given direction, search along the direction
    Args:
        direction: [dx, dy]
            dx, dy = 1, -1
    """
    u, v = pt
    u, v = int(u), int(v)
    changed = False
    dx, dy = direction
    W, H = depth_map.shape[1], depth_map.shape[0]

    # tune pt to within image
    if u == W: u-=1
    if v == H: v-=1
    
    if depth_map[v, u] > 0:
        return depth_map[v,u], pt, changed

    u_search_bound = 0 if dx == -1 else W
    v_search_bound = 0 if dy == -1 else H

    u_search_start = u
    v_search_start = v

    while u_search_start != u_search_bound and v_search_start != v_search_bound:
        u_search_start += dx
        v_search_start += dy

        if depth_map[v_search_start, u_search_start] > 0:
            pt = [u_search_start, v_search_start]
            changed = True
            return depth_map[v_search_start, u_search_start], pt, changed
    
    return None, None, changed

    
