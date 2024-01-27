'''
Author: Eason
Date: 2022-07-08 15:12:16
LastEditTime: 2024-01-17 20:33:35
LastEditors: EasonZhang
Description: visulization utils
FilePath: /SGAM/utils/vis.py
'''


import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

import os
from loguru import logger
import collections

from .img_process import img_to_color

def plot_img_with_text(img, text, mode='lr'):
    """
    """

def get_n_colors(n):
    """
    """
    label_color_dict = {}

    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, n)))

    for i in range(n):
        c = cmap(i)
        label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
    
    return label_color_dict

def img_to_color(img):
    """
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def draw_matched_area(img0, img1, area0, area1, color, out_path, name0, name1, save=True):
    """
    """
    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:
        img1 = img_to_color(img1)
    
    W, H = img0.shape[1], img0.shape[0]
    
    out = stack_img(img0, img1)

    draw_matched_area_in_img(out, area0, area1, color)

    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_area.png"), out)
        logger.info(f"save matched area to {os.path.join(out_path, f'{name0}_{name1}_matched_area.png')}")
    
    return out

def draw_matched_area_with_mkpts(img0, img1, area0, area1, mkpts0, mkpts1, color, out_path, name0, name1, save=True):
    """
    """
    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:
        img1 = img_to_color(img1)
    
    W, H = img0.shape[1], img0.shape[0]
    
    out = stack_img(img0, img1)

    out = draw_matched_area_in_img(out, area0, area1, color)

    out = draw_mkpts_in_img(out, mkpts0, mkpts1, color)

    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_area_kpts.png"), out)
        logger.info(f"save matched area to {os.path.join(out_path, f'{name0}_{name1}_matched_area_kpts.png')}")
    
    return out

def draw_mkpts_in_img(out, mkpts0, mkpts1, color):
    """
    Args:
        out: img pair stacked with same W
        mkpts0: [N, 2]
        mkpts1: [N, 2], np.array
    """
    W = out.shape[1] // 2
    W = int(W)

    for i in range(mkpts0.shape[0]):
        cv2.circle(out, (int(mkpts0[i][0]), int(mkpts0[i][1])), 2, color, -1)
        cv2.circle(out, (int(mkpts1[i][0])+W, int(mkpts1[i][1])), 2, color, -1)
        cv2.line(out, (int(mkpts0[i][0]), int(mkpts0[i][1])), (int(mkpts1[i][0])+W, int(mkpts1[i][1])), color=color, thickness=1, lineType=cv2.LINE_AA)

    return out

def draw_matched_area_in_img(out, patch0, patch1, color):
    """
    """
    W = out.shape[1] // 2
    W = int(W)
    patch0 = [int(i) for i in patch0]
    patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]
    try:
        patch1_s = [int(i) for i in patch1_s]
    except ValueError as e:
        logger.exception(e)
        return False


    # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
    # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

    cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(color), 3)
    try:
        cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), color, 3)
    except cv2.error:
        logger.exception("what?")
        return False

    line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
    line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

    cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=color, thickness=3, lineType=cv2.LINE_AA)

    return True

def stack_img(img0, img1):
    """ stack two image in horizontal
    Args:
        img0: numpy array 3 channel
    """
    # assert img0.shape == img1.shape

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:    
        img1 = img_to_color(img1)

    assert len(img0.shape) == 3

    W0, H0 = img0.shape[1], img0.shape[0]
    W1, H1 = img1.shape[1], img1.shape[0]

    H_s = max(H0, H1)
    W_s = W0 + W1

    out = 255 * np.ones((H_s, W_s, 3), np.uint8)

    try:
        out[:H0, :W0, :] = img0.copy()
        out[:H1, W0:, :] = img1.copy()
    except ValueError as e:
        logger.exception(e)
        logger.info(f"img0 shape is {img0.shape}, img1 shape is {img1.shape}")
        logger.info(f"out shape is {out.shape}")
        raise e

    return out

def draw_matched_area_list(img0, img1, area0_list, area1_list, W, H, out_path, name0, name1, save=True):
    """
    """
    n = len(area0_list)
    assert n == len(area1_list)

    color_map = get_n_colors(n)

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:  
        img1 = img_to_color(img1)

    W, H = img0.shape[1], img0.shape[0]
    out = stack_img(img0, img1)

    flag = True
    for i in range(n):
        color = color_map[i]
        flag = draw_matched_area_in_img(out, area0_list[i], area1_list[i], color)
    
    if save:
        cv2.imwrite(os.path.join(out_path, f"/{name0}_{name1}_matched_areas.png"), out)
    
    return flag

def plot_matches_with_label(image0, image1, kpts0, kpts1, assert_label, layout="lr"):
    """
    plot matches between two images. 
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param assert label: 0 is flase, 1 is true, -1 is cannot assert ==> red is bad, green is good, blue is not asserted
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    for (x0, y0), (x1, y1), m in zip(kpts0, kpts1, assert_label):
        if layout == "lr":
            if m == 0:
                c = [0,0,255] # BGR
            elif m == 1:
                c = [0, 255, 0]
            elif m == -1:
                c = [255, 0, 0]
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            if m == 0:
                c = [0,0,255]
            elif m == 1:
                c = [0, 255, 0]
            elif m == -1:
                c = [255, 0, 0]
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out

# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    image = img_to_color(image)
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        scores = np.array(scores)
        # get color
        smin, smax = scores.min(), scores.max()
        scores = (scores - smin) / smax
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image

# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        scores = np.array(scores)
        smin, smax = scores.min(), scores.max()
        scores = (scores - smin) / smax
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out

def plot_matches_lists_lr(image0, image1, matches, outPath, name):
    """
    Args:
        matches: [u0, v0, u1,v1]s
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) + W0
        v1 = int(v1)
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

def plot_matches_lists_ud(image0, image1, matches, outPath, name):
    """
    Args:
        matches: [u0, v0, u1,v1]s
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

def plot_matches_with_mask_ud(image0, image1, mask, gt_pts, bad_ratio, matches, outPath, name, sample_num=500):
    """
    Args:
        mask: 0 -> false match
    """
    # random sample
    if len(matches) > sample_num:
        matches = random.sample(matches, sample_num)

    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    # l = len(mask)
    # font = cv2.FONT_HERSHEY_PLAIN
    # cv2.putText(out, f"bad match ratio: {bad_ratio: .2f}%/{l: d}", (50,20), font, 1, (255, 255, 255), 1)

    # gt_pt_c = [255, 0, 0]

    for i, match in enumerate(matches):
        if mask[i] == 0: c = [0, 0, 255]
        if mask[i] == 1: c = [0, 255, 0]
        if mask[i] == -1: continue

        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

        # if mask[i] == 0:
        #     u1_gt, v1_gt = gt_p1[i]
        #     u1_gt = int(u1_gt)
        #     v1_gt = int(v1_gt)

            # cv2.line(out, (u0, v0), (u1_gt, v1_gt+H1), color=gt_pt_c, thickness=1, lineType=cv2.LINE_AA)
            # cv2.circle(out, (u0, v0), 2, gt_pt_c, -1, lineType=cv2.LINE_AA)
            # cv2.circle(out, (u1_gt, v1_gt+H1), 2, gt_pt_c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

    return out

def plot_matches_with_mask_lr(image0, image1, mask, gt_pts, bad_ratio, matches, outPath, name, sample_num=500):
    """
    Args:
        mask: 0 -> false match
    """
    # random sample
    if len(matches) > sample_num:
        matches = random.sample(matches, sample_num)

    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1
    
    # l = len(mask)
    # font = cv2.FONT_HERSHEY_PLAIN
    # cv2.putText(out, f"bad match ratio: {bad_ratio: .2f}%/{l: d}", (50,20), font, 1, (255, 255, 255), 1)

    # gt_pt_c = [255, 0, 0]

    for i, match in enumerate(matches):
        if mask[i] == 0: c = [0, 0, 255]
        if mask[i] == 1: c = [0, 255, 0]
        if mask[i] == -1: continue

        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) + W0
        v1 = int(v1)
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)


    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

    return out

def paint_semantic_single(sem, outpath, name):
    """
    """
    assert len(sem.shape) == 2
    H, W = sem.shape

    temp_sem_list = np.squeeze(sem.reshape((-1,1))).tolist()
    temp_stas_dict = collections.Counter(temp_sem_list)

    label_list = list(temp_stas_dict.keys())
    label_color_dict = {}

    label_list = sorted(label_list)

    # print(label_list)

    N = len(label_list)
    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

    for i in range(N):
        c = cmap(i)
        label_color_dict[label_list[i]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]

    outImg = np.zeros((H,W,3))

    for label in label_list:
        outImg[np.where(sem==label)] = label_color_dict[label]
    
    cv2.imwrite(os.path.join(outpath, f"{name}.jpg"), outImg)
    path = os.path.join(outpath, f"{name}.jpg")

    logger.info(f"sem img written in {path}")

def paint_semantic(ins0, ins1, out_path, name0, name1):
    """ fill color by sematic label
    """
    assert len(ins0.shape) == len(ins1.shape) == 2

    H, W = ins0.shape

    label_list = []
    label_color_dict = {}

    for i in range(H):
        for j in range(W):
            temp0 = ins0[i, j]
            temp1 = ins1[i, j]
            if temp0 not in label_list:
                label_list.append(temp0)
            if temp1 not in label_list:
                label_list.append(temp1)

    label_list = sorted(label_list)

    # print(label_list)

    N = len(label_list)
    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

    for i in range(N):
        c = cmap(i)
        label_color_dict[label_list[i]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]

    outImg0 = np.zeros((H,W,3))
    # print(outImg0.shape)

    for i in range(H):
        for j in range(W):
            outImg0[i, j, :] = label_color_dict[ins0[i,j]]

    outImg1 = np.zeros((H,W,3))
    # print(outImg0.shape)

    for i in range(H):
        for j in range(W):
            outImg1[i, j, :] = label_color_dict[ins1[i,j]]

    cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name0)), outImg0)
    cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name1)), outImg1)

    return outImg0, outImg1

def draw_MMA(thds, ratios_list, names, outPath, MMA_postfix, label_flag=True):
    """
    Args:
        thds: small -> big
        ratios_list: [[ratios0]]
    """
    logger.info(f"label flag is {label_flag}")
    # from scipy.interpolate import spline
    matplotlib.use('Agg')
    
    name = "jet"
    cmap = cm.get_cmap(name, len(ratios_list))
    colors = cmap(np.linspace(0, 1, len(ratios_list)))
    linestyles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '-', '-', '-']
    l_c = min(len(colors), len(linestyles))
    plt.figure(figsize=(7,5))
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    ratio_num = len(ratios_list)
    assert l_c >= ratio_num, "out of color or linestyle"
    l = len(thds)
    thd_x = np.arange(thds[0], thds[-1]+thds[1] - thds[0], thds[1] - thds[0])
    for i, ratios in enumerate(ratios_list):
        assert len(ratios) == l, f"{i} strange ratios: {ratios}"
        # xnew = np.linspace(min(thds),max(thds),100) #300 represents number of points to make between T.min and T.max
        # power_smooth = spline(thds, ratios, xnew)
        # plt.plot(xnew, power_smooth, color=colors[i], ls=linestyles[i], linewidth=2, label=names[i])
        if label_flag:
            plt.plot(thds, ratios, color=colors[i], ls=linestyles[i], linewidth=2, label=names[i])
        else:
            plt.plot(thds, ratios, color=colors[i], ls=linestyles[i], linewidth=2)

    plt.xlim([thds[0], thds[-1]])
    plt.xticks(thds)
    plt.xlabel("threds")
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.savefig(os.path.join(outPath, "MMA_"+ MMA_postfix +".png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_heatmap_from_array(ndarray):
    """
    Args:   
        ndarray: [W x H]
    """
    ndarray = 1 - ndarray
    norm_img = np.zeros(ndarray.shape)
    norm_img = cv2.normalize(ndarray , None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)

    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) 
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

    return heat_img

def plot_scatters(x_data, y_data, x_name, y_name, out_path, out_name):
    """
    Args:
        x/y_data: []
    """
    assert len(x_data) == len(y_data), f"{len(x_data) != {len(y_data)}}"
    logger.info(f"plot data with len = {len(x_data)}")

    data = {
        x_name: x_data,
        y_name: y_data
    }

    data_pd = pd.DataFrame(data=data)

    out_path_ = os.path.join(out_path, out_name+".png")

    fig = sns.lmplot(x=x_name, y=y_name, data=data_pd)
    fig_save = fig.fig
    fig_save.savefig(out_path_, dpi=200)
    plt.close()
    
    logger.info(f"fig saved in {out_path_}")