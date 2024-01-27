'''
Author: Eason
Date: 2022-07-08 15:05:44
LastEditTime: 2024-01-17 20:31:30
LastEditors: EasonZhang
Description: utils for image processing
FilePath: /SGAM/utils/img_process.py
'''

from loguru import logger
import numpy as np
import cv2

def img_crop(ori_img, crop_list):
    """
        -------> x(u)
        |
        |  img
        |
        v
        y(v)
    Args:
        crop_list: [x_l, x_r, y_u, y_d] == [u_min, u_max, v_min, v_max]
    """
    crop_list_ = [int(x) for x in crop_list]
    u_min, u_max, v_min, v_max = crop_list_
    crop_img = ori_img[v_min:v_max, u_min:u_max] # NOTE the inverse operation
    return crop_img

def img_crop_with_resize(ori_img, crop_list, size):
    """ crop & resize
    Args:
        ori_img: cv2.img
        size = [W, H]
    Returns:
        crop_resized: 
        scales: [W_ori/W_re, H_ori/H_re] -- (u, v)_resized * scale = (u, v)_ori
        offsets: [u_offset, v_offset]
    """
    W_resized, H_resized = size

    crop_list_ = [int(x) for x in crop_list]
    x_l, x_r, y_u, y_d = crop_list_
    offsets = [x_l, y_u]
    crop_img = ori_img[y_u:y_d, x_l:x_r] # NOTE the inverse operation
    H_ori, W_ori = crop_img.shape[0], crop_img.shape[1]
    scales = [W_ori/W_resized, H_ori/H_resized]
    crop_resized = cv2.resize(crop_img, tuple(size))

    return crop_resized, scales, offsets
    
def img_crop_without_Diffscale(ori_img, area, size):
    """ crop a square size 
    Args:
        size=W=H: int:the crop size
    Returns:
        crop_resized: 
        scales: [W_ori/W_re, H_ori/H_re] -- (u, v)_resized * scale = (u, v)_ori
        offset: [u_offset, v_offset]
    """
    H, W = ori_img.shape[0], ori_img.shape[1]
    u_min, u_max, v_min, v_max = area
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    u_radius = u_max - u_min
    v_radius = v_max - v_min

    max_len = max(u_radius, v_radius)
    max_len = max(size, max_len)
    max_radius = max_len / 2
    
    # tune center
    if (u_center - max_radius) < 0 and (u_center + max_radius) >= W:
        u_min_f = 0
        u_max_f = W
    elif (u_center - max_radius) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + max_len, W)
    elif (u_center + max_radius) >= W:
        u_max_f = W
        u_min_f = max(0, u_max_f - max_len) 
    else:
        u_min_f = u_center - max_radius
        u_max_f = u_center + max_radius
    
    if (v_center - max_radius) < 0 and (v_center + max_radius) >= H:
        v_min_f = 0
        v_max_f = H
    elif (v_center - max_radius) < 0:
        v_min_f = 0
        v_max_f = min(H, v_min_f+max_len)
    elif (v_center + max_radius) >= H:
        v_max_f = H
        v_min_f = max(0, v_max_f - max_len)
    else:
        v_min_f = v_center - max_radius
        v_max_f = v_center + max_radius
    
    square_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    offset = [square_area[0], square_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, square_area, [size, size])
    
    return crop, scale, offset

def img_crop_fix_aspect_ratio(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ crop area from ori image
        spread area toward same aspect ratio of crop size
    """
    ori_W, ori_H = ori_img.shape[1], ori_img.shape[0]

    u_min, u_max, v_min, v_max = area
    aspect_ratio = crop_W / crop_H
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    # fix the longer side, spread the shorter side
    if (u_max - u_min) / (v_max - v_min) > aspect_ratio:
        W_ori_len = (u_max - u_min)*spread_ratio
        H_ori_len = W_ori_len / aspect_ratio
    else:
        H_ori_len = (v_max - v_min)*spread_ratio
        W_ori_len = H_ori_len * aspect_ratio
    
    # tune the center, ensure the crop area is in the image
    if (u_center - W_ori_len/2) < 0 and (u_center + W_ori_len/2) >= ori_W:
        u_min_f = 0
        u_max_f = ori_W
    elif (u_center - W_ori_len/2) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + W_ori_len, ori_W)
    elif (u_center + W_ori_len/2) >= ori_W:
        u_max_f = ori_W
        u_min_f = max(0, u_max_f - W_ori_len)
    else:
        u_min_f = u_center - W_ori_len/2
        u_max_f = u_center + W_ori_len/2
    
    if (v_center - H_ori_len/2) < 0 and (v_center + H_ori_len/2) >= ori_H:
        v_min_f = 0
        v_max_f = ori_H
    elif (v_center - H_ori_len/2) < 0:
        v_min_f = 0
        v_max_f = min(v_min_f + H_ori_len, ori_H)
    elif (v_center + H_ori_len/2) >= ori_H:
        v_max_f = ori_H
        v_min_f = max(0, v_max_f - H_ori_len)
    else:
        v_min_f = v_center - H_ori_len/2
        v_max_f = v_center + H_ori_len/2

    crop_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    offset = [crop_area[0], crop_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, crop_area, [crop_W, crop_H])

    return crop, scale, offset

def img_crop_direct(ori_img, area, crop_W, crop_H, spread_ratio=1.2, dfactor=8):
    """crop img with specific size
    Funcs:
        small -> spread to crop size
        big -> resize to crop size
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    logger.info(f"crop size {crop_W}x{crop_H} of area {area} from img size {ori_W}x{ori_H}")

    u_min, u_max, v_min, v_max = area
    
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    W_ori_len = (u_max - u_min)*spread_ratio
    H_ori_len = (v_max - v_min)*spread_ratio

    max_W_len = max(W_ori_len, crop_W)
    max_W_radius = max_W_len / 2
    max_H_len = max(H_ori_len, crop_H)
    max_H_radius = max_H_len / 2

    # tune center
    if (u_center - max_W_radius) < 0 and (u_center + max_W_radius) >= ori_W:
        u_min_f = 0
        u_max_f = ori_W
    elif (u_center - max_W_radius) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + max_W_len, ori_W)
    elif (u_center + max_W_radius) >= ori_W:
        u_max_f = ori_W
        u_min_f = max(0, u_max_f - max_W_len) 
    else:
        u_min_f = u_center - max_W_radius
        u_max_f = u_center + max_W_radius
    # H
    if (v_center - max_H_radius) < 0 and (v_center + max_H_radius) >= ori_H:
        v_min_f = 0
        v_max_f = ori_H
    elif (v_center - max_H_radius) < 0:
        v_min_f = 0
        v_max_f = min(ori_H, v_min_f+max_H_len)
    elif (v_center + max_H_radius) >= ori_H:
        v_max_f = ori_H
        v_min_f = max(0, v_max_f - max_H_len)
    else:
        v_min_f = v_center - max_H_radius
        v_max_f = v_center + max_H_radius

    crop_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    logger.info(f"acctually crop as {crop_area}")

    offset = [crop_area[0], crop_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, crop_area, [crop_W, crop_H])
    
    return crop, scale, offset

def img_to_color(img):
    """
    Args:
        img: np or cv2
    """
    if len(img.shape) == 2 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img

def patch_adjust_with_square_min_limit(crop_list, W, H, min_size):
    """ achieve a square patch
    """
    u_min, u_max, v_min, v_max = crop_list
    u_len = u_max - u_min
    v_len = v_max - v_min

    final_radius = max(min_size, max(u_len, v_len) // 2)

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    u_min_ = max(0, center_u - final_radius)
    u_max_ = min(W, center_u + final_radius)
    v_min_ = max(0, center_v - final_radius)
    v_max_ = min(H, center_v + final_radius)

    return [u_min_, u_max_, v_min_, v_max_]

def patch_adjust_with_size_limits(crop_list, W, H, radius_thd):
    """adjust patch to be square
    """
    u_min, u_max, v_min, v_max = crop_list
    size_max, size_min = radius_thd

    radius_max = max((u_max - u_min)/2, (v_max - v_min)/2)
    radius_max = max(radius_max, size_min)
    radius_max = min(radius_max, size_max)

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    u_min_ = max(0, center_u - radius_max)
    u_max_ = min(W-1, center_u + radius_max)

    v_min_ = max(0, center_v - radius_max)
    v_max_ = min(H-1, center_v + radius_max)

    return [u_min_, u_max_, v_min_, v_max_]

def patch_adjust_fix_size(crop_list, W, H, fix_size=256):
    """ crop with fix size
    """
    u_min, u_max, v_min, v_max = crop_list

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    radius = fix_size // 2
    if center_u - radius < 0:
        center_u = radius
    if center_u + radius > W:
        center_u = W - radius
    
    if center_v - radius < 0:
        center_v = radius
    if center_v + radius > H:
        center_v = H - radius
    
    u_min_ = center_u - radius
    u_max_ = center_u + radius
    v_min_ = center_v - radius
    v_max_ = center_v + radius

    return [u_min_, u_max_, v_min_, v_max_]
    
    
    