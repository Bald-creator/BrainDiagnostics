#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, orig_size, new_size):
    """
    调整位置嵌入大小以匹配新的输入尺寸
    
    参数:
        pos_embed: 原位置嵌入 [B, N, C]
        orig_size: 原始图像尺寸 (H, W)
        new_size: 新图像尺寸 (H', W')
    
    返回:
        调整大小后的位置嵌入
    """
    if pos_embed.shape[1] == new_size[0] * new_size[1] + 1:
        return pos_embed
    
    # 类别标记和位置标记
    cls_pos_embed = pos_embed[:, 0:1, :]
    pos_embed = pos_embed[:, 1:, :]
    
    # 调整大小
    pos_embed = pos_embed.reshape(1, orig_size[0], orig_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=new_size, mode='bicubic', align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    
    # 合并类别标记
    new_pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
    
    return new_pos_embed

def interpolate_pos_embed_moco(pos_embed, orig_size, new_size):
    """
    用于MoCo风格位置嵌入的插值函数，处理模型迁移
    
    参数:
        pos_embed: 原位置嵌入
        orig_size: 原始图像尺寸 (H, W)
        new_size: 新图像尺寸 (H', W')
    
    返回:
        调整大小后的位置嵌入
    """
    npatch = orig_size[0] * orig_size[1]
    N = pos_embed.shape[1] - 1
    
    if npatch == N:
        return pos_embed
    
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    
    dim = pos_embed.shape[-1]
    
    # 调整patch位置嵌入
    w = orig_size[1]
    h = orig_size[0]
    patch_pos_embed = patch_pos_embed.reshape(1, h, w, dim).permute(0, 3, 1, 2)
    
    # 双线性插值
    new_h, new_w = new_size
    patch_pos_embed = F.interpolate(
        patch_pos_embed, size=(new_h, new_w), mode='bicubic', align_corners=False)
    
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
    
    # 拼接类别标记嵌入
    new_pos_embed = torch.cat([class_pos_embed.unsqueeze(0).unsqueeze(1), patch_pos_embed], dim=1)
    
    return new_pos_embed 