#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class MaskingGenerator:
    """
    生成随机遮蔽图案的生成器，用于自监督学习
    """
    
    def __init__(
        self,
        input_size,
        mask_ratio=0.6,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        """
        参数:
            input_size: 输入图像大小，应为(H, W)格式
            mask_ratio: 要遮蔽的区域比例
            min_num_patches: 最小遮蔽块数量
            max_num_patches: 最大遮蔽块数量
            min_aspect: 遮蔽块最小长宽比
            max_aspect: 遮蔽块最大长宽比
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
            
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.mask_ratio = mask_ratio
        
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        
        if min_aspect is not None and min_aspect < 1:
            min_aspect = 1. / min_aspect
        
        self.log_aspect_ratio = (
            np.log(min_aspect), np.log(1. / min_aspect) if max_aspect is None else np.log(max_aspect)
        )
        
    def _mask_block(self, mask, height_start, width_start, height_end, width_end):
        """
        在给定的区域内应用遮蔽
        """
        mask[height_start:height_end, width_start:width_end] = 1
        return mask
        
    def _random_block(self, mask_in=None):
        """
        生成随机遮蔽块的位置
        """
        if mask_in is None:
            mask = np.zeros((self.height, self.width), dtype=np.int)
        else:
            mask = mask_in.copy()
            
        target_area = self.mask_ratio * self.num_patches
        mask_sum = mask.sum()
        
        if mask_sum >= target_area:
            return mask
        
        # 选择遮蔽块的大小和位置
        remaining = target_area - mask_sum
        log_ratio = self.log_aspect_ratio[0] + np.random.rand() * (self.log_aspect_ratio[1] - self.log_aspect_ratio[0])
        aspect_ratio = np.exp(log_ratio)
        
        block_height = int(np.round(np.sqrt(remaining * aspect_ratio)))
        block_width = int(np.round(np.sqrt(remaining / aspect_ratio)))
        
        # 确保不超出范围
        block_height = min(block_height, self.height)
        block_width = min(block_width, self.width)
        
        # 随机选择起始位置
        height_start = np.random.randint(0, self.height - block_height + 1)
        width_start = np.random.randint(0, self.width - block_width + 1)
        
        self._mask_block(
            mask, height_start, width_start, height_start + block_height, width_start + block_width)
            
        return mask
        
    def __call__(self):
        """
        生成随机遮蔽图案
        
        返回:
            遮蔽图案, 形状为(H, W)的二值矩阵, 1表示遮蔽, 0表示可见
        """
        mask = None
        max_patches = self.max_num_patches or self.num_patches
        num_patches = min(self.min_num_patches + np.random.randint(max_patches - self.min_num_patches + 1), self.num_patches)
        
        # 生成多个遮蔽块
        for _ in range(num_patches):
            mask = self._random_block(mask)
            
        return mask 