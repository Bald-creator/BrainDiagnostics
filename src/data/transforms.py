#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
import numpy as np

class DataAugmentation(object):
    """
    MDD数据增强类，适用于ROI时间序列数据
    """

    def __init__(
        self,
        normalize=True,
        mean=0.0,
        std=1.0,
    ):
        """
        参数:
            normalize: 是否进行归一化
            mean: 归一化平均值
            std: 归一化标准差
        """
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        应用数据增强
        
        参数:
            x: 输入数据 [C, H, W]
        
        返回:
            增强后的数据
        """
        # 如果输入是Numpy数组，转换为Tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # 应用归一化（如果启用）
        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-10)
        
        return x 