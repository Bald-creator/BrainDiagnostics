#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.distributed as dist

class NativeScalerWithGradNormCount:
    """梯度缩放器，用于混合精度训练"""
    
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
        self.state_dict_key = "amp_scaler"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        """
        参数:
            loss: 损失值
            optimizer: 优化器
            clip_grad: 梯度裁剪值
            parameters: 需要梯度裁剪的参数
            create_graph: 是否创建计算图
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # 解除scale，使得能够执行梯度裁剪
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            norm = None
        
        self._scaler.step(optimizer)
        self._scaler.update()
        
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_model_without_ddp(model):
    """获取没有分布式训练包装的模型"""
    
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def is_dist_avail_and_initialized():
    """检查分布式环境是否可用并初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """获取分布式训练的世界大小"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """获取当前进程的排名"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    """检查是否是主进程"""
    return get_rank() == 0

def setup_for_distributed(is_master):
    """为分布式环境设置打印功能"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print 