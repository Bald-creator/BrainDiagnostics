#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

def cosine_scheduler(base_value, final_value, epochs, niter_per_epoch, 
                     warmup_epochs=0, start_warmup_value=0):
    """
    创建余弦学习率调度计划。
    
    参数:
        base_value: 起始学习率
        final_value: 最终学习率
        epochs: 总训练轮数
        niter_per_epoch: 每轮迭代次数
        warmup_epochs: 预热轮数
        start_warmup_value: 预热开始值
    
    返回:
        包含所有迭代的学习率列表
    """
    total_iters = epochs * niter_per_epoch
    
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * niter_per_epoch
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    remaining_iters = total_iters - len(warmup_schedule)
    if remaining_iters <= 0:
        # 如果预热迭代次数等于或大于总迭代次数，则直接返回预热调度表
        return warmup_schedule[:total_iters]
    
    # 生成余弦退火调度表
    iters = np.arange(remaining_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / remaining_iters))
    
    # 合并预热和余弦退火调度表
    schedule = np.concatenate((warmup_schedule, schedule))
    
    # 确保调度表长度与预期一致
    if len(schedule) != total_iters:
        # 如果长度不一致，通过截断或填充使其一致
        if len(schedule) > total_iters:
            schedule = schedule[:total_iters]
        else:
            # 如果太短，使用最后一个值填充
            padding = np.ones(total_iters - len(schedule)) * schedule[-1]
            schedule = np.concatenate((schedule, padding))
    
    assert len(schedule) == epochs * niter_per_epoch, "Schedule length does not match expected length"
    
    return schedule 