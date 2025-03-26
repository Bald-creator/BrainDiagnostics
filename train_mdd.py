#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import math
import numpy as np
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma

import src.utils.logging as logging
import src.utils.lr_sched as lr_sched
from src.data.transforms import DataAugmentation
from src.data.masking_generator import MaskingGenerator
from src.models.vision_transformer import VisionTransformer
from src.models.predictor import Predictor
from src.utils.pos_embed import interpolate_pos_embed, interpolate_pos_embed_moco
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from src.utils.misc import get_model_without_ddp
from src.utils.tensors import trunc_normal_

from downstream_tasks.utils.load_mdd import get_dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('MDD Classification Training', add_help=False)
    
    # 基本配置
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for training')
    parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Frequency of checkpoint saving')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    # 模型配置
    parser.add_argument('--model', default='vit_base', type=str, help='Model name')
    parser.add_argument('--crop_size', default=[90, 116], type=int, nargs='+', help='Crop size for input data')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch size')
    parser.add_argument('--drop_path', default=0.0, type=float, help='Drop path rate')
    parser.add_argument('--attn_mode', default='normal', type=str, help='Attention mode (normal or flash_attn)')
    
    # 优化配置
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs')
    
    # 数据配置
    parser.add_argument('--data_dir', default='/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF', type=str, help='Path to dataset')
    parser.add_argument('--output_dir', default='./output_dirs/mdd_classification', type=str, help='Output directory')
    parser.add_argument('--rows', default=90, type=int, help='Number of rows in input data')
    parser.add_argument('--cols', default=116, type=int, help='Number of columns in input data')
    parser.add_argument('--test_size', default=0.2, type=float, help='Test set size')
    parser.add_argument('--val_size', default=0.2, type=float, help='Validation set size')
    
    # 设备配置
    parser.add_argument('--device', default='cuda:0', help='Device to use for training')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    
    # 配置文件
    parser.add_argument('--config_file', default='./configs/mdd_classification.yaml', type=str, help='Configuration file path')
    return parser

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 设置日志
    log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 加载数据
    print(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        rows=args.rows,
        cols=args.cols,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=args.seed
    )
    print(f"Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples, {len(test_loader.dataset)} test samples")
    
    # 创建模型
    print(f"Creating model: {args.model}")
    model = VisionTransformer(
        img_size=args.crop_size,
        patch_size=args.patch_size,
        in_chans=1,  # 使用单通道输入 (ROI时间序列)
        num_classes=2,  # 二分类任务 (MDD vs Control)
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        drop_path_rate=args.drop_path,
        attn_mode=args.attn_mode,
        gradient_pos_embed=None  # 不使用梯度位置编码
    )
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    param_groups = [
        {'params': model.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    lr_schedule = lr_sched.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs
    )
    
    # 损失缩放器
    loss_scaler = NativeScaler()
    
    # 恢复检查点
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    
    print(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, lr_schedule
        )
        
        # 在验证集上评估
        val_stats = validate(model, criterion, val_loader, device)
        
        # 记录日志
        print(f"Epoch {epoch} | Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['acc1']:.4f} | Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['acc1']:.4f}")
        log_writer.add_scalar('train/loss', train_stats['loss'], epoch)
        log_writer.add_scalar('train/acc1', train_stats['acc1'], epoch)
        log_writer.add_scalar('val/loss', val_stats['loss'], epoch)
        log_writer.add_scalar('val/acc1', val_stats['acc1'], epoch)
        
        # 保存检查点
        if (epoch % args.save_ckpt_freq == 0) or (epoch == args.epochs - 1):
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # 在测试集上评估最终模型
    test_stats = validate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_stats['loss']:.4f}, Test Acc: {test_stats['acc1']:.4f}")
    log_writer.add_scalar('test/loss', test_stats['loss'], 0)
    log_writer.add_scalar('test/acc1', test_stats['acc1'], 0)
    
    # 关闭日志记录器
    log_writer.close()
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model': model.state_dict(),
        'args': args,
    }, final_path)
    print(f"Saved final model to {final_path}")

def train_one_epoch(model, criterion, data_loader, optimizer, 
                    device, epoch, loss_scaler, lr_schedule):
    model.train()
    metric_logger = AverageMeter()
    metric_logger_loss = AverageMeter()
    metric_logger_acc1 = AverageMeter()
    
    header = f'Epoch: [{epoch}]'
    
    for step, (samples, targets) in enumerate(data_loader):
        # 更新学习率
        it = epoch * len(data_loader) + step
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        acc1 = accuracy(outputs, targets)[0]
        
        # 更新梯度
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=is_second_order)
        
        # 记录指标
        metric_logger_loss.update(loss.item())
        metric_logger_acc1.update(acc1.item())
        
        if step % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Train: {header} [{step}/{len(data_loader)}]\t"
                  f"Loss: {loss.item():.4f}\t"
                  f"Acc@1: {acc1.item():.2f}%\t"
                  f"LR: {lr:.6f}")
    
    # 收集统计信息
    train_stats = {'loss': metric_logger_loss.avg, 'acc1': metric_logger_acc1.avg}
    return train_stats

@torch.no_grad()
def validate(model, criterion, data_loader, device):
    model.eval()
    
    metric_logger_loss = AverageMeter()
    metric_logger_acc1 = AverageMeter()
    
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        acc1 = accuracy(outputs, targets)[0]
        
        # 记录指标
        metric_logger_loss.update(loss.item())
        metric_logger_acc1.update(acc1.item())
    
    # 收集统计信息
    val_stats = {'loss': metric_logger_loss.avg, 'acc1': metric_logger_acc1.avg}
    return val_stats

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    # 加载配置文件
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
            # 将配置文件中的参数更新到命令行参数
            for k, v in config.items():
                if k in vars(args):
                    setattr(args, k, v)
    
    # 打印主要参数
    print(f"=== Training Configuration ===")
    print(f"Model: {args.model}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Attention Mode: {args.attn_mode}")
    print(f"=============================")
    
    # 开始训练
    main(args) 