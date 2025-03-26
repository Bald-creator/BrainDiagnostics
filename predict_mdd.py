#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.vision_transformer import VisionTransformer
from downstream_tasks.utils.load_mdd import BrainROIDataset, get_dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('MDD Classification Prediction', add_help=False)
    
    # 基本配置
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model checkpoint')
    
    # 数据配置
    parser.add_argument('--data_dir', default='/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF', type=str, help='Path to dataset')
    parser.add_argument('--output_dir', default='./output_dirs/mdd_prediction', type=str, help='Output directory')
    parser.add_argument('--rows', default=90, type=int, help='Number of rows in input data')
    parser.add_argument('--cols', default=116, type=int, help='Number of columns in input data')
    parser.add_argument('--test_size', default=0.2, type=float, help='Test set size for evaluation')
    parser.add_argument('--val_size', default=0.2, type=float, help='Validation set size')
    
    # 设备配置
    parser.add_argument('--device', default='cuda:0', help='Device to use for evaluation')
    
    # 预测模式
    parser.add_argument('--mode', default='test', choices=['test', 'val', 'all'], help='Evaluation mode (test, val, or all)')
    
    return parser

def load_model(args, device):
    print(f"Loading model from {args.model_path}")
    
    # 加载模型检查点
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 获取模型参数
    model_args = checkpoint.get('args', {})
    
    # 创建模型
    model = VisionTransformer(
        img_size=[args.rows, args.cols],
        patch_size=getattr(model_args, 'patch_size', 8),
        in_chans=1,  # 使用单通道输入 (ROI时间序列)
        num_classes=2,  # 二分类任务 (MDD vs Control)
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        drop_path_rate=getattr(model_args, 'drop_path', 0.0),
        attn_mode=getattr(model_args, 'attn_mode', 'normal')
    )
    
    # 加载模型权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def evaluate(model, data_loader, device):
    """对给定数据加载器的数据进行评估"""
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                outputs = model(samples)
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # 取第1类（MDD）的概率
    
    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    report = classification_report(all_targets, all_preds, target_names=['Control', 'MDD'], output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    # 组织结果
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    return results

def main(args):
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        rows=args.rows,
        cols=args.cols,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=42  # 使用固定的随机种子确保数据分割一致
    )
    
    # 加载模型
    model = load_model(args, device)
    
    # 进行预测
    results = {}
    
    if args.mode == 'test' or args.mode == 'all':
        print("Evaluating on test set...")
        test_results = evaluate(model, test_loader, device)
        results['test'] = test_results
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test AUC: {test_results['auc']:.4f}")
        print("Classification Report:")
        print(json.dumps(test_results['classification_report'], indent=2))
        print("Confusion Matrix:")
        print(np.array(test_results['confusion_matrix']))
    
    if args.mode == 'val' or args.mode == 'all':
        print("Evaluating on validation set...")
        val_results = evaluate(model, val_loader, device)
        results['val'] = val_results
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"Validation AUC: {val_results['auc']:.4f}")
    
    if args.mode == 'all':
        print("Evaluating on training set...")
        train_results = evaluate(model, train_loader, device)
        results['train'] = train_results
        print(f"Training Accuracy: {train_results['accuracy']:.4f}")
        print(f"Training AUC: {train_results['auc']:.4f}")
    
    # 保存结果
    results_file = os.path.join(args.output_dir, f'results_{args.mode}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    # 打印主要参数
    print(f"=== Prediction Configuration ===")
    print(f"Model Path: {args.model_path}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Evaluation Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"===============================")
    
    # 开始预测
    main(args) 