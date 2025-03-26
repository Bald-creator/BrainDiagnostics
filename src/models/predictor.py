#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Predictor(nn.Module):
    """
    用于MDD分类的预测器
    """
    
    def __init__(
        self,
        embed_dim,
        pred_depth,
        pred_emb_dim=None,
        num_heads=None,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop_path=0.0,
        num_classes=2,
    ):
        """
        参数:
            embed_dim: 输入特征维度
            pred_depth: 预测器深度
            pred_emb_dim: 预测器嵌入维度(如果为None则使用embed_dim)
            num_heads: 注意力头数量(如果为None则使用embed_dim // 64)
            mlp_ratio: MLP层的隐藏层倍率
            act_layer: 激活函数
            drop_path: 路径丢弃率
            num_classes: 分类类别数量
        """
        super().__init__()
        
        # 如果未指定预测器嵌入维度，使用输入嵌入维度
        pred_emb_dim = pred_emb_dim or embed_dim
        
        # 如果未指定头数量，使用预测器嵌入维度除以64
        num_heads = num_heads or pred_emb_dim // 64
        
        # 投影层: 将输入特征投影到预测器嵌入维度
        if embed_dim != pred_emb_dim:
            self.proj = nn.Linear(embed_dim, pred_emb_dim)
        else:
            self.proj = nn.Identity()
        
        # 创建包含多个TransformerBlock的预测器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=pred_emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop_path=drop_path,
            )
            for _ in range(pred_depth)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(pred_emb_dim)
        
        # 分类头，用于MDD分类任务
        self.head = nn.Linear(pred_emb_dim, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, N, C]
            
        返回:
            预测结果 [B, num_classes]
        """
        # 投影
        x = self.proj(x)
        
        # 应用TransformerBlock
        for block in self.blocks:
            x = block(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # [B, C]
        
        # 归一化
        x = self.norm(x)
        
        # 分类输出
        x = self.head(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    标准Transformer块
    """
    
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
    ):
        """
        参数:
            dim: 特征维度
            num_heads: 注意力头数量
            mlp_ratio: MLP隐藏层大小相对于输入维度的倍数
            qkv_bias: 是否使用qkv偏置
            act_layer: 激活函数
            norm_layer: 归一化层
            drop_path: 路径丢弃率
        """
        super().__init__()
        
        # 第一个层归一化
        self.norm1 = norm_layer(dim)
        
        # 多头注意力
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        
        # 路径丢弃
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 第二个层归一化
        self.norm2 = norm_layer(dim)
        
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, N, C]
            
        返回:
            输出特征 [B, N, C]
        """
        # 残差连接和注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 残差连接和MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class Attention(nn.Module):
    """
    多头自注意力
    """
    
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """
        参数:
            dim: 输入维度
            num_heads: 注意力头数量
            qkv_bias: 是否使用qkv偏置
            qk_scale: 缩放因子(如果为None则使用头维度的平方根)
            attn_drop: 注意力丢弃率
            proj_drop: 输出投影丢弃率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 注意力丢弃
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        
        # 输出丢弃
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, N, C]
            
        返回:
            输出特征 [B, N, C]
        """
        B, N, C = x.shape
        
        # QKV投影并分头
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力并合并头
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Mlp(nn.Module):
    """
    多层感知机
    """
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度(如果为None则使用in_features)
            out_features: 输出特征维度(如果为None则使用in_features)
            act_layer: 激活函数
            drop: 丢弃率
        """
        super().__init__()
        
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        # 第一个线性层
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 激活函数
        self.act = act_layer()
        
        # 第二个线性层
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # 丢弃层
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征
            
        返回:
            输出特征
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

def DropPath(drop_prob: float = 0.0):
    """
    路径丢弃函数
    
    参数:
        drop_prob: 丢弃概率
        
    返回:
        丢弃层或Identity层
    """
    if drop_prob == 0.0:
        return nn.Identity()
    else:
        return _DropPath(drop_prob)

class _DropPath(nn.Module):
    """
    每个样本随机丢弃路径
    
    参数:
        drop_prob: 丢弃概率
    """
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值掩码
        output = x.div(keep_prob) * random_tensor
        
        return output 