# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# --------------------------------------------------------

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, DropPath, PatchEmbed
from einops import rearrange, repeat

# 从src.utils.tensors导入需要的函数
try:
    from src.utils.tensors import (
        trunc_normal_,
        repeat_interleave_batch,
    )
except ImportError:
    # 如果无法导入，提供备用实现
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # 使用PyTorch内置的截断正态分布初始化
        return torch.nn.init.trunc_normal_(tensor, mean, std, a, b)
    
    def repeat_interleave_batch(tensor, batch_size):
        # 实现batch维度的重复
        return tensor.repeat(batch_size, *[1 for _ in range(tensor.dim() - 1)])


# 尝试导入flash_attn，如果不可用则继续
flash_attn_available = False
try:
    from flash_attn import flash_attn_qkvpacked_func
    flash_attn_available = True
except ImportError:
    # 如果flash_attn不可用，不导入对应函数
    pass


# CUDA兼容性检查
def is_flash_attn_available():
    """检查是否可以使用flash attention"""
    return flash_attn_available and torch.cuda.is_available()


# 实现apply_masks函数
def apply_masks(x, masks):
    """应用掩码到输入张量
    
    参数:
        x: 输入张量 [B, N, C]
        masks: 掩码张量 [B, N], 值为0或1，1表示保留，0表示遮盖
        
    返回:
        应用掩码后的张量
    """
    if masks is None:
        return x
    
    # 确保掩码是布尔类型
    if masks.dtype != torch.bool:
        masks = masks.bool()
    
    # 应用掩码
    B, N, C = x.shape
    x_masked = x.clone()
    
    # 将遮盖位置的值设为0
    for i in range(B):
        x_masked[i, ~masks[i]] = 0
    
    return x_masked


class GradTs_2dPE(nn.Module):
    def __init__(self, in_chan, embed_dim, grid_size, add_w=False, cls_token=False) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.grid = self.get_grid(grid_size)
        self.emb_h = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], embed_dim // 2), requires_grad=False)
        pos_emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[0])  # (H*W, D/2)
        self.emb_h.data.copy_(torch.from_numpy(pos_emb_h).float())
        
        self.add_w = add_w
        if add_w == 'origin':
            self.emb_w = nn.Parameter(torch.zeros(grid_size[0]*grid_size[1], embed_dim // 2), requires_grad=False)
            pos_emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, self.grid[1])  # (H*W, D/2)
            self.emb_w.data.copy_(torch.from_numpy(pos_emb_w).float())
            
        if add_w == 'mapping':
            self.predictor_pos_embed_proj = nn.Linear(in_chan, embed_dim//2)
        self.cls_token = cls_token

        
    def get_grid(self, grid_size):    
        grid_h = np.arange(grid_size[0], dtype=float)
        grid_w = np.arange(grid_size[1], dtype=float)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
        return grid
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   # (D/2,)

        pos = pos.reshape(-1)   # (M,)
        out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
    
    def forward(self, gradient):  # TODO: check which one is emb_w and which one is emb_h !!!!!
        if self.add_w == 'mapping':
            gradient_pos_embed = self.predictor_pos_embed_proj(gradient)
            # emb_w = torch.cat([gradient_pos_embed.squeeze()]*10, dim=0)  # (H*W, D/2)
            emb_w = gradient_pos_embed.squeeze().repeat_interleave(10, dim=0)
            emb_w = (emb_w - emb_w.min()) / (emb_w.max() - emb_w.min()) * 2 - 1
            
        if self.add_w == 'mapping':
            emb_w = emb_w
        elif self.add_w == 'origin':
            emb_w = self.emb_w
        else:
            raise Exception('self.add_w error')
        
            
        emb = torch.cat([self.emb_h, emb_w], dim=1).unsqueeze(0)  # (H*W, D)
        
        if self.cls_token:
            pos_embed = torch.concat([torch.zeros([1, 1, emb.shape[2]]).cuda(), emb], dim=1)
        else:
            pos_embed = emb
        return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 使用普通注意力机制的自注意力层
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_mode='normal'
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 注意力模式，可以是flash_attn或normal
        self.attn_mode = attn_mode

    def forward(self, x, return_attn=False):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, C/num_heads

        if self.attn_mode == 'flash_attn' and is_flash_attn_available():
            try:
                from flash_attn import flash_attn_qkvpacked_func
                with torch.cuda.amp.autocast(enabled=False):
                    # qkv: [B, N, 3, num_heads, C/num_heads]
                    qkv_packed = qkv.permute(1, 3, 0, 2, 4)  # [B, N, 3, num_heads, C/num_heads]
                    # Convert to half precision
                    qkv_packed = qkv_packed.half()
                    out = flash_attn_qkvpacked_func(qkv_packed, dropout_p=0.0 if not self.training else self.attn_drop.p)
                    x = out.view(B, N, C)
                    if return_attn:
                        # 计算普通注意力以返回注意力矩阵
                        attn = (q @ k.transpose(-2, -1)) * self.scale
                        attn = attn.softmax(dim=-1)
                        return x, attn
                    return x, None
            except (ImportError, Exception) as e:
                print(f"Warning: Flash Attention unavailable, error: {e}. Falling back to regular attention.")
                # 使用普通的注意力机制
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # 普通的注意力机制
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            if return_attn:
                return x, attn

        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, None
        return x, None


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_mode='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_mode=attn_mode)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0]) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_patches_2d = (img_size[0], img_size[1] // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        num_patches_2d,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode='normal',
        add_w=False,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # -- predictor_pos_embedding gradient
        self.gradient_pos_embed = gradient_pos_embed
        self.predictor_2dpe_proj = GradTs_2dPE(gradient_pos_embed.shape[-1], predictor_embed_dim, num_patches_2d, add_w=add_w, cls_token=False)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attn_mode=attn_mode)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, return_attention=False):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        predictor_pos_embed = self.predictor_2dpe_proj(self.gradient_pos_embed)
        
        x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)

        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        attn_set = []
        for blk in self.predictor_blocks:
            if return_attention:
                x, attn = blk(x, return_attention)
                attn_set.append(attn.detach().cpu())
            else:
                x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        if return_attention:
            return x, attn_set
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=(224,224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        gradient_pos_embed=None,
        attn_mode='normal',
        add_w=False,
        gradient_checkpointing=False,
        num_classes=0,  # 0表示没有分类头
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_2d = self.patch_embed.num_patches_2d
        
        # -- 处理位置编码
        if gradient_pos_embed is not None:
            # 使用梯度位置编码
            self.gradient_pos_embed = gradient_pos_embed
            self.pos_embed_proj = GradTs_2dPE(gradient_pos_embed.shape[-1], embed_dim, self.num_patches_2d, add_w=add_w, cls_token=False)
        else:
            # 使用标准的正弦余弦位置编码
            self.gradient_pos_embed = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            pos_embed = get_2d_sincos_pos_embed(embed_dim, (self.num_patches_2d[0], self.num_patches_2d[1]), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attn_mode=attn_mode)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # 添加分类头（如果需要）
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = None
            
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None, return_attention=False):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- 添加位置编码
        if self.gradient_pos_embed is not None:
            # 使用梯度位置编码
            pos_embed = self.pos_embed_proj(self.gradient_pos_embed)
            x = x + pos_embed
        else:
            # 使用标准位置编码
            x = x + self.pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        attn_set = []
        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                if return_attention:
                    x, attn = torch.utils.checkpoint.checkpoint(blk, x, return_attention, use_reentrant=False)
                    attn_set.append(attn.detach().cpu())
                else:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                if return_attention:
                    x, attn = blk(x, return_attention)
                    attn_set.append(attn.detach().cpu())
                else:
                    x = blk(x)

        if self.norm is not None:
            x = self.norm(x)
        
        # 如果有分类头，计算分类结果（全局平均池化后接线性层）
        if self.head is not None:
            x_cls = x.mean(dim=1)  # 全局平均池化 [B, N, D] -> [B, D]
            x_cls = self.head(x_cls)  # [B, num_classes]
            if return_attention:
                return x_cls, attn_set
            return x_cls
            
        if return_attention:
            return x, attn_set
        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
}
