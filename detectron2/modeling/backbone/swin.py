# Copyright (c) OpenMMLab. All rights reserved.
# modified by author
import numpy as np
import warnings
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union
import logging
import math
from copy import deepcopy
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.layers import to_2tuple, trunc_normal_, ShapeSpec, DropPath, PatchEmbed, PatchMerging

import torch.utils.checkpoint as cp
from .backbone import Backbone
from .build import BACKBONE_REGISTRY


__all__ = [
    "WindowMSA",
    "ShiftWindowMSA",
    "FFN",
    "SwinBlock",
    "SwinBlockSequence",
    "SwinTransformer",
    "build_swin_backbone",
]

class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
    """
    def __init__(self, embed_dim, num_heads, window_size, qkv_bias=True,
                 qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.,
                 ):

        super().__init__()
        self.embed_dims = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dim // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
    """
    def __init__(self, embed_dim, num_heads, window_size, shift_size=0,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0,
                 drop_path_rate=0.):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(embed_dim=embed_dim, num_heads=num_heads,
            window_size=to_2tuple(window_size), qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop_path(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dim (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_layer (layer, optional): The activation config for FFNs.
            Default: nn.ReLU(inplace=True)
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
    """
    def __init__(self, embed_dim=256, feedforward_channels=1024, num_fcs=2,
                 act_layer=nn.ReLU(inplace=True), ffn_drop=0., drop_path_rate=0.):
        super().__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dim = embed_dim
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    act_layer, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dim))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    # @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x):
        """Forward function for `FFN`.
        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        return self.drop_path(out)

class SwinBlock(nn.Module):
    """"
    Args:
        embed_dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_layer (layer, optional): The activation config for FFNs.
            Default: nn.ReLU(inplace=True)
        norm_layer (layer, optional): The config of normalization.
            Default: nn.LayerNorm.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """
    def __init__(self, embed_dim, num_heads, feedforward_channels, window_size=7,
                 shift=False, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm, with_cp=False
                 ):

        super().__init__()

        self.with_cp = with_cp
        
        self.norm1 = norm_layer(embed_dim)
        self.attn = ShiftWindowMSA(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size,
            shift_size=window_size // 2 if shift else 0, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate, drop_path_rate=drop_path_rate
            )

        self.norm2 = norm_layer(embed_dim)
        self.ffn = FFN(embed_dim=embed_dim, feedforward_channels=feedforward_channels, num_fcs=2,
            ffn_drop=drop_rate, drop_path_rate=drop_path_rate, act_layer=act_layer)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x) + identity
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class SwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dim (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_layer (layer, optional): The activation config for FFNs.
            Default: nn.ReLU(inplace=True)
        norm_layer (layer, optional): The config of normalization.
            Default: nn.LayerNorm.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
    """
    def __init__(self, embed_dim, num_heads, feedforward_channels, depth, window_size=7,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 downsample=None, act_layer=nn.GELU(), norm_layer=nn.LayerNorm, with_cp=False
                 ):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dim=embed_dim, num_heads=num_heads, feedforward_channels=feedforward_channels,
                window_size=window_size, shift=False if i % 2 == 0 else True, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i],
                act_layer=act_layer, norm_layer=norm_layer, with_cp=with_cp)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class SwinTransformer(Backbone):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dim (int): The feature dimension. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_layer (layer, optional): The activation config for FFNs.
            Default: nn.ReLU(inplace=True)
        norm_layer (layer, optional): The config of normalization.
            Default: nn.LayerNorm.
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_channels=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4,
                 strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3), qkv_bias=True,
                 qk_scale=None, patch_norm=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, use_abs_pos_embed=False, act_layer=nn.GELU(),
                 norm_layer=nn.LayerNorm, with_cp=False):
        # self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        super().__init__()

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=embed_dim,
            kernel_size=patch_size, stride=strides[0], norm_layer=norm_layer if patch_norm else None,
            )

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dim)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages_and_indices = []
        in_channels = embed_dim
        
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(in_channels=in_channels, out_channels=2 * in_channels,
                    stride=strides[i + 1], norm_layer=norm_layer if patch_norm else None,
                    )
            else:
                downsample = None

            stage = SwinBlockSequence(embed_dim=in_channels, num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels, depth=depths[i], window_size=window_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=downsample,
                act_layer=act_layer, norm_layer=norm_layer, with_cp=with_cp)
            self.add_module("stages{}".format(i), stage)
            self.stages_and_indices.append((stage, i))
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dim * 2**i) for i in range(num_layers)]

        self._out_features = []
        self._out_feature_channels = []
        self._out_feature_strides = []
        # Add a norm layer for each output
        for i in out_indices:
            layer = norm_layer(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

            feature_name = f'swin{i}'
            self._out_features.append(feature_name)
            self._out_feature_channels.append(embed_dim * 2** i)
            self._out_feature_strides.append(4 * 2** i)

    def output_shape(self):
        return {
            self._out_features[i]: ShapeSpec(
                channels=self._out_feature_channels[i], stride=self._out_feature_strides[i]
            )
            for i in self.out_indices
        }

    def _freeze_stages(self, frozen_stages):
        if frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages_and_indices[i - 1][0]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        return self

    def init_weights(self):
        logger = logging.getLogger(__name__)
        logger.info(f'No pre-trained weights for '
                    f'{self.__class__.__name__}, '
                    f'training start from scratch')
    
        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outputs = {}
        for stage, i in self.stages_and_indices:
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                feature_name = self._out_features[i]
                outputs[feature_name] = out
        return outputs


@BACKBONE_REGISTRY.register()
def build_swin_backbone(cfg, input_shape):
    """
    Create a Swin Transformer instance from config.

    Returns:
        Swin Transformer: a :class:`Swin Transformer` instance.
    """

    patch_size        = cfg.MODEL.SWIN.PATCH_SIZE
    in_channels       = input_shape.channels
    embed_dim         = cfg.MODEL.SWIN.EMBED_DIM
    depths            = cfg.MODEL.SWIN.DEPTHS  
    heads             = cfg.MODEL.SWIN.HEADS
    window_size       = cfg.MODEL.SWIN.WINDOW_SIZE
    mlp_ratio         = cfg.MODEL.SWIN.MLP_RATIO
    out_indices       = cfg.MODEL.SWIN.OUT_INDICES
    qkv_bias          = cfg.MODEL.SWIN.QKV_BIAS
    qk_scale          = cfg.MODEL.SWIN.QK_SCALE
    patch_norm        = cfg.MODEL.SWIN.PATCH_NORM
    drop_rate         = cfg.MODEL.SWIN.DROP_RATE
    attn_drop_rate    = cfg.MODEL.SWIN.ATTN_DROP_RATE
    drop_path_rate    = cfg.MODEL.SWIN.DROP_PATH_RATE
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    return SwinTransformer(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
                           depths=depths, num_heads=heads, window_size=window_size, mlp_ratio=mlp_ratio, 
                           out_indices=out_indices, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                          patch_norm=patch_norm, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate)._freeze_stages(freeze_at)
