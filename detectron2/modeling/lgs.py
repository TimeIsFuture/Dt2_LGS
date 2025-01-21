# coding=utf-8

from typing import Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

class LocalGlobalSemantics(nn.Module):
    """
    LocalGlobalSemantics: The input and output of `forward()` method must be NCHW tensors.
    """
    def __init__(self, in_channels, out_channels, K=7, groups=8, heads=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels & in_channels
        channel_dict = {256: 768, 512: 512, 1024: 384, 2048: 256, 96: 768, 192: 512, 384: 384, 768: 256}
        inter_channels = channel_dict[in_channels]
        self.inter_channels = inter_channels
        self.heads = heads
        self.head_dim = inter_channels//heads
        
        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=K, padding=(K - 1) // 2, groups=groups)
        self.norm = nn.LayerNorm(inter_channels)
        self.q_proj = nn.Linear(inter_channels, inter_channels)
        self.k_proj = nn.Linear(inter_channels, inter_channels)
        self.v_proj = nn.Linear(inter_channels, inter_channels)
        self.atten_proj = nn.Linear(inter_channels, in_channels)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x
        N, C, H, W = x.shape
        HW=H*W
        
        x_c = self.conv(x).flatten(2).permute(0, 2, 1)
        
        x_norm = self.norm(x_c)
        q = self.q_proj(x_norm).view(N, HW, self.heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # N, heads, HW, head_dim
        k = self.k_proj(x_norm).view(N, HW, self.heads, self.head_dim).permute(0, 2, 3, 1).contiguous()  # N, heads, head_dim, HW
        v = self.v_proj(x_norm).view(N, HW, self.heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # N, heads, HW, head_dim
        k = torch.softmax(k, dim=3)
        correlation = k @ v
        atten = q @ correlation
        atten = atten.permute(0, 2, 1, 3).contiguous()
        atten = atten.view(N, HW, self.inter_channels)
        x_a = self.atten_proj(atten)
        x_a = x_a.permute(0, 2, 1).contiguous()
        x_a = x_a.view(N, C, H, W)
        
        x_o = self.beta * x_a + shortcut
        return x_o
