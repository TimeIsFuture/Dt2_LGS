#coding=utf-8
#modified from https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        """Implementation of Self-Attention
        args:
            in_channels: original channel size
            inter_channels: channel size inside the module (in_channels//2 in the non-local block)
        """
        super(SelfAttention, self).__init__()

        self.in_channels = in_channels
        # the channel size is reduced to half inside the module
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # function g, using conv. with kernel size 1
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

        # from section 3.3 of the non-local block paper by initializing Wz to 0, this module can be inserted to any existing architecture
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    def forward(self, x):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        scale_factor = theta_x.size(-1)** (-0.5)
        f = torch.matmul(theta_x, phi_x) * scale_factor 
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W_z(y)
        z = W_y + x

        return z