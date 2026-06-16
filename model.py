"""
快速风格迁移变换网络 (TransformerNet)
基于 Johnson et al. "Perceptual Losses for Real-Time Style Transfer"

架构: Encoder → Residual Blocks → Decoder
使用 Instance Normalization 提升风格迁移质量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块：两个 3×3 卷积 + InstanceNorm + ReLU"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class TransformerNet(nn.Module):
    """
    图像变换网络。
    输入: 内容图像 (B, 3, H, W)
    输出: 风格化图像 (B, 3, H, W)

    结构:
      Encoder:   Conv 9×9 → Conv 3×3↓ → Conv 3×3↓
      Residual:  5 × ResidualBlock
      Decoder:    ↑Conv 3×3 → ↑Conv 3×3 → Conv 9×9
    """

    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        # 初始卷积 (保持尺寸，扩充通道)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        # 下采样 ×2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        # ---- Residual Blocks ----
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])

        # ---- Decoder ----
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.out_conv(x)
        # 输出范围 [-1, 1]，使用时反归一化到 [0, 1]
        return x


# ---------- 工具函数 ----------
def deprocess(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 TransformerNet 的输出 [-1, 1] 转换回 [0, 1]
    """
    return tensor.clamp(-1, 1).detach() * 0.5 + 0.5
