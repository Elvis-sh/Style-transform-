# 🎨 神经风格迁移 (Neural Style Transfer)

基于 Gatys et al. 经典算法的风格迁移工具——将一张图片的艺术风格迁移到另一张图片上，保留内容结构的同时改变其风格。

## 效果示例

| 风格图 | 内容图 | 输出 |
|--------|--------|------|
| 梵高《星月夜》 | 城市照片 | 梵高风格的城市场景 |
| 任意纹理/色彩 | 任意照片 | 风格化的照片 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 VGG19 预训练权重（约 500MB）。

### 2. 基础使用

```bash
python main.py --style-image style.jpg --content-image content.jpg
```

### 3. 自定义参数

```bash
python main.py \
    --style-image style.jpg \
    --content-image content.jpg \
    --output result.jpg \
    --image-size 512 \
    --steps 500 \
    --style-weight 1e6 \
    --tv-weight 1e-4
```

## 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--style-image` | `-s` | (必需) | 风格图路径 |
| `--content-image` | `-c` | (必需) | 内容图路径 |
| `--output` | `-o` | `output.jpg` | 输出路径 |
| `--image-size` | | `512` | 处理尺寸（像素），越大越精细但更慢 |
| `--style-weight` | | `1e6` | 风格权重，越大风格越强 |
| `--content-weight` | | `1.0` | 内容权重 |
| `--tv-weight` | | `0` | 平滑正则化，推荐 `1e-4` ~ `1e-3` |
| `--steps` | | `300` | 迭代步数，越多越精细 |
| `--lr` | | `0.003` | 学习率 |
| `--gpu` | | 自动 | 强制使用 GPU |
| `--no-gpu` | | 自动 | 强制使用 CPU |

## 参数调优建议

- **风格不够强**：增大 `--style-weight`（如 `5e6`）
- **内容丢失太多**：减小 `--style-weight` 或增大 `--content-weight`
- **图像有噪点**：添加 `--tv-weight 1e-4`
- **需要更精细**：增大 `--image-size` 和 `--steps`
- **显存不足**：减小 `--image-size`（如 `256` 或 `384`）

## 原理简介

1. 使用预训练的 **VGG19** 卷积神经网络提取图像特征
2. **内容表示**：取高层特征图（`conv4_2`）保留空间结构
3. **风格表示**：取多层特征图的 **Gram 矩阵** 捕获纹理和色彩统计
4. 从内容图出发，用 **L-BFGS** 优化器迭代生成一张图像，使其同时匹配内容特征和风格 Gram 矩阵

> 论文：Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style.*

## 系统要求

- Python 3.8+
- 建议 8GB+ RAM
- 可选：NVIDIA GPU + CUDA（加速 10-50x）
