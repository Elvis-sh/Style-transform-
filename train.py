"""
快速风格迁移 - 训练脚本

使用 ImageNet/COCO 内容图 + 预训练 VGG-16 损失网络训练 TransformerNet。
训练完成后，风格迁移只需一次前向传播（< 0.1 秒）。

基于: Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import TransformerNet
from dataset import create_dataloader, download_coco_subset
from style_transfer import load_image, tensor_to_image, get_device


# ================================================================
#  VGG-16 损失网络
# ================================================================

class VGGLossNet(nn.Module):
    """
    预训练 VGG-16，用于计算感知损失（内容损失 + 风格损失）。
    冻结所有参数，只用于前向计算。
    """

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()

        # 截取 4 段，用于计算多层损失
        slice1 = nn.Sequential()
        slice2 = nn.Sequential()
        slice3 = nn.Sequential()
        slice4 = nn.Sequential()

        for i in range(4):    slice1.add_module(str(i), vgg[i])      # relu1_2 (after 4 layers)
        for i in range(4, 9):  slice2.add_module(str(i), vgg[i])    # relu2_2
        for i in range(9, 16): slice3.add_module(str(i), vgg[i])    # relu3_3
        for i in range(16, 23): slice4.add_module(str(i), vgg[i])   # relu4_3

        self.slices.append(slice1)
        self.slices.append(slice2)
        self.slices.append(slice3)
        self.slices.append(slice4)

        for param in self.parameters():
            param.requires_grad = False
        self.to(device)
        self.eval()

    def forward(self, x):
        """返回 relu1_2, relu2_2, relu3_3, relu4_3 的特征图"""
        feats = []
        for slc in self.slices:
            x = slc(x)
            feats.append(x)
        return feats

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [-1, 1] 的 TransformerNet 输出归一化为 VGG 输入格式。
        VGG 期望 ImageNet 归一化 (mean/std)，输入范围约 [0, 1]。
        """
        # x 在 [-1, 1] → 先到 [0, 1]
        x = (x + 1) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


# ================================================================
#  损失函数
# ================================================================

def gram_matrix(tensor):
    """计算 Gram 矩阵"""
    b, c, h, w = tensor.shape
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def compute_losses(
    vgg: VGGLossNet,
    generated: torch.Tensor,
    content_target_feats: list[torch.Tensor],
    style_target_grams: list[torch.Tensor],
    content_weight: float,
    style_weight: float,
    tv_weight: float,
):
    """
    计算总损失。

    Args:
        vgg: VGG 损失网络
        generated: TransformerNet 输出 (B, 3, H, W) in [-1, 1]
        content_target_feats: 内容图在 VGG 各层的特征（预计算）
        style_target_grams: 风格图在 VGG 各层的 Gram 矩阵（预计算）

    Returns:
        (total_loss, content_loss_val, style_loss_val, tv_loss_val)
    """
    gen_norm = vgg.normalize_input(generated)
    gen_feats = vgg(gen_norm)

    # 内容损失: relu3_3 层
    content_loss_val = F.mse_loss(gen_feats[2], content_target_feats[2])

    # 风格损失: relu1_2, relu2_2, relu3_3, relu4_3 层 Gram 矩阵
    style_loss_val = 0.0
    batch_size = gen_feats[0].shape[0]
    for gen_f, style_gram in zip(gen_feats, style_target_grams):
        gen_gram = gram_matrix(gen_f)
        # 将 style_gram 从 (1, C, C) 扩展到 (B, C, C) 以匹配批次维度
        sg = style_gram.expand(batch_size, -1, -1)
        style_loss_val += F.mse_loss(gen_gram, sg)
    style_loss_val = style_loss_val / len(gen_feats)

    # 总变差损失（平滑正则化）
    tv_loss_val = 0.0
    if tv_weight > 0:
        tv_loss_val = (
            torch.mean((generated[:, :, 1:, :] - generated[:, :, :-1, :]) ** 2) +
            torch.mean((generated[:, :, :, 1:] - generated[:, :, :, :-1]) ** 2)
        )

    total = content_weight * content_loss_val + style_weight * style_loss_val + tv_weight * tv_loss_val

    return total, content_loss_val.item(), style_loss_val.item(), tv_loss_val.item()


# ================================================================
#  预处理风格图和内容特征目标
# ================================================================

def prepare_targets(
    vgg: VGGLossNet,
    style_image_path: str,
    content_image_batch: torch.Tensor,
    device: torch.device,
    image_size: int,
):
    """
    预计算风格图的 Gram 矩阵目标和一批内容图的特征目标。

    Returns:
        (style_target_grams, content_target_feats)
    """
    # 风格目标
    style_img = load_image(style_image_path, target_size=image_size, device=device)
    with torch.no_grad():
        style_feats = vgg(style_img)
        style_grams = [gram_matrix(f) for f in style_feats]

    # 内容目标（用这批内容图的第一张）
    # 实际训练中每张图都作为自己的内容目标
    content_norm = vgg.normalize_input(content_image_batch)
    with torch.no_grad():
        content_feats = vgg(content_norm)

    return style_grams, content_feats


# ================================================================
#  训练主流程
# ================================================================

def train(
    style_image_path: str,
    train_data_dir: str,
    output_dir: str = "checkpoints",
    image_size: int = 256,
    batch_size: int = 4,
    epochs: int = 2,
    style_weight: float = 1e5,
    content_weight: float = 1.0,
    tv_weight: float = 1e-6,
    lr: float = 1e-3,
    log_interval: int = 100,
    save_interval: int = 500,
    use_gpu: bool | None = None,
    num_images: int | None = None,
):
    """
    训练快速风格迁移模型。

    Args:
        style_image_path: 风格图路径
        train_data_dir: 训练图片文件夹
        output_dir: 模型输出目录
        image_size: 训练 patch 尺寸
        batch_size: 批次大小
        epochs: 训练轮数
        style_weight: 风格损失权重
        content_weight: 内容损失权重
        tv_weight: 总变差损失权重
        lr: 学习率
        log_interval: 日志间隔（批次）
        save_interval: 模型保存间隔（批次）
        use_gpu: 是否使用 GPU
        num_images: 限制训练图片数量（None=全部）
    """
    device = get_device(use_gpu)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  快 速 风 格 迁 移 - 训 练")
    print("=" * 60)
    print(f"  设备:       {device}")
    print(f"  风格图:     {style_image_path}")
    print(f"  训练数据:   {train_data_dir}")
    print(f"  输出目录:   {output_dir}")
    print(f"  图像尺寸:   {image_size}")
    print(f"  批次大小:   {batch_size}")
    print(f"  训练轮数:   {epochs}")
    print(f"  风格权重:   {style_weight:.0e}")
    print(f"  内容权重:   {content_weight}")
    print(f"  TV 权重:    {tv_weight:.0e}")
    print("=" * 60)
    print()

    # ---- 加载数据集 ----
    dataloader = create_dataloader(
        train_data_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    # ---- 构建模型 ----
    transformer = TransformerNet().to(device)
    transformer.train()

    vgg = VGGLossNet(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    # ---- 预计算风格目标 ----
    print("预计算风格图 Gram 矩阵...")
    style_img = load_image(style_image_path, target_size=image_size, device=device)
    with torch.no_grad():
        style_feats = vgg(style_img)
        style_target_grams = [gram_matrix(f).detach() for f in style_feats]
    print("完成。\n")

    # ---- 训练循环 ----
    total_batches = len(dataloader) * epochs
    batch_count = 0
    best_loss = float("inf")

    epoch_style_losses = []
    epoch_content_losses = []

    print("开始训练...")
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        for batch_idx, content_imgs in enumerate(dataloader):
            batch_count += 1

            if num_images and batch_count * batch_size > num_images:
                break

            content_imgs = content_imgs.to(device)

            # ---- 前向传播 ----
            optimizer.zero_grad()
            generated = transformer(content_imgs)

            # ---- 计算内容目标（当前批次） ----
            content_norm = vgg.normalize_input(content_imgs)
            with torch.no_grad():
                content_feats = vgg(content_norm)

            # ---- 计算损失 ----
            total_loss, c_loss, s_loss, tv = compute_losses(
                vgg, generated, content_feats, style_target_grams,
                content_weight, style_weight, tv_weight,
            )

            # ---- 反向传播 ----
            total_loss.backward()
            optimizer.step()

            epoch_style_losses.append(s_loss)
            epoch_content_losses.append(c_loss)

            # ---- 日志 ----
            if batch_count % log_interval == 0 or batch_count == 1:
                elapsed = time.time() - train_start
                print(
                    f"  Epoch {epoch:2d}/{epochs} | "
                    f"Batch {batch_count:5d}/{total_batches} | "
                    f"Total: {total_loss.item():.4f} | "
                    f"Content: {c_loss:.4f} | "
                    f"Style: {s_loss:.4f} | "
                    f"{elapsed:.0f}s"
                )

            # ---- 保存模型 ----
            if batch_count % save_interval == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{batch_count}.pth")
                torch.save(transformer.state_dict(), ckpt_path)
                print(f"  保存检查点: {ckpt_path}")

            if num_images and batch_count * batch_size >= num_images:
                break

        if num_images and batch_count * batch_size >= num_images:
            break

    train_end = time.time()

    # ---- 保存最终模型 ----
    final_path = os.path.join(output_dir, "transformer_final.pth")
    torch.save(transformer.state_dict(), final_path)

    avg_content = sum(epoch_content_losses) / len(epoch_content_losses) if epoch_content_losses else 0
    avg_style   = sum(epoch_style_losses)   / len(epoch_style_losses)   if epoch_style_losses   else 0

    print()
    print("=" * 60)
    print("  训 练 完 成")
    print("=" * 60)
    print(f"  总批次:       {batch_count}")
    print(f"  训练时长:     {train_end - train_start:.0f}s")
    print(f"  平均内容损失: {avg_content:.4f}")
    print(f"  平均风格损失: {avg_style:.4f}")
    print(f"  最终模型:     {final_path}")
    print("=" * 60)

    # ---- 测试推理速度 ----
    print("\n测试推理速度...")
    transformer.eval()
    test_img = torch.randn(1, 3, image_size, image_size).to(device)
    times = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            transformer(test_img)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append(time.perf_counter() - t0)
    avg_time = sum(times[2:]) / len(times[2:])  # 去掉预热
    print(f"  平均推理时间 (delta_time): {avg_time*1000:.2f} ms/张")
    print(f"  约 {1/avg_time:.0f} 张/秒")

    return final_path


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="训练快速风格迁移模型")
    parser.add_argument("--style-image", "-s", required=True, help="风格图路径")
    parser.add_argument("--data-dir", "-d", default="train_data",
                        help="训练数据文件夹 (默认: train_data)")
    parser.add_argument("--output-dir", "-o", default="checkpoints",
                        help="模型输出目录 (默认: checkpoints)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--style-weight", type=float, default=1e5)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--download-coco", action="store_true",
                        help="自动下载 COCO 子集作为训练数据")
    parser.add_argument("--num-images", type=int, default=None,
                        help="限制训练图片数量（快速测试）")
    parser.add_argument("--gpu", action="store_true", dest="use_gpu")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu")
    parser.set_defaults(use_gpu=None)

    args = parser.parse_args()

    # 准备训练数据
    data_dir = args.data_dir
    if args.download_coco or not os.path.isdir(data_dir):
        try:
            data_dir = download_coco_subset(data_dir, num_images=args.num_images or 5000)
        except Exception:
            print("\n自动下载失败。请手动准备训练数据:")
            print(f"  1. 创建文件夹: {data_dir}/")
            print(f"  2. 放入至少几十张任意 RGB 图片")
            sys.exit(1)

    train(
        style_image_path=args.style_image,
        train_data_dir=data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
        use_gpu=args.use_gpu,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()
