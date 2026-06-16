"""
神经风格迁移核心模块
基于 Gatys et al. "A Neural Algorithm of Artistic Style" 论文实现。
使用预训练 VGG19 提取特征，通过优化像素来迁移风格。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
# ---------- 设备选择 ----------
def get_device(use_gpu: bool | None = None) -> torch.device:
    """选择运算设备。"""
    if use_gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if use_gpu else "cpu")


# ---------- 图像处理 ----------
def load_image(
    path: str,
    target_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    加载图片并预处理为可用于 VGG 输入的 Tensor。

    Args:
        path: 图片路径
        target_size: 短边缩放到该尺寸（保持长宽比）
        device: 运算设备

    Returns:
        形状为 (1, 3, H, W) 的归一化 Tensor
    """
    img = Image.open(path).convert("RGB")
    # 保持长宽比，缩放至目标尺寸
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    将 VGG 归一化后的 Tensor 转回 PIL Image。

    Args:
        tensor: 形状为 (1, 3, H, W) 的归一化 Tensor

    Returns:
        PIL Image
    """
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().clone().squeeze(0)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return transforms.ToPILImage()(img)


def build_extractor_and_indices():
    """
    构建 VGG19 特征提取器，返回模型及内容/风格层索引。

    Returns:
        (extractor, content_indices, style_indices)
    """
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

    # 找到我们需要的层
    # VGG19 features: 0-35 (36 layers)
    # 每个 conv block 结构: Conv2d, ReLU, Conv2d, ReLU, MaxPool2d (除最后一层外)
    # conv1_1=0, conv1_2=2
    # conv2_1=5, conv2_2=7
    # conv3_1=10, conv3_2=12, conv3_3=14, conv3_4=16
    # conv4_1=19, conv4_2=21, conv4_3=23, conv4_4=25
    # conv5_1=28, conv5_2=30, conv5_3=32, conv5_4=34

    content_indices = [21]   # conv4_2
    style_indices = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

    # 截取到需要的最后一层 (conv5_1 = index 28)
    required = max(max(content_indices), max(style_indices))
    features = vgg[:required + 1]

    for param in features.parameters():
        param.requires_grad = False

    return features, content_indices, style_indices


# ---------- 损失函数 ----------
def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算 Gram 矩阵。
    Gram 矩阵刻画了特征图之间的相关性，用于表示"风格"。

    Args:
        tensor: 形状为 (B, C, H, W) 的特征图

    Returns:
        形状为 (B, C, C) 的 Gram 矩阵
    """
    B, C, H, W = tensor.shape
    features_flat = tensor.view(B, C, H * W)       # (B, C, H*W)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))  # (B, C, C)
    return gram / (C * H * W)  # 归一化


def content_loss(content_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """内容损失：生成图与内容图在 conv4_2 层的特征差异。"""
    return F.mse_loss(content_features, target_features)


def style_loss(
    generated_features: list[torch.Tensor],
    style_gram_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """
    风格损失：多层 Gram 矩阵的 MSE 之和。

    Args:
        generated_features: 生成图在各风格层的特征列表
        style_gram_matrices: 风格图在各风格层的 Gram 矩阵列表（预先计算好）

    Returns:
        风格损失（已按层数平均）
    """
    loss = 0.0
    for gen_feat, style_gram in zip(generated_features, style_gram_matrices):
        gen_gram = gram_matrix(gen_feat)
        loss += F.mse_loss(gen_gram, style_gram)
    return loss / len(generated_features)


def tv_loss(image: torch.Tensor) -> torch.Tensor:
    """
    总变差损失 (Total Variation Loss)：鼓励图像平滑，减少高频噪声。

    Args:
        image: 形状为 (1, 3, H, W) 的图像 Tensor

    Returns:
        TV loss 值
    """
    # 水平和垂直方向相邻像素差异的平方和
    h_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    w_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
    return h_diff.pow(2).mean() + w_diff.pow(2).mean()


# ---------- 主流程 ----------
class StyleTransfer:
    """神经风格迁移执行器。"""

    def __init__(
        self,
        style_weight: float = 1e6,
        content_weight: float = 1.0,
        tv_weight: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Args:
            style_weight: 风格损失权重（通常较大，如 1e4 ~ 1e7）
            content_weight: 内容损失权重
            tv_weight: 总变差损失权重（0 表示不使用）
            device: 运算设备
        """
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
        self.device = device or get_device()

        # 构建特征提取器
        self.extractor, self.content_indices, self.style_indices = \
            build_extractor_and_indices()
        self.extractor.to(self.device)
        self.extractor.eval()

    def transfer(
        self,
        style_image: torch.Tensor,
        content_image: torch.Tensor,
        steps: int = 300,
        lr: float = 0.2,
        optimizer_type: str = "adam",
        verbose: bool = True,
        callback: callable | None = None,
    ) -> torch.Tensor:
        """
        执行风格迁移。

        Args:
            style_image: 风格图 Tensor (1, 3, H, W)
            content_image: 内容图 Tensor (1, 3, H, W)
            steps: 优化迭代次数
            lr: 学习率（Adam 默认 0.2，L-BFGS 默认 1.0）
            optimizer_type: 优化器类型，"adam" 或 "lbfgs"
            verbose: 是否打印进度
            callback: 每 50 步调用的回调函数，接收 (step, generated_image_tensor)

        Returns:
            生成的图像 Tensor (1, 3, H, W)
        """
        # ---- 预计算内容特征 ----
        with torch.no_grad():
            content_feats_full, _ = self._forward(content_image)
            content_target = content_feats_full[0].detach()  # conv4_2 特征

        # ---- 预计算风格 Gram 矩阵 ----
        with torch.no_grad():
            _, style_feats_full = self._forward(style_image)
            style_grams = [gram_matrix(f).detach() for f in style_feats_full]

        # ---- 初始化生成图像（从内容图开始） ----
        generated = content_image.clone().detach()
        generated.requires_grad_(True)

        # ---- 优化器选择 ----
        if optimizer_type == "adam":
            optimizer = optim.Adam([generated], lr=lr)
        else:
            optimizer = optim.LBFGS([generated], lr=lr, max_iter=1,
                                    history_size=100, line_search_fn="strong_wolfe")

        # ---- 优化循环 ----
        for step in range(1, steps + 1):
            if optimizer_type == "adam":
                optimizer.zero_grad()
                gen_content_feats, gen_style_feats = self._forward(generated)

                c_loss = content_loss(gen_content_feats[0], content_target)
                s_loss = style_loss(gen_style_feats, style_grams)
                total = self.content_weight * c_loss + self.style_weight * s_loss
                if self.tv_weight > 0:
                    total += self.tv_weight * tv_loss(generated)

                total.backward()
                optimizer.step()
                loss_val = total
            else:
                def closure():
                    optimizer.zero_grad()
                    gen_content_feats, gen_style_feats = self._forward(generated)
                    c_loss = content_loss(gen_content_feats[0], content_target)
                    s_loss = style_loss(gen_style_feats, style_grams)
                    total = self.content_weight * c_loss + self.style_weight * s_loss
                    if self.tv_weight > 0:
                        total += self.tv_weight * tv_loss(generated)
                    total.backward()
                    return total
                loss_val = optimizer.step(closure)

            # 裁剪像素值到 ImageNet 归一化空间合法范围
            with torch.no_grad():
                generated.clamp_(-2.5, 2.5)

            if verbose and (step % 50 == 0 or step == 1 or step == steps):
                print(f"  Step {step:4d}/{steps}  |  Loss: {loss_val.item():.4f}")

            if callback and step % 50 == 0:
                callback(step, generated.clone().detach())

        return generated.detach()

    def _forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """提取内容和风格特征。"""
        content_features = []
        style_features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in self.content_indices:
                content_features.append(x)
            if i in self.style_indices:
                style_features.append(x)
        return content_features, style_features


# ---------- 便捷函数 ----------
def run_style_transfer(
    style_image_path: str,
    content_image_path: str,
    output_path: str = "output.jpg",
    image_size: int = 512,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    tv_weight: float = 0.0,
    steps: int = 300,
    lr: float = 0.2,
    optimizer: str = "adam",
    use_gpu: bool | None = None,
    verbose: bool = True,
) -> str:
    """
    一键执行风格迁移。

    Args:
        style_image_path: 风格图路径
        content_image_path: 内容图路径
        output_path: 输出路径
        image_size: 处理尺寸
        style_weight: 风格损失权重
        content_weight: 内容损失权重
        tv_weight: 总变差损失权重
        steps: 迭代次数
        lr: 学习率（Adam 默认 0.2）
        optimizer: 优化器类型 "adam" 或 "lbfgs"
        use_gpu: 是否使用 GPU
        verbose: 是否打印详细信息

    Returns:
        输出文件路径
    """
    device = get_device(use_gpu)
    if verbose:
        print(f"使用设备: {device}")
        print(f"加载风格图: {style_image_path}")
        print(f"加载内容图: {content_image_path}")

    style_img = load_image(style_image_path, target_size=image_size, device=device)
    content_img = load_image(content_image_path, target_size=image_size, device=device)

    if verbose:
        print(f"风格图尺寸: {style_img.shape}")
        print(f"内容图尺寸: {content_img.shape}")

    transfer = StyleTransfer(
        style_weight=style_weight,
        content_weight=content_weight,
        tv_weight=tv_weight,
        device=device,
    )

    if verbose:
        print(f"优化器: {optimizer.upper()}, 学习率: {lr}")
        print("开始风格迁移...")

    result = transfer.transfer(
        style_image=style_img,
        content_image=content_img,
        steps=steps,
        lr=lr,
        optimizer_type=optimizer,
        verbose=verbose,
    )

    out_img = tensor_to_image(result)
    out_img.save(output_path)

    if verbose:
        print(f"完成！结果已保存至: {output_path}")

    return output_path
