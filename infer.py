"""
快速风格迁移 - 推理脚本

使用训练好的 TransformerNet 模型，单次前向传播完成风格迁移。
"""

import os
import sys
import time
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import TransformerNet, deprocess
from style_transfer import get_device


def load_model(checkpoint_path: str, device: torch.device) -> TransformerNet:
    """加载训练好的模型权重。"""
    model = TransformerNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def style_transfer_fast(
    model: TransformerNet,
    content_image: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    单次前向传播完成风格迁移。

    Args:
        model: TransformerNet
        content_image: 内容图 (1, 3, H, W) 在 [0, 1] 范围
        device: 设备

    Returns:
        (stylized_image, delta_time_seconds)
    """
    # 转换为 [-1, 1]（模型输入范围）
    img = content_image.to(device)
    img = img * 2 - 1

    # 推理计时
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model(img)

    if device.type == "cuda":
        torch.cuda.synchronize()
    delta_time = time.perf_counter() - t0

    # 转换回 [0, 1]
    result = deprocess(output)

    return result, delta_time


def load_image_as_tensor(path: str, max_size: int = 720) -> torch.Tensor:
    """加载图片为 Tensor [0, 1]，保持长宽比。"""
    img = Image.open(path).convert("RGB")

    # 保持长宽比缩放
    w, h = img.size
    if max(w, h) > max_size:
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        img = img.resize((new_w, new_h), Image.LANCZOS)

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    return tensor


def infer(
    content_image_path: str,
    checkpoint_path: str,
    output_path: str = "output.jpg",
    max_size: int = 720,
    use_gpu: bool | None = None,
):
    """
    快速风格迁移推理。

    Args:
        content_image_path: 内容图路径
        checkpoint_path: 训练好的模型权重
        output_path: 输出路径
        max_size: 图片最大边长
        use_gpu: 是否使用 GPU
    """
    device = get_device(use_gpu)

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # 加载图片
    print(f"加载图片: {content_image_path}")
    content_tensor = load_image_as_tensor(content_image_path, max_size=max_size)
    print(f"图片尺寸: {content_tensor.shape[2]}×{content_tensor.shape[3]}")

    # 推理
    print("执行风格迁移...")
    result, delta_time = style_transfer_fast(model, content_tensor, device)
    print(f"  delta_time: {delta_time*1000:.2f} ms ({delta_time:.4f}s)")

    # 保存
    out_img = transforms.ToPILImage()(result.squeeze(0).cpu())
    out_img.save(output_path)
    print(f"完成！结果: {output_path}")

    return output_path, delta_time


def main():
    parser = argparse.ArgumentParser(
        description="快速风格迁移推理 - 使用训练好的模型进行单次前向传播"
    )
    parser.add_argument("--content-image", "-c", required=True, help="内容图路径")
    parser.add_argument("--model", "-m", default="checkpoints/transformer_final.pth",
                        help="训练好的模型权重路径")
    parser.add_argument("--output", "-o", default="output.jpg", help="输出路径")
    parser.add_argument("--max-size", type=int, default=720,
                        help="图片最大边长 (默认: 720)")
    parser.add_argument("--gpu", action="store_true", dest="use_gpu")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu")
    parser.set_defaults(use_gpu=None)

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"错误: 模型文件不存在: {args.model}", file=sys.stderr)
        print("请先运行训练: python train.py --style-image <风格图> --download-coco", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.content_image):
        print(f"错误: 图片不存在: {args.content_image}", file=sys.stderr)
        sys.exit(1)

    infer(
        content_image_path=args.content_image,
        checkpoint_path=args.model,
        output_path=args.output,
        max_size=args.max_size,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
