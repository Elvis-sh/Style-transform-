"""
神经风格迁移 - 命令行入口
用法:
    python main.py --style-image <风格图> --content-image <内容图> [--output <输出>]

示例:
    # 基础用法
    python main.py --style-image style.jpg --content-image content.jpg

    # 自定义参数
    python main.py --style-image style.jpg --content-image content.jpg \\
                   --output result.jpg --image-size 512 --steps 500 \\
                   --style-weight 1e6 --tv-weight 1e-3

    # 使用 GPU（默认自动检测）
    python main.py --style-image style.jpg --content-image content.jpg --gpu

    # 强制使用 CPU
    python main.py --style-image style.jpg --content-image content.jpg --no-gpu
"""

import argparse
import sys
import os

from style_transfer import run_style_transfer


def main():
    parser = argparse.ArgumentParser(
        description="神经风格迁移 - 将风格图的风格迁移到内容图上",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --style-image style.jpg --content-image content.jpg
  python main.py --style-image style.jpg --content-image content.jpg --output result.jpg --steps 500
        """,
    )

    # ---- 必需参数 ----
    parser.add_argument(
        "--style-image", "-s",
        type=str,
        required=True,
        help="风格图像路径（提供艺术风格）",
    )
    parser.add_argument(
        "--content-image", "-c",
        type=str,
        required=True,
        help="内容图像路径（提供内容结构）",
    )

    # ---- 可选参数 ----
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.jpg",
        help="输出图像路径 (默认: output.jpg)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="处理尺寸，短边缩放到该像素数 (默认: 512)。"
             "越大细节越多但更吃显存和耗时",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1e6,
        help="风格损失权重 (默认: 1e6)。"
             "增大 = 更强烈的风格效果，减小 = 更接近原图",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1.0,
        help="内容损失权重 (默认: 1.0)",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="总变差损失权重 (默认: 0)。"
             "设为 1e-3 ~ 1e-4 可获得更平滑的效果",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="优化迭代次数 (默认: 300)。越多越精细但更耗时",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="学习率 (默认: 0.2，适用于 Adam; L-BFGS 建议 1.0)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "lbfgs"],
        help="优化器类型 (默认: adam)。adam 收敛更稳定，lbfgs 收敛更快",
    )

    # ---- GPU 控制 ----
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        action="store_true",
        dest="use_gpu",
        default=None,
        help="强制使用 GPU",
    )
    gpu_group.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        default=None,
        help="强制使用 CPU",
    )

    args = parser.parse_args()

    # ---- 验证输入文件存在 ----
    for path, name in [(args.style_image, "风格图"), (args.content_image, "内容图")]:
        if not os.path.isfile(path):
            print(f"错误: {name}不存在: {path}", file=sys.stderr)
            sys.exit(1)

    # ---- 确保输出目录存在 ----
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # ---- 执行风格迁移 ----
    print("=" * 50)
    print("  神 经 风 格 迁 移")
    print("=" * 50)
    print(f"  风格图:   {args.style_image}")
    print(f"  内容图:   {args.content_image}")
    print(f"  输出:     {args.output}")
    print(f"  尺寸:     {args.image_size}px")
    print(f"  迭代:     {args.steps} 步")
    print(f"  风格权重: {args.style_weight:.0e}")
    print(f"  内容权重: {args.content_weight}")
    if args.tv_weight > 0:
        print(f"  TV 权重:  {args.tv_weight:.0e}")
    print("=" * 50)

    try:
        result_path = run_style_transfer(
            style_image_path=args.style_image,
            content_image_path=args.content_image,
            output_path=args.output,
            image_size=args.image_size,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            tv_weight=args.tv_weight,
            steps=args.steps,
            lr=args.lr,
            optimizer=args.optimizer,
            use_gpu=args.use_gpu,
            verbose=True,
        )
        print(f"\n✓ 风格迁移完成！结果: {result_path}")
    except KeyboardInterrupt:
        print("\n\n用户中断操作。", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
