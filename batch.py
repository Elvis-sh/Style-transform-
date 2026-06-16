"""
批量风格迁移处理脚本

目录结构:
  tasks/                    ← 所有任务组
  ├── 001/                  ← 编号 001 组
  │   ├── 001_content.jpg   ← 内容图（待迁移的图片）
  │   ├── 001_style.jpg     ← 风格图（提供风格）
  │   └── 001_result.jpg    ← 结果图（自动生成）
  ├── 002/
  │   ├── 002_content.jpg
  │   ├── 002_style.jpg
  │   └── 002_result.jpg
  └── ...

用法:
  # 创建新任务组（生成文件夹模板）
  python batch.py --create 004

  # 处理所有任务组（迭代优化，较慢）
  python batch.py

  # 快速推理（需先训练模型）
  python batch.py --fast --model checkpoints/transformer_final.pth

  # 只处理指定编号
  python batch.py --ids 001,003

  # 预览模式（只列出配对，不执行）
  python batch.py --dry-run
"""

import argparse
import os
import re
import sys
import time
import shutil

import torch

from style_transfer import run_style_transfer, get_device
from model import TransformerNet, deprocess
from infer import load_model, load_image_as_tensor, style_transfer_fast


# ---------- 配置 ----------
TASKS_DIR = "tasks"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_task_groups(tasks_dir: str = TASKS_DIR) -> list[str]:
    """
    扫描 tasks/ 下所有编号子文件夹。

    Returns:
        ["001", "002", ...] 按编号排序
    """
    if not os.path.isdir(tasks_dir):
        return []

    ids = []
    for name in os.listdir(tasks_dir):
        subdir = os.path.join(tasks_dir, name)
        if os.path.isdir(subdir) and re.match(r"^\d+", name):
            ids.append(name)
    return sorted(ids)


def find_images_in_group(group_dir: str, group_id: str) -> tuple[str | None, str | None]:
    """
    在任务组文件夹中查找内容图和风格图。

    匹配规则:
      内容图: {group_id}_content.{ext}
      风格图: {group_id}_style.{ext}

    Returns:
        (content_path, style_path)  — 缺失的为 None
    """
    if not os.path.isdir(group_dir):
        return None, None

    content_path = None
    style_path = None

    for f in os.listdir(group_dir):
        full = os.path.join(group_dir, f)
        if not os.path.isfile(full):
            continue
        if not any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue

        name_no_ext = os.path.splitext(f)[0]
        if name_no_ext == f"{group_id}_content":
            content_path = full
        elif name_no_ext == f"{group_id}_style":
            style_path = full

    return content_path, style_path


def create_task_group(group_id: str, tasks_dir: str = TASKS_DIR) -> str:
    """
    创建一个新的任务组文件夹。

    Returns:
        新文件夹路径
    """
    group_dir = os.path.join(tasks_dir, group_id)
    os.makedirs(group_dir, exist_ok=True)
    print(f"创建任务组文件夹: {group_dir}/")
    print(f"\n请将以下文件放入该文件夹:")
    print(f"  {group_id}_content.jpg  ← 内容图（要迁移风格的图片）")
    print(f"  {group_id}_style.jpg    ← 风格图（提供艺术风格）")
    print(f"\n然后运行: python batch.py --ids {group_id}")
    return group_dir


def main():
    parser = argparse.ArgumentParser(
        description="批量神经风格迁移 - 处理 tasks/{编号}/ 中的所有任务组",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 操作命令 ----
    parser.add_argument(
        "--create", type=str, default=None, metavar="ID",
        help="创建新任务组文件夹，如: --create 004",
    )

    # ---- 参数 ----
    parser.add_argument(
        "--image-size", type=int, default=384,
        help="处理尺寸 (默认: 384)",
    )
    parser.add_argument(
        "--style-weight", type=float, default=1e6,
        help="风格权重 (默认: 1e6)",
    )
    parser.add_argument(
        "--content-weight", type=float, default=1.0,
        help="内容权重 (默认: 1.0)",
    )
    parser.add_argument(
        "--tv-weight", type=float, default=1e-4,
        help="TV 平滑权重 (默认: 1e-4)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="迭代步数 (默认: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.2,
        help="学习率 (默认: 0.2)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "lbfgs"],
        help="优化器 (默认: adam)",
    )
    parser.add_argument(
        "--ids", type=str, default=None,
        help="只处理指定编号，用逗号分隔（如: 001,003）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只列出任务组，不执行迁移",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="使用训练好的 TransformerNet 进行快速推理（< 0.1s/张）",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="checkpoints/transformer_final.pth",
        help="快速推理模式下的模型权重路径 (默认: checkpoints/transformer_final.pth)",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", action="store_true", dest="use_gpu")
    gpu_group.add_argument("--no-gpu", action="store_false", dest="use_gpu")
    gpu_group.set_defaults(use_gpu=None)

    args = parser.parse_args()

    # ---- --create 模式 ----
    if args.create:
        create_task_group(args.create)
        return

    # ---- 扫描任务组 ----
    group_ids = find_task_groups()

    if not group_ids:
        print(f"未找到任何任务组！请先将图片放入 '{TASKS_DIR}/' 下。")
        print(f"\n用法:")
        print(f"  1. 创建任务组: python batch.py --create 001")
        print(f"  2. 放入图片:   将内容图和风格图放入 tasks/001/")
        print(f"  3. 运行迁移:   python batch.py")
        sys.exit(1)

    # ---- 过滤指定编号 ----
    if args.ids:
        target_ids = set(i.strip() for i in args.ids.split(","))
        group_ids = [gid for gid in group_ids if gid in target_ids]
        if not group_ids:
            print(f"没有找到编号为 {args.ids} 的任务组")
            sys.exit(1)

    # ---- 构建任务列表 ----
    tasks = []
    skipped = []
    for gid in group_ids:
        group_dir = os.path.join(TASKS_DIR, gid)
        content_path, style_path = find_images_in_group(group_dir, gid)
        if content_path and style_path:
            # 输出到同目录的 _result 文件
            ext = os.path.splitext(content_path)[1]
            output_path = os.path.join(group_dir, f"{gid}_result{ext}")
            tasks.append((gid, content_path, style_path, output_path))
        else:
            missing = []
            if not content_path:
                missing.append(f"{gid}_content.*")
            if not style_path:
                missing.append(f"{gid}_style.*")
            skipped.append((gid, missing))

    # ---- 显示任务列表 ----
    print("=" * 60)
    print("  批 量 风 格 迁 移")
    print("=" * 60)

    if skipped:
        print(f"\n⚠  跳过的任务组（缺少文件）:\n")
        for gid, missing in skipped:
            print(f"  [{gid}] 缺少: {', '.join(missing)}")

    if tasks:
        print(f"\n待处理: {len(tasks)} 组\n")
        for gid, cp, sp, op in tasks:
            print(f"  [{gid}] {os.path.basename(sp)} → {os.path.basename(cp)}")
            print(f"        └→ {os.path.basename(op)}")
    else:
        print("\n没有可以处理的任务组！")
        sys.exit(1)

    print()

    if args.dry_run:
        print("(预览模式，不执行迁移)")
        return

    # ---- 设备 ----
    device = get_device(args.use_gpu)

    if args.fast:
        # ---- 快速推理模式 ----
        if not os.path.isfile(args.model):
            print(f"\n错误: 模型文件不存在: {args.model}", file=sys.stderr)
            print("请先训练模型: python train.py --style-image <风格图> --download-coco", file=sys.stderr)
            sys.exit(1)

        print(f"模式: 快速推理 | 设备: {device} | 模型: {args.model}")
        print(f"delta_time 预计: < 0.1s/张\n")

        model = load_model(args.model, device)

        total = len(tasks)
        start_time = time.time()
        success = 0
        total_delta = 0.0

        for i, (gid, content_path, style_path, output_path) in enumerate(tasks, 1):
            print(f"[{i}/{total}] 处理编号 {gid}")
            print(f"  内容: {os.path.basename(content_path)}")

            try:
                content_tensor = load_image_as_tensor(content_path, max_size=args.image_size)
                result, delta_time = style_transfer_fast(model, content_tensor, device)

                # 保存
                from torchvision import transforms
                out_img = transforms.ToPILImage()(result.squeeze(0).cpu())
                out_img.save(output_path)

                total_delta += delta_time
                print(f"  ✓ 完成 | delta_time: {delta_time*1000:.1f}ms\n")
                success += 1
            except KeyboardInterrupt:
                print(f"\n用户中断。已完成 {success}/{total} 组。", file=sys.stderr)
                sys.exit(130)
            except Exception as e:
                print(f"  ✗ 失败: {e}\n", file=sys.stderr)

        total_elapsed = time.time() - start_time
        print("=" * 60)
        print(f"  全部完成！成功 {success}/{total} 组，耗时 {total_elapsed:.0f}s")
        print(f"  总 delta_time: {total_delta:.3f}s | 平均: {total_delta/total*1000:.1f}ms/张")
        print(f"  结果已保存在各组的 '{TASKS_DIR}/{{编号}}/' 文件夹中")
        print("=" * 60)

    else:
        # ---- 迭代优化模式 ----
        print(f"模式: 迭代优化 | 设备: {device} | 尺寸: {args.image_size}px | 步数: {args.steps}")
        print(f"优化器: {args.optimizer} | 风格权重: {args.style_weight:.0e} | TV权重: {args.tv_weight:.0e}\n")

        total = len(tasks)
        start_time = time.time()
        success = 0

        for i, (gid, content_path, style_path, output_path) in enumerate(tasks, 1):
            print(f"[{i}/{total}] 处理编号 {gid}")
            print(f"  风格: {os.path.basename(style_path)}")
            print(f"  内容: {os.path.basename(content_path)}")

            pair_start = time.time()
            try:
                run_style_transfer(
                    style_image_path=style_path,
                    content_image_path=content_path,
                    output_path=output_path,
                    image_size=args.image_size,
                    style_weight=args.style_weight,
                    content_weight=args.content_weight,
                    tv_weight=args.tv_weight,
                    steps=args.steps,
                    lr=args.lr,
                    optimizer=args.optimizer,
                    use_gpu=args.use_gpu,
                    verbose=False,
                )
                elapsed = time.time() - pair_start
                print(f"  ✓ 完成 ({elapsed:.0f}s)\n")
                success += 1
            except KeyboardInterrupt:
                print(f"\n用户中断。已完成 {success}/{total} 组。", file=sys.stderr)
                sys.exit(130)
            except Exception as e:
                print(f"  ✗ 失败: {e}\n", file=sys.stderr)

        total_elapsed = time.time() - start_time
        print("=" * 60)
        print(f"  全部完成！成功 {success}/{total} 组，耗时 {total_elapsed:.0f}s")
        print(f"  结果已保存在各组的 '{TASKS_DIR}/{{编号}}/' 文件夹中")
        print("=" * 60)


if __name__ == "__main__":
    main()
