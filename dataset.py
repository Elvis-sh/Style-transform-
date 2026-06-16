"""
训练数据集加载器

支持:
  1. COCO 2017 数据集（自动下载）
  2. ImageNet 子集
  3. 自定义图片文件夹

训练时随机裁剪 256×256 的 patch。
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import tarfile


class ImageFolderDataset(Dataset):
    """从任意图片文件夹加载训练数据。"""

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
    ):
        """
        Args:
            root_dir: 图片文件夹路径
            image_size: 训练 patch 尺寸
        """
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_paths.append(os.path.join(root, f))

        if not self.image_paths:
            raise RuntimeError(f"在 {root_dir} 中未找到任何图片！")

        self.transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),          # [0, 1]
            transforms.Lambda(lambda x: x * 2 - 1),  # → [-1, 1] for tanh
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def download_coco_subset(target_dir: str = "train_data", num_images: int = 5000):
    """
    下载 COCO 2017 验证集的一个子集作为训练数据。
    COCO val2017 有 5000 张图片，约 1GB，适合快速训练。

    Args:
        target_dir: 下载目标目录
        num_images: 使用图片数量（最多 5000）

    Returns:
        图片文件夹路径
    """
    import glob

    image_dir = os.path.join(target_dir, "coco_val2017")
    os.makedirs(target_dir, exist_ok=True)

    # 如果已经解压，直接返回
    if os.path.isdir(image_dir) and len(os.listdir(image_dir)) >= num_images:
        print(f"COCO 数据集已存在于: {image_dir}")
        return image_dir

    # 下载 COCO 2017 val 图片 (约 1GB)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(target_dir, "val2017.zip")

    if not os.path.isfile(zip_path):
        print(f"正在下载 COCO 2017 验证集 (~1GB)...")
        print(f"URL: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"自动下载失败: {e}")
            print("\n请手动下载数据集并放入 train_data/ 文件夹，或使用自定义图片文件夹:")
            print(f"  mkdir -p {target_dir}/my_images")
            print(f"  将你的训练图片复制到 {target_dir}/my_images/")
            raise

    # 解压
    if not os.path.isdir(image_dir):
        print("正在解压...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            image_files = [f for f in zf.namelist() if f.endswith(".jpg")][:num_images]
            for f in image_files:
                zf.extract(f, target_dir)

    print(f"数据集准备完成: {image_dir} ({len(os.listdir(image_dir))} 张图片)")
    return image_dir


def create_dataloader(
    image_dir: str,
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    创建训练 DataLoader。

    Args:
        image_dir: 图片文件夹路径
        batch_size: 批次大小（CPU 上建议 1-2）
        image_size: 训练 patch 尺寸
        num_workers: 数据加载线程数
        shuffle: 是否随机打乱

    Returns:
        DataLoader
    """
    dataset = ImageFolderDataset(image_dir, image_size=image_size)
    print(f"数据集: {len(dataset)} 张图片, batch_size={batch_size}, size={image_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
