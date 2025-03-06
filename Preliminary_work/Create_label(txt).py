import os
import random

def generate_train_txt(image_dir, output_path):
    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.tif')]

    with open(output_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")


image_dir = 'data/train/building/JPEGImages'  # 替换为你的图像目录
output_path = 'data/train/building/ImageSets/Segmentation/train.txt'  # 输出文件路径
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录（如果不存在）
generate_train_txt(image_dir, output_path)


def generate_val_txt(image_dir, output_path, val_split=0.2):
    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.tif')]
    val_count = int(len(filenames) * val_split)
    val_files = random.sample(filenames, val_count)

    with open(output_path, 'w') as f:
        for filename in val_files:
            f.write(f"{filename}\n")


image_dir = 'data/val/building/SegmentationClass'  # 替换为你的图像目录
output_path = 'data/val/building/ImageSets/Segmentation/val.txt'  # 输出文件路径
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录（如果不存在）
generate_val_txt(image_dir, output_path)


def generate_train_txt(image_dir, output_path):
    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.tif')]

    with open(output_path, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")


image_dir = 'data/test/building/JPEGImages'  # 替换为你的图像目录
output_path = 'data/test/building/ImageSets/Segmentation/test.txt'  # 输出文件路径
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录（如果不存在）
generate_train_txt(image_dir, output_path)