from PIL import Image
import os


# 将单张tif图片转换为png
def convert_tif_to_png(tif_image_path, output_folder):
    # 打开tif图片
    img = Image.open(tif_image_path)

    # 获取文件名，不带扩展名
    image_name = os.path.splitext(os.path.basename(tif_image_path))[0]

    # 构造输出的png文件路径
    png_image_path = os.path.join(output_folder, f"{image_name}.png")

    # 保存为png格式
    img.save(png_image_path, "PNG")
    print(f"Converted {tif_image_path} to {png_image_path}")


# 遍历文件夹中的所有tif图片并转换为png
def convert_all_tif_in_folder(input_folder, output_folder):
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构造完整文件路径
        file_path = os.path.join(input_folder, filename)

        # 检查是否是tif文件
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            try:
                # 转换tif为png
                convert_tif_to_png(file_path, output_folder)
            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")


# 示例用法
input_folder = 'Massachusetts/labels'  # 输入tif图片的文件夹路径
output_folder = 'Massachusetts/labels（png）'  # 输出png图片的文件夹路径
convert_all_tif_in_folder(input_folder, output_folder)