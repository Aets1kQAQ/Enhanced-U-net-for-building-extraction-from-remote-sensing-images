from PIL import Image
import os


# 切割图片并存储到输入图片所在目录的子文件夹中
def split_image(image_path, tile_size=(500, 500)):
    # 打开图片
    img = Image.open(image_path)

    # 获取图片的宽度和高度
    img_width, img_height = img.size

    # 计算列和行的数量
    tile_width, tile_height = tile_size
    cols = img_width // tile_width
    rows = img_height // tile_height

    # 获取图片所在的目录和文件名（不带后缀的）
    image_dir = os.path.dirname(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 在图片目录中创建一个新的文件夹用于存储切片
    output_dir = os.path.join(image_dir, f"{image_name}_tiles")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 开始切割图片
    for row in range(rows):
        for col in range(cols):
            # 计算每一个切片的左上角和右下角坐标
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # 切割图片
            cropped_img = img.crop((left, upper, right, lower))

            # 生成文件名，保存切割后的图片
            cropped_img_name = f"tile_{row}_{col}.tif"
            cropped_img.save(os.path.join(output_dir, cropped_img_name))
            print(f"Saved {cropped_img_name} in {output_dir}")


# 遍历文件夹，处理所有图片
def process_images_in_folder(folder_path, tile_size=(500, 500)):
    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是图片
        if os.path.isfile(file_path) and filename.lower().endswith(supported_formats):
            print(f"Processing image: {file_path}")
            try:
                split_image(file_path, tile_size)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")


# 示例用法
folder_path = 'Massachusetts/images'  # 替换为你的文件夹路径
process_images_in_folder(folder_path)