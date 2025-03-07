from PIL import Image
import os

# 定义函数：检查像素是否为非黑色
def is_non_black(pixel):
    r, g, b = pixel
    # 如果像素不是黑色 (0, 0, 0)，则返回 True
    return not (r == 0 and g == 0 and b == 0)

# 定义函数：将非黑色像素替换为白色
def replace_non_black_with_white(image):
    # 将图片转换为RGB模式
    image = image.convert("RGB")
    pixels = image.load()  # 获取像素数据

    for i in range(image.size[0]):  # 遍历图片的宽度
        for j in range(image.size[1]):  # 遍历图片的高度
            if is_non_black(pixels[i, j]):
                pixels[i, j] = (255, 255, 255)  # 将非黑色像素替换为白色

    return image

# 定义函数：处理文件夹中的所有图片
def process_images_in_folder(folder_path, output_folder):
    # 如果输出文件夹不存在，创建文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 确保只处理图片文件，可以根据需要扩展支持的格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}...")

            # 打开图片
            with Image.open(image_path) as img:
                # 将非黑色替换为白色
                new_img = replace_non_black_with_white(img)

                # 保存处理后的图片到输出文件夹
                output_path = os.path.join(output_folder, filename)
                new_img.save(output_path)
                print(f"Saved processed image to {output_path}")

# 主程序：指定输入和输出文件夹路径
if __name__ == "__main__":
    input_folder = "INRIA(Before)/JPEGImages"  # 输入文件夹路径（INRIA(Before)/SegmentationClass）
    output_folder = "INRIA/JPEGImages"  # 输出文件夹路径（INRIA/SegmentationClass）
    process_images_in_folder(input_folder, output_folder)