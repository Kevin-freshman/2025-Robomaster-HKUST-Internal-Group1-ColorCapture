import cv2
import numpy as np

# 1. 跨文件夹读取文件（假设图片在上级目录的images文件夹中，你可以根据实际情况修改路径）
# 例如：如果程序在 /code/main.py，图片在 /images/picture.jpg，则路径为 '../images/picture.jpg'
image_path = 'picture_test/forest.jpg'  # 请替换为实际相对或绝对路径
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# 转换为HSV颜色空间，便于颜色检测
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义要检测的颜色范围
# 红色（两个范围）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# 绿色
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

# 蓝色
lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])

# 颜色列表，便于循环处理
colors = {
    'Red': [(lower_red1, upper_red1), (lower_red2, upper_red2)],
    'Green': [(lower_green, upper_green)],
    'Blue': [(lower_blue, upper_blue)]
}

# 定义浅色叠加层（BGR格式，浅色版本，半透明）
overlay_colors = {
    'Red': (0, 0, 255),    # 红色
    'Green': (0, 255, 0),  # 绿色
    'Blue': (255, 0, 0)    # 蓝色
}

# 标注颜色：使用对应颜色的浅色叠加，并用更美观的字体
annotation_colors = {
    'Red': (0, 0, 255),    # BGR: 红色
    'Green': (0, 255, 0),  # BGR: 绿色
    'Blue': (255, 0, 0)    # BGR: 蓝色
}

for color_name, ranges in colors.items():
    mask = None
    for lower, upper in ranges:
        partial_mask = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = partial_mask
        else:
            mask = cv2.bitwise_or(mask, partial_mask)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤小区域
            # 创建浅色叠加层
            overlay = np.zeros_like(image)
            cv2.drawContours(overlay, [contour], -1, overlay_colors[color_name], thickness=cv2.FILLED)
            
            # 叠加到原图像（alpha=0.3表示30%不透明，浅色效果）
            image = cv2.addWeighted(image, 1.0, overlay, 0.3, 0)
            
            # 计算边界框以放置文本
            x, y, w, h = cv2.boundingRect(contour)
            
            # 使用更美观的字体：FONT_HERSHEY_COMPLEX，小尺寸scale=0.6，厚度=1
            cv2.putText(image, f'{color_name}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, annotation_colors[color_name], 1)

# 3. 输出已经标注好颜色的图片
output_path = 'forest_tagged3.jpg'
cv2.imwrite(output_path, image)

print(f"图片处理完成，已保存为 {output_path}")