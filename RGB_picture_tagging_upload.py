import cv2
import numpy as np

image_path = 'picture_test/forest.jpg'  
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

# 颜色列表
colors = {
    'Red': [(lower_red1, upper_red1), (lower_red2, upper_red2)],
    'Green': [(lower_green, upper_green)],
    'Blue': [(lower_blue, upper_blue)]
}

# 定义浅色
overlay_colors = {
    'Red': (0, 0, 255),    # 红色
    'Green': (0, 255, 0),  # 绿色
    'Blue': (255, 0, 0)    # 蓝色
}

# 标注颜色
annotation_colors = {
    'Red': (0, 0, 255),    # 红色
    'Green': (0, 255, 0),  # 绿色
    'Blue': (255, 0, 0)    # 蓝色
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
        if cv2.contourArea(contour) > 500:  # 过滤
            # 浅色叠加层
            overlay = np.zeros_like(image)
            cv2.drawContours(overlay, [contour], -1, overlay_colors[color_name], thickness=cv2.FILLED)
            
            # 叠加
            image = cv2.addWeighted(image, 1.0, overlay, 0.3, 0)
            
            # 计算
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image, f'{color_name}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, annotation_colors[color_name], 1)

# 输出
output_path = 'forest_tagged3.jpg'
cv2.imwrite(output_path, image)

print(f"图片处理完成，已保存为 {output_path}")