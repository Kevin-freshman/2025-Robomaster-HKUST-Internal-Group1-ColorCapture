# 识别目标：
# 绿色塑料板
# 红色/蓝色 塑料板
# 其他若干颜色矿石
# 本文件仅适用红色图像识别测试





import cv2
import numpy as np


def main():

    # 1. 读取文件中的照片
    image = cv2.imread('picture.jpg')

    # 转换为HSV颜色空间，便于颜色检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义要检测的颜色范围，例如红色（HSV范围：下限和上限）
    lower_red = np.array([0, 100, 100])    # 红色下限
    upper_red = np.array([10, 255, 255])   # 红色上限（注意：红色在HSV中可能需要两个范围，因为它跨越0度）
    lower_red2 = np.array([160, 100, 100]) # 第二个红色范围下限
    upper_red2 = np.array([179, 255, 255]) # 第二个红色范围上限

    # 创建掩码，检测红色区域
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 应用掩码到原图像，只保留红色部分
    result = cv2.bitwise_and(image, image, mask=mask)

    # 在原图像上标注检测到的颜色区域（例如，绘制轮廓）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤小区域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色矩形标注
            cv2.putText(image, 'Red Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 3. 输出已经标注好颜色的图片（原图像上标注）
    cv2.imwrite('picture_tagged.jpg', image)

    print("the tagged picture is saved as picture_tagged.jpg")





if __name__ =="__main__":
    main()