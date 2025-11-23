# 识别目标：
# 绿色塑料板
# 红色/蓝色 塑料板
# 其他若干颜色矿石
# 本文件仅适用红色图像识别测试





import cv2
import numpy as np


def main():

    
    image = cv2.imread('picture.jpg')

    # 转换HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色
    lower_red = np.array([0, 100, 100])    # 红色下限
    upper_red = np.array([10, 255, 255])   # 红色上限
    lower_red2 = np.array([160, 100, 100]) # 红色2下限
    upper_red2 = np.array([179, 255, 255]) # 红色2上限

    # 掩码 检测红色区域
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 应用掩码到原图像，只保留红色部分
    result = cv2.bitwise_and(image, image, mask=mask)

    # 标注
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤小区域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色矩形标注
            cv2.putText(image, 'Red Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 3. 输出
    cv2.imwrite('picture_tagged.jpg', image)

    print("the tagged picture is saved as picture_tagged.jpg")





if __name__ =="__main__":
    main()