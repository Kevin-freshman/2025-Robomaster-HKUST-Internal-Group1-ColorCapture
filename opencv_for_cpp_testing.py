import cv2
import os
print(os.path.dirname(cv2.__file__))

# 假设你的 OpenCV 安装在以下路径（请根据实际情况调整）
g++ -std=c++11 color_detection.cpp -o color_detection ^
-I "C:\Users\29488\AppData\Local\Programs\Python\Python313\Lib\site-packages\cv2" ^
-L "C:\Users\29488\AppData\Local\Programs\Python\Python313\Lib\site-packages\cv2" ^
-lopencv_core4xx ^
-lopencv_imgproc4xx ^
-lopencv_highgui4xx ^
-lopencv_videoio4xx