# 这是我(张宸宇)之前的python小项目
# 调用了yolo算法和利用电脑的前置摄像头来实现日常物品的识别
# 先上传在此做一个之后AR图像识别的template
#testing in changing
#new test

import torch
import cv2


modelzcy = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)



# 打开摄像头
cap = cv2.VideoCapture(0)
print("📷 摄像头打开，按 'q' 退出")

ret, frame = cap.read()
if not ret or frame is None:
    print("错误：无法读取有效帧")
    
print(f"帧尺寸: {frame.shape if frame is not None else 'None'}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推理（frame是BGR，自动转换）
    results = modelzcy(frame)

    # 画框 & 标签（可选 render）
    annotated_frame = results.render()[0]

    # 显示画面
    cv2.imshow('YOLOv5 目标检测', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
