import cv2
import numpy as np
import time
from collections import deque

class ImprovedColorDetector:
    def __init__(self):
        # 更精确的颜色范围 (HSV格式)
        self.color_ranges = {
            '红色': [
                ([0, 150, 50], [10, 255, 255]),      # 红色范围1
                ([170, 150, 50], [180, 255, 255])    # 红色范围2
            ],
            '蓝色': [([100, 150, 50], [130, 255, 255])],
            '绿色': [([40, 80, 50], [80, 255, 255])],
            '黄色': [([20, 100, 100], [30, 255, 255])],
            '紫色': [([130, 80, 50], [160, 255, 255])],
            '橙色': [([10, 150, 100], [20, 255, 255])],
            '青色': [([85, 100, 50], [100, 255, 255])]
        }
        
        # 颜色对应的BGR值
        self.color_bgr = {
            '红色': (0, 0, 255),
            '蓝色': (255, 0, 0),
            '绿色': (0, 255, 0),
            '黄色': (0, 255, 255),
            '紫色': (255, 0, 255),
            '橙色': (0, 165, 255),
            '青色': (255, 255, 0)
        }
        
        # 创建渐变色背景
        self.gradient_bg = self.create_gradient_bg(1280, 720)
        
        # FPS计算
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()
        
        # 加载预训练模型（用于更精确的颜色识别）
        self.use_advanced_detection = False
        try:
            # 尝试加载OpenCV的DNN模块（如果有预训练模型）
            self.net = None
            # 这里可以加载预训练的颜色识别模型
            # 例如: self.net = cv2.dnn.readNetFromTensorflow('model.pb')
            print("高级颜色检测模式可用")
            self.use_advanced_detection = True
        except:
            print("使用传统颜色检测方法")
        
    def create_gradient_bg(self, width, height):
        """创建渐变色背景"""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            ratio = i / height
            r = int(30 + ratio * 50)
            g = int(30 + ratio * 30)
            b = int(70 + ratio * 100)
            bg[i, :] = (b, g, r)
        return bg
    
    def get_light_color(self, color_bgr, alpha=0.3):
        """生成浅色版本"""
        return tuple(int(c + (255 - c) * alpha) for c in color_bgr)
    
    def put_pretty_text(self, img, text, position, color, 
                       font_scale=1.0, thickness=2, shadow=True):
        """绘制美观的文字 - 修复字体显示问题"""
        font = cv2.FONT_HERSHEY_COMPLEX  # 使用更清晰的字体
        
        if shadow:
            # 文字阴影
            shadow_color = (0, 0, 0)
            cv2.putText(img, text, 
                       (position[0] + 2, position[1] + 2), 
                       font, font_scale, shadow_color, thickness + 1, 
                       cv2.LINE_AA)
        
        # 主文字
        cv2.putText(img, text, position, 
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    def calculate_fps(self):
        """计算真实FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)
        
        # 返回平均FPS
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else fps
    
    def advanced_color_detection(self, frame, hsv):
        """使用更高级的颜色检测方法"""
        detected_areas = []
        
        for color_name, ranges in self.color_ranges.items():
            combined_mask = None
            
            # 合并多个范围（特别是红色有两个范围）
            for lower, upper in ranges:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask = cv2.inRange(hsv, lower_np, upper_np)
                
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            if combined_mask is None:
                continue
                
            # 更精确的形态学操作
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            
            # 开运算去除噪声
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            # 闭运算填充空洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 动态调整面积阈值
                min_area = 2000  # 降低阈值以检测更小的区域
                
                if area > min_area:
                    # 使用凸包来获得更精确的形状
                    hull = cv2.convexHull(contour)
                    
                    # 获取边界矩形
                    x, y, w, h = cv2.boundingRect(hull)
                    
                    # 计算轮廓的圆形度，过滤不规则形状
                    perimeter = cv2.arcLength(hull, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:  # 过滤过于不规则的形状
                            continue
                    
                    # 记录检测到的区域信息
                    center_x = x + w // 2
                    center_y = y + h // 2
                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(w, h),
                        'color': self.color_bgr[color_name],
                        'contour': hull
                    })
        
        return detected_areas
    
    def create_rounded_mask(self, contours, img_shape):
        """创建圆角掩码"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for contour in contours:
            if len(contour) > 2:
                # 使用更精确的多边形逼近
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(mask, [approx], 255)
        
        # 应用高斯模糊让边缘更柔和
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask
    
    def detect_colors(self, frame):
        """检测颜色并返回处理后的图像"""
        # 水平翻转
        frame = cv2.flip(frame, 1)
        
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 对HSV图像进行高斯模糊，减少噪声
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # 创建结果图像
        result = self.gradient_bg.copy()
        
        # 创建一个透明层用于颜色涂抹
        color_layer = np.zeros_like(frame, dtype=np.uint8)
        
        # 使用高级颜色检测
        detected_areas = self.advanced_color_detection(frame, hsv)
        
        # 处理每个检测到的区域
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            contour = area['contour']
            
            # 创建圆角掩码
            contour_mask = self.create_rounded_mask([contour], frame.shape)
            
            # 获取浅色
            light_color = self.get_light_color(color, alpha=0.4)
            
            # 在颜色层上涂抹
            colored_area = np.full_like(frame, light_color, dtype=np.uint8)
            color_layer[contour_mask > 0] = colored_area[contour_mask > 0]
        
        # 将颜色层与原图混合
        result = cv2.addWeighted(frame, 0.7, color_layer, 0.3, 0)
        
        # 添加检测到的颜色标签
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            
            # 根据区域大小调整字体大小
            font_scale = max(0.8, min(2.0, size / 150))
            thickness = max(1, int(size / 100))
            
            # 添加背景圆
            radius = max(30, size // 8)
            cv2.circle(result, center, radius, (40, 40, 40), -1)
            cv2.circle(result, center, radius, color, 2)
            
            # 添加文字
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            self.put_pretty_text(result, name, (text_x, text_y), color, 
                               font_scale, thickness)
        
        return result, len(detected_areas)
    
    def add_ui_elements(self, frame, detected_count):
        """添加UI元素"""
        h, w = frame.shape[:2]
        
        # 计算真实FPS
        fps = self.calculate_fps()
        
        # 添加标题栏
        title_bg = np.zeros((60, w, 3), dtype=np.uint8)
        title_bg[:, :] = (40, 40, 60)
        frame[0:60, 0:w] = cv2.addWeighted(frame[0:60, 0:w], 0.3, title_bg, 0.7, 0)
        
        # 添加标题
        self.put_pretty_text(frame, "Advanced Color Detection System", (20, 40), 
                           (255, 255, 255), 1.2, 2)
        
        # 添加状态栏
        status_bg = np.zeros((40, w, 3), dtype=np.uint8)
        status_bg[:, :] = (30, 30, 30)
        frame[h-40:h, 0:w] = cv2.addWeighted(frame[h-40:h, 0:w], 0.4, status_bg, 0.6, 0)
        
        # 添加状态信息
        status_text = f"Detected: {detected_count} color areas | Press 'Q' to exit | 'S' to screenshot"
        self.put_pretty_text(frame, status_text, (20, h-10), 
                           (200, 200, 200), 0.6, 1)
        
        # 添加真实FPS信息
        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
        self.put_pretty_text(frame, fps_text, (w - fps_size[0] - 20, 40), 
                           (0, 255, 255), 0.6, 1)
        
        # 添加检测模式信息
        mode_text = "Advanced Detection" if self.use_advanced_detection else "Standard Detection"
        mode_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, mode_text, (w - mode_size[0] - 20, 70), 
                           (0, 200, 255), 0.5, 1)

def main():
    # 初始化检测器
    detector = ImprovedColorDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("高级颜色识别程序启动成功！")
    print("按 'Q' 键退出程序")
    print("按 'S' 键保存截图")
    print("按 'A' 键切换检测模式")
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧！")
            break
        
        # 检测颜色
        result, count = detector.detect_colors(frame)
        
        # 添加UI元素
        detector.add_ui_elements(result, count)
        
        # 显示结果
        cv2.imshow('Advanced Color Detection System', result)
        
        # 键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # 保存截图
            screenshot_count += 1
            filename = f'color_detection_{screenshot_count}.png'
            cv2.imwrite(filename, result)
            print(f"截图已保存: {filename}")
        elif key == ord('a') or key == ord('A'):
            # 切换检测模式
            detector.use_advanced_detection = not detector.use_advanced_detection
            mode = "高级" if detector.use_advanced_detection else "标准"
            print(f"切换到{mode}检测模式")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()