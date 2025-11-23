import cv2
import numpy as np
import time
from collections import deque

class OptimizedColorDetector:
    def __init__(self):
        # 优化的颜色范围 (HSV格式)
        self.color_ranges = {
            'Red': [
                ([0, 150, 100], [8, 255, 255]),      # 红色范围1 - 调整参数
                ([172, 150, 100], [180, 255, 255])   # 红色范围2 - 调整参数
            ],
            'Blue': [([100, 150, 80], [130, 255, 255])],
            'Green': [([40, 80, 80], [80, 255, 255])],
            'Yellow': [([22, 100, 100], [32, 255, 255])],
            'Purple': [([130, 80, 80], [160, 255, 255])],
            'Orange': [([10, 150, 100], [18, 255, 255])],
            'Cyan': [([85, 100, 80], [100, 255, 255])]
        }
        
        # 颜色对应的BGR值
        self.color_bgr = {
            'Red': (0, 0, 255),
            'Blue': (255, 0, 0),
            'Green': (0, 255, 0),
            'Yellow': (0, 255, 255),
            'Purple': (255, 0, 255),
            'Orange': (0, 165, 255),
            'Cyan': (255, 255, 0)
        }
        
        # FPS计算
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()
        
        # 性能优化设置
        self.processing_scale = 0.5  # 处理时的缩放比例，提高帧率
        self.target_width = 1920
        self.target_height = 1080
        
        # 预处理kernel
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # 创建渐变色背景
        self.gradient_bg = self.create_gradient_bg(self.target_width, self.target_height)
        
        # 性能统计
        self.processing_times = deque(maxlen=30)
        
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
        """绘制美观的文字"""
        font = cv2.FONT_HERSHEY_COMPLEX
        
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
    
    def optimize_frame(self, frame):
        """优化帧处理性能 - 降低处理分辨率但保持显示分辨率"""
        # 保持原始帧用于显示
        display_frame = frame.copy()
        
        # 创建用于处理的缩小版本
        if self.processing_scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.processing_scale)
            new_h = int(h * self.processing_scale)
            process_frame = cv2.resize(frame, (new_w, new_h))
        else:
            process_frame = frame.copy()
            
        return display_frame, process_frame
    
    def enhance_red_detection(self, hsv):
        """专门增强红色检测"""
        # 红色在HSV中有两个范围
        red_lower1 = np.array([0, 150, 100])
        red_upper1 = np.array([8, 255, 255])
        red_lower2 = np.array([172, 150, 100])
        red_upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 对红色使用更强的形态学操作
        kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_red)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_red)
        
        return red_mask
    
    def advanced_color_detection(self, frame, hsv):
        """使用更高级的颜色检测方法"""
        start_time = time.time()
        detected_areas = []
        
        # 专门处理红色
        red_mask = self.enhance_red_detection(hsv)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 800:  # 红色使用更低的面积阈值
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                
                # 调整坐标到原始尺寸
                if self.processing_scale != 1.0:
                    scale = 1.0 / self.processing_scale
                    x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
                    hull = (hull * scale).astype(np.int32)
                
                center_x = x + w // 2
                center_y = y + h // 2
                detected_areas.append({
                    'name': 'Red',
                    'center': (center_x, center_y),
                    'size': max(w, h),
                    'color': self.color_bgr['Red'],
                    'contour': hull
                })
        
        # 处理其他颜色
        for color_name, ranges in self.color_ranges.items():
            if color_name == 'Red':  # 红色已经处理过
                continue
                
            combined_mask = None
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
                
            # 形态学操作
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = 1000  # 降低阈值以检测更小的区域
                
                if area > min_area:
                    hull = cv2.convexHull(contour)
                    x, y, w, h = cv2.boundingRect(hull)
                    
                    # 调整坐标到原始尺寸
                    if self.processing_scale != 1.0:
                        scale = 1.0 / self.processing_scale
                        x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
                        hull = (hull * scale).astype(np.int32)
                    
                    perimeter = cv2.arcLength(hull, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:
                            continue
                    
                    center_x = x + w // 2
                    center_y = y + h // 2
                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(w, h),
                        'color': self.color_bgr[color_name],
                        'contour': hull
                    })
        
        # 记录处理时间
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        self.processing_times.append(processing_time)
        
        return detected_areas
    
    def create_rounded_mask(self, contours, img_shape):
        """创建圆角掩码"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for contour in contours:
            if len(contour) > 2:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(mask, [approx], 255)
        
        # 应用高斯模糊让边缘更柔和
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask
    
    def detect_colors(self, frame):
        """检测颜色并返回处理后的图像"""
        # 分离显示帧和处理帧
        display_frame, process_frame = self.optimize_frame(frame)
        
        # 水平翻转
        display_frame = cv2.flip(display_frame, 1)
        process_frame = cv2.flip(process_frame, 1)
        
        # 转换为HSV
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
        
        # 对HSV图像进行高斯模糊，减少噪声
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        
        # 创建结果图像
        result = self.gradient_bg.copy()
        h_display, w_display = display_frame.shape[:2]
        if (w_display, h_display) != (self.target_width, self.target_height):
            result = cv2.resize(result, (w_display, h_display))
        
        # 创建一个透明层用于颜色涂抹
        color_layer = np.zeros_like(display_frame, dtype=np.uint8)
        
        # 使用高级颜色检测
        detected_areas = self.advanced_color_detection(process_frame, hsv)
        
        # 处理每个检测到的区域
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            contour = area['contour']
            
            # 创建圆角掩码
            contour_mask = self.create_rounded_mask([contour], display_frame.shape)
            
            # 获取浅色
            light_color = self.get_light_color(color, alpha=0.4)
            
            # 在颜色层上涂抹
            colored_area = np.full_like(display_frame, light_color, dtype=np.uint8)
            color_layer[contour_mask > 0] = colored_area[contour_mask > 0]
        
        # 将颜色层与原图混合
        result = cv2.addWeighted(display_frame, 0.7, color_layer, 0.3, 0)
        
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
        
        # 计算平均处理时间
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # 添加标题栏
        title_bg = np.zeros((60, w, 3), dtype=np.uint8)
        title_bg[:, :] = (40, 40, 60)
        frame[0:60, 0:w] = cv2.addWeighted(frame[0:60, 0:w], 0.3, title_bg, 0.7, 0)
        
        # 添加标题
        self.put_pretty_text(frame, "Optimized Color Detection", (20, 40), 
                           (255, 255, 255), 1.2, 2)
        
        # 添加状态栏
        status_bg = np.zeros((40, w, 3), dtype=np.uint8)
        status_bg[:, :] = (30, 30, 30)
        frame[h-40:h, 0:w] = cv2.addWeighted(frame[h-40:h, 0:w], 0.4, status_bg, 0.6, 0)
        
        # 添加状态信息
        status_text = f"Detected: {detected_count} colors | Q:Exit | S:Screenshot | +/-:Scale({self.processing_scale})"
        self.put_pretty_text(frame, status_text, (20, h-10), 
                           (200, 200, 200), 0.6, 1)
        
        # 添加性能信息
        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
        self.put_pretty_text(frame, fps_text, (w - fps_size[0] - 20, 40), 
                           (0, 255, 255), 0.6, 1)
        
        # 添加处理时间信息
        time_text = f"Process: {avg_processing_time:.1f}ms"
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, time_text, (w - time_size[0] - 20, 70), 
                           (0, 200, 255), 0.5, 1)
        
        # 添加分辨率信息
        res_text = f"Resolution: {w}x{h}"
        res_size = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, res_text, (w - res_size[0] - 20, 95), 
                           (0, 200, 255), 0.5, 1)

def main():
    # 初始化检测器
    detector = OptimizedColorDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    # 尝试设置高分辨率和高帧率
    print("Setting camera to maximum resolution and frame rate...")
    
    # 尝试设置高分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    
    # 尝试设置高帧率
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 获取实际设置的值
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")
    print(f"Processing scale: {detector.processing_scale}")
    print("Press '+' to increase processing scale (more accurate)")
    print("Press '-' to decrease processing scale (faster)")
    
    # 如果实际分辨率低于目标，更新检测器的目标分辨率
    if actual_width != detector.target_width or actual_height != detector.target_height:
        detector.target_width = actual_width
        detector.target_height = actual_height
        detector.gradient_bg = detector.create_gradient_bg(actual_width, actual_height)
        print(f"Adjusted target resolution to: {actual_width}x{actual_height}")
    
    print("Optimized color detection started!")
    print("Press 'Q' to exit")
    print("Press 'S' to save screenshot")
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break
        
        # 检测颜色
        result, count = detector.detect_colors(frame)
        
        # 添加UI元素
        detector.add_ui_elements(result, count)
        
        # 显示结果
        cv2.imshow('Optimized Color Detection', result)
        
        # 键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # 保存截图
            screenshot_count += 1
            filename = f'color_detection_{screenshot_count}.png'
            cv2.imwrite(filename, result)
            print(f"Screenshot saved: {filename}")
        elif key == ord('+') or key == ord('='):
            # 增加处理比例（更精确）
            detector.processing_scale = min(1.0, detector.processing_scale + 0.1)
            print(f"Processing scale increased to: {detector.processing_scale}")
        elif key == ord('-') or key == ord('_'):
            # 减少处理比例（更快）
            detector.processing_scale = max(0.3, detector.processing_scale - 0.1)
            print(f"Processing scale decreased to: {detector.processing_scale}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == "__main__":
    main()