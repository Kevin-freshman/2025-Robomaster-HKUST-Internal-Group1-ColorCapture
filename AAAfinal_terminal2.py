import cv2
import numpy as np
import time
from collections import deque
import os
import serial

class OptimizedColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),
                ([170, 120, 70], [180, 255, 255])
            ],
            # --- 修改 1：绿色范围大幅放宽 ---
            # S(饱和度)下限降至30：允许发白的绿
            # V(亮度)下限降至25：关键！允许非常暗的绿色（解决左侧被识别为Black的问题）
            'Green': [([35, 30, 25], [95, 255, 255])],
            
            'Blue': [([90, 50, 50], [135, 255, 255])],
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 50, 230])],
            
            # --- 修改 2：黑色范围收紧 ---
            # V(亮度)上限降至45：只有非常黑的才算Black，把暗绿色留给Green
            'Black': [([0, 0, 0], [180, 255, 45])],
            
            # --- 修改 3：矿物范围优化 ---
            # 浅灰：S上限放宽到60，防止有点杂色的矿物漏掉
            'Mineral_LightGray': [([0, 0, 60], [180, 60, 220])],
            
            # 深灰（关键）：
            # V下限降至15：你的矿物非常黑，必须允许更暗的值
            # S上限提至90：右边的矿物看起来有点偏色，允许一定的饱和度
            'Mineral_DarkGray': [([0, 0, 15], [180, 90, 105])]
        }

        self.color_bgr = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Gold': (0, 215, 255),
            'Silver': (192, 192, 192),
            'Black': (50, 50, 50),
            'Mineral_LightGray': (180, 180, 180),  # 浅灰色
            'Mineral_DarkGray': (80, 80, 80)       # 深灰色
        }

        self.processing_scale = 0.4
        self.target_width = 1920
        self.target_height = 1080

        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()

        # 改进的形态学参数 - 更清晰的轮廓识别
        self.kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_open_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_close_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        self.processing_times = deque(maxlen=30)

        self.status_file = "color_status.txt"
        self.block_file = "block_states.txt"
        self.last_status = None

        # 串口初始化
        self.ser = None
        try:
            serial_port = 'COM3'  
            baud_rate = 9600
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"串口 {serial_port} 已成功打开，波特率 {baud_rate}")
        except serial.SerialException as e:
            print(f"打开串口时发生错误: {e}")
            print("将继续运行，但不进行串口通信。")

        print("done")
        print("target color:red green blue gold silver black")

    def optimize_frame(self, frame):
        display_frame = frame.copy()
        if self.processing_scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.processing_scale)
            new_h = int(h * self.processing_scale)
            process_frame = cv2.resize(frame, (new_w, new_h))
        else:
            process_frame = frame.copy()
        return display_frame, process_frame

    def enhance_color_detection(self, hsv, color_name):
        ranges = self.color_ranges[color_name]
        mask = None
        for lower, upper in ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            temp_mask = cv2.inRange(hsv, lower_np, upper_np)
            if mask is None:
                mask = temp_mask
            else:
                mask = cv2.bitwise_or(mask, temp_mask)
        if mask is None:
            return None
        
        # 改进的形态学处理 - 根据颜色类型使用不同的参数
        if color_name == 'Red':
            # 红色使用更强的形态学处理
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_medium)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close_medium)
        elif color_name == 'Gold':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_medium)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close_small)
        elif color_name in ['Silver', 'Black']:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_small)
        elif color_name in ['Mineral_LightGray', 'Mineral_DarkGray']:
            # 矿物使用专门的形态学处理
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close_medium)
        else:
            # 其他颜色使用标准处理
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close_small)
        return mask

    def detect_mineral_above(self, hsv_full, x, y, w, h):
        """检测红色或蓝色区域上方是否有矿物（浅灰色或深灰色）"""
        # 在色块上方区域搜索矿物
        roi_h = int(h * 1.2)
        roi_y_start = max(0, y - roi_h - 15)
        roi_y_end = max(0, y - 5)
        
        if roi_y_start >= roi_y_end:
            return False, None

        roi_hsv = hsv_full[roi_y_start:roi_y_end, x:x+w]
        
        # 检测浅灰色矿物
        light_gray_mask = self.enhance_color_detection(roi_hsv, 'Mineral_LightGray')
        # 检测深灰色矿物
        dark_gray_mask = self.enhance_color_detection(roi_hsv, 'Mineral_DarkGray')
        
        # 合并矿物mask
        mineral_mask = None
        if light_gray_mask is not None:
            mineral_mask = light_gray_mask
        if dark_gray_mask is not None:
            if mineral_mask is None:
                mineral_mask = dark_gray_mask
            else:
                mineral_mask = cv2.bitwise_or(mineral_mask, dark_gray_mask)
        
        if mineral_mask is not None:
            # 稍微减小形态学处理的核，防止把本来就小的矿物腐蚀没了
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 改小
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # 改小
            mineral_mask = cv2.morphologyEx(mineral_mask, cv2.MORPH_OPEN, kernel_open)
            mineral_mask = cv2.morphologyEx(mineral_mask, cv2.MORPH_CLOSE, kernel_close)
            
            contours, _ = cv2.findContours(mineral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                x_min, y_min, w_min, h_min = cv2.boundingRect(largest_contour)
                rect_area = w_min * h_min
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
                mineral_pixel_ratio = area / total_pixels
                
                # --- 修改处：放宽判定条件 ---
                # rectangularity > 0.4 (原0.6): 矿物如果是圆的或者不规则，矩形度会很低
                # area > 50 (原100): 远处的矿物可能很小
                if (area > 50 and 
                    rectangularity > 0.4 and 
                    0.01 < mineral_pixel_ratio < 0.6):
                    
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        center_x_global = x + center_x
                        center_y_global = roi_y_start + center_y
                        return True, (center_x_global, center_y_global)
        
        return False, None

    def update_status_file(self, status):
        if status != self.last_status:
            try:
                with open(self.status_file, 'w') as f:
                    f.write(str(status))
                self.last_status = status
            except Exception as e:
                print(f"file error: {e}")

    
    def update_block_file(self, detected_areas):
        color_priority = {'Red': 1, 'Blue': 1, 'Green': 0}

        # 过滤出红蓝绿色块
        filtered = [a for a in detected_areas if a['name'] in color_priority]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]
        top4.sort(key=lambda a: a['center'][0])

        # 获取红蓝色块（用于矿物检测）
        rb_areas = [a for a in detected_areas if a['name'] in ['Red', 'Blue']]
        rb_areas.sort(key=lambda a: a['center'][0])  # 按x坐标排序

        # 生成基础块状态
        block_states = [color_priority[a['name']] for a in top4]
        while len(block_states) < 4:
            block_states.append(-1)

        # 生成矿物状态
        mineral_states = []
        for area in rb_areas[:2]:  # 只考虑前两个红蓝色块
            mineral_states.append(1 if area.get('has_mineral', False) else 0)
        
        # 确保矿物状态有两个元素
        while len(mineral_states) < 2:
            mineral_states.append(0)

        # 合并状态
        combined_states = block_states + mineral_states

        try:
            # 1. 写入文件
            with open(self.block_file, 'w') as f:
                f.write(str(combined_states))
            print(f"updated: {combined_states}")

            # 发送到串口
            if self.ser and self.ser.is_open:
                data_to_send = f"{str(combined_states)}\n"
                self.ser.write(data_to_send.encode('utf-8'))
                print(f"已发送到串口: {data_to_send.strip()}")

        except Exception as e:
            print(f"更新block文件或发送串口时出错: {e}")

    def fast_color_detection(self, process_frame, original_dims):
        start_time = time.time()
        detected_areas = []
        
        # 改进的图像预处理
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(process_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 应用直方图均衡化增强对比度
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])
        
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]

        # 改进的轮廓检测 - 先检测所有颜色
        for color_name in self.color_ranges.keys():
            mask = self.enhance_color_detection(hsv, color_name)
            if mask is None:
                continue
                
            # 改进的轮廓查找 - 使用更精确的方法
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 400
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # 使用凸包或近似多边形来获得更清晰的轮廓
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 计算轮廓的矩形度和长宽比
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    rect_area = w * h
                    contour_rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    # 过滤掉明显不是矩形的轮廓
                    if contour_rectangularity < 0.6 or aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        continue
                    
                    hull = cv2.convexHull(approx)
                    
                    # 检测红蓝色块上方的矿物
                    has_mineral = False
                    mineral_pos = None
                    if color_name in ['Red', 'Blue']:
                        has_mineral, mineral_pos = self.detect_mineral_above(hsv, x, y, w, h)
                    
                    if self.processing_scale != 1.0:
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w = int(w * scale_x)
                        orig_h = int(h * scale_y)
                        hull = (hull * np.array([scale_x, scale_y])).astype(np.int32)
                        if mineral_pos:
                            mineral_pos = (int(mineral_pos[0] * scale_x), int(mineral_pos[1] * scale_y))
                    else:
                        orig_x, orig_y, orig_w, orig_h = x, y, w, h
                    
                    center_x = orig_x + orig_w // 2
                    center_y = orig_y + orig_h // 2
                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(orig_w, orig_h),
                        'color': self.color_bgr[color_name],
                        'contour': hull,
                        'has_mineral': has_mineral,
                        'mineral_center': mineral_pos
                    })

        self.update_block_file(detected_areas)
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        return detected_areas

    def calculate_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else fps

    def process_frame(self, frame):
        display_frame, process_frame = self.optimize_frame(frame)
        display_frame = cv2.flip(display_frame, 1)
        process_frame = cv2.flip(process_frame, 1)
        original_dims = (display_frame.shape[1], display_frame.shape[0])
        detected_areas = self.fast_color_detection(process_frame, original_dims)
        result = display_frame.copy()
        
        for area in detected_areas:
            # 绘制更清晰的轮廓
            cv2.drawContours(result, [area['contour']], -1, area['color'], 3)
            cv2.putText(result, area['name'], area['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, area['color'], 2)
            
            # 如果检测到矿物，绘制矿物标记
            if area.get('has_mineral', False):
                mineral_center = area.get('mineral_center', area['center'])
                
                # 根据矿物类型选择颜色
                mineral_color = (128, 128, 128)  # 默认灰色
                if area.get('mineral_type') == 'light':
                    mineral_color = (200, 200, 200)  # 浅灰色
                elif area.get('mineral_type') == 'dark':
                    mineral_color = (80, 80, 80)    # 深灰色
                
                # 绘制矿物矩形
                cv2.rectangle(result, 
                             (mineral_center[0]-15, mineral_center[1]-10),
                             (mineral_center[0]+15, mineral_center[1]+10),
                             mineral_color, -1)
                cv2.putText(result, "MINERAL", (mineral_center[0]-30, mineral_center[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # 绘制从矿物到色块的箭头
                cv2.arrowedLine(result, mineral_center, area['center'], 
                               mineral_color, 2)

        return result, len(detected_areas)

    def add_performance_info(self, frame, detected_count):
        h, w = frame.shape[:2]
        fps = self.calculate_fps()
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Detected: {detected_count}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Process: {avg_processing_time:.1f}ms",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def main():
    detector = OptimizedColorDetector()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    screenshot_count = 0
    try:
        while True:
            ret, frame0 = cap.read()
            frame = cv2.flip(frame0, 1)
            if not ret:
                print("Failed to grab frame!")
                break
            result, count = detector.process_frame(frame)
            detector.add_performance_info(result, count)
            cv2.imshow(' Optimized Color Detection System', result)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
            elif key in [ord('s'), ord('S')]:
                screenshot_count += 1
                filename = f'color_detection_{screenshot_count}.png'
                cv2.imwrite(filename, result)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+'):
                detector.processing_scale = min(1.0, detector.processing_scale + 0.1)
                print(f"Scale increased to: {detector.processing_scale}")
            elif key == ord('-'):
                detector.processing_scale = max(0.2, detector.processing_scale - 0.1)
                print(f"Scale decreased to: {detector.processing_scale}")

    except KeyboardInterrupt:
        print("User interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if detector.ser and detector.ser.is_open:
            detector.ser.close()
            print("串口已关闭。")

        print("Program exited")


if __name__ == "__main__":
    main()