import cv2
import numpy as np
import time
from collections import deque
import os
#import flip from cv2
import serial  # --- 新增 ---

class OptimizedColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),
                ([170, 120, 70], [180, 255, 255])
            ],
            'Green': [([35, 50, 50], [95, 255, 255])],
            'Blue': [([90, 50, 50], [135, 255, 255])],
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 50, 230])],
            'Black': [([0, 0, 0], [180, 255, 60])],
            'Mineral_Gray': [([0, 0, 20], [180, 80, 220])]  # 新增矿物灰色范围
        }

        self.color_bgr = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Gold': (0, 215, 255),
            'Silver': (192, 192, 192),
            'Black': (50, 50, 50),
            'Mineral_Gray': (128, 128, 128)  # 新增矿物颜色
        }

        self.processing_scale = 0.4
        self.target_width = 1920
        self.target_height = 1080

        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        self.kernel_gold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.processing_times = deque(maxlen=30)

        self.status_file = "color_status.txt"
        self.block_file = "block_states.txt"
        self.last_status = None

        # --- 新增: 初始化串口 ---
        # 串口初始化 - 自动扫描 COM1 到 COM19
        self.ser = None
        baud_rate = 9600
        print("正在自动扫描可用串口 (COM10 - COM19)...")

        for i in range(10, 20): # 扫描 1 到 19
            port_name = f'COM{i}'
            try:
                # 尝试打开串口
                # write_timeout 也很重要，防止发送数据时卡死
                temp_ser = serial.Serial(port_name, baud_rate, timeout=1, write_timeout=1)
                
                # 如果上面这行没报错，说明连接成功
                self.ser = temp_ser
                print(f"--------------------------------")
                print(f"✅ 成功连接到串口: {port_name}")
                print(f"   波特率: {baud_rate}")
                print(f"--------------------------------")
                break # 找到一个可用的就退出循环
            except serial.SerialException:
                # 如果端口被占用或不存在，直接跳过，不报错
                continue

        if self.ser is None:
            print("⚠️ 未找到可用的串口 (COM1-COM19)")
            print("程序将继续运行，但不进行串口通信。")

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
        if color_name == 'Red':
            kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_red)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_red)
        elif color_name == 'Gold':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_gold)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_gold)
        elif color_name in ['Silver', 'Black']:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        else:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    def detect_mineral_above(self, hsv_full, x, y, w, h):
        """检测红色或蓝色区域上方是否有矿物（灰色或黑色）"""
        # 在色块上方区域搜索矿物
        roi_h = int(h * 0.8)  # 搜索高度为色块高度的0.8倍
        roi_y_start = max(0, y - roi_h - 10)  # 从色块上方10像素开始
        roi_y_end = max(0, y - 5)  # 到色块上方5像素结束
        
        if roi_y_start >= roi_y_end:
            return False, None

        roi_hsv = hsv_full[roi_y_start:roi_y_end, x:x+w]
        
        # 检测灰色矿物
        gray_mask = self.enhance_color_detection(roi_hsv, 'Mineral_Gray')
        # 检测黑色矿物
        black_mask = self.enhance_color_detection(roi_hsv, 'Black')
        
        # 合并矿物mask
        mineral_mask = None
        if gray_mask is not None:
            mineral_mask = gray_mask
        if black_mask is not None:
            if mineral_mask is None:
                mineral_mask = black_mask
            else:
                mineral_mask = cv2.bitwise_or(mineral_mask, black_mask)
        
        if mineral_mask is not None:
            # 对矿物mask进行形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mineral_mask = cv2.morphologyEx(mineral_mask, cv2.MORPH_OPEN, kernel)
            
            # 计算矿物区域
            count = cv2.countNonZero(mineral_mask)
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            
            # 如果矿物像素占比超过15%且绝对数量大于50，则认为有矿物
            if count > (total_pixels * 0.15) and count > 50:
                # 计算矿物的中心位置
                M = cv2.moments(mineral_mask)
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
            # 1. 写入文件 (原功能保留)
            with open(self.block_file, 'w') as f:
                f.write(str(combined_states))
            print(f"updated: {combined_states}")

            # --- 新增: 发送到串口 ---
            if self.ser and self.ser.is_open:
                # 2. 将列表转换为字符串，并添加换行符 (例如: "[-1, 0, 1, -1, 1, 0]\n")
                data_to_send = f"{str(combined_states)}\n"
                
                # 3. 编码为字节并发送
                self.ser.write(data_to_send.encode('utf-8'))
                print(f"已发送到串口: {data_to_send.strip()}")
            # --- 新增结束 ---

        except Exception as e:
            # 更新了错误信息，使其包含文件和串口两种可能的错误
            print(f"更新block文件或发送串口时出错: {e}")

    def fast_color_detection(self, process_frame, original_dims):
        start_time = time.time()
        detected_areas = []
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 3)
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]

        for color_name in self.color_ranges.keys():
            mask = self.enhance_color_detection(hsv, color_name)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 400
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    hull = cv2.convexHull(contour)
                    x, y, w, h = cv2.boundingRect(hull)
                    
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
            cv2.drawContours(result, [area['contour']], -1, area['color'], 2)
            cv2.putText(result, area['name'], area['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, area['color'], 2)
            
            # 如果检测到矿物，绘制矿物标记
            if area.get('has_mineral', False):
                mineral_center = area.get('mineral_center', area['center'])
                cv2.circle(result, mineral_center, 10, self.color_bgr['Mineral_Gray'], -1)
                cv2.putText(result, "MINERAL", (mineral_center[0]-30, mineral_center[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # 绘制从矿物到色块的箭头
                cv2.arrowedLine(result, mineral_center, area['center'], 
                               self.color_bgr['Mineral_Gray'], 2)

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
        
        # --- 新增: 关闭串口 ---
        if detector.ser and detector.ser.is_open:
            detector.ser.close()
            print("串口已关闭。")
        # --- 新增结束 ---

        print("Program exited")


if __name__ == "__main__":
    main()