import cv2
import numpy as np
import time
from collections import deque
import os
import serial
import datetime

class OptimizedColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),
                ([170, 120, 70], [180, 255, 255])
            ],
            'Green': [([35, 50, 50], [85, 255, 255])],
            # 保持较高的S和V下限，防止误识别黑色物体
            'Blue': [([100, 110, 70], [130, 255, 255])], 
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 50, 230])],
            'Black': [([0, 0, 0], [180, 255, 90])] 
        }

        self.color_bgr = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Gold': (0, 215, 255),
            'Silver': (192, 192, 192),
            'Black': (50, 50, 50)
        }

        self.processing_scale = 0.4
        self.target_width = 1920
        self.target_height = 1080

        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_gold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.processing_times = deque(maxlen=30)

        self.status_file = "color_status.txt"
        self.block_file = "block_states.txt"
        
        self.last_status = None
        self.last_sent_data = None 
        self.locked_sequence = None 
        self.lock_counter = 0       
        self.is_locked = False      

        self.ser = None

        # 串口扫描
        print("Scanning serial ports COM10-COM19...")
        for port_num in range(10, 20):
            port_name = f'COM{port_num}'
            try:
                self.ser = serial.Serial(port_name, 9600, timeout=1)
                print(f"串口打开成功！Connected to {port_name}")
                self.ser.write(b"hello_serial\r\n")
                print("发送测试信息：hello_serial")
                break 
            except serial.SerialException:
                continue
        
        if self.ser is None:
            print("Warning: No available serial port found between COM10 and COM19.")
            print("Continue running, but without serial communication.")

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

    def check_mineral(self, hsv_roi):
        ranges = self.color_ranges['Black']
        mask = None
        for lower, upper in ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            temp_mask = cv2.inRange(hsv_roi, lower_np, upper_np)
            if mask is None:
                mask = temp_mask
            else:
                mask = cv2.bitwise_or(mask, temp_mask)
        
        if mask is not None:
            count = cv2.countNonZero(mask)
            return count > 50 
        return False

    def update_block_file(self, detected_areas):
        color_priority = {'Red': 0, 'Blue': 0, 'Green': 1}

        # 筛选主要的四个色块
        filtered = [a for a in detected_areas if a['name'] in color_priority]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]
        top4.sort(key=lambda a: a['center'][0]) 

        # --- 状态变量初始化 ---
        txt_log_content = ""
        serial_send_content = ""
        current_data_signature = "" # 用于判断状态是否改变

        # --- 逻辑分支：锁定前 vs 锁定后 ---
        if not self.is_locked:
            # [阶段1] 正在识别颜色序列
            if len(top4) == 4:
                self.lock_counter += 1
                # 连续稳定检测 20 帧后锁定
                if self.lock_counter > 20:
                    self.is_locked = True
                    self.locked_sequence = [color_priority[a['name']] for a in top4]
                    print(f"【SYSTEM LOCKED】Color Sequence: {self.locked_sequence}")
            else:
                self.lock_counter = 0
            
            # 生成当前颜色数组
            current_colors = [color_priority[a['name']] for a in top4]
            while len(current_colors) < 4:
                current_colors.append(-1)
            
            # 锁定前：TXT记录颜色，串口发送颜色
            txt_log_content = str(current_colors)
            serial_send_content = str(current_colors)
            current_data_signature = str(current_colors)
            
        else:
            # [阶段2] 颜色已锁定，只更新矿物
            # 提取当前top4中的矿物信息（假设排序仍大致对应，因为摄像头不动）
            # 我们关注的是 top4 中哪几个位置有 has_mineral=True
            
            mineral_status = [] # 存储 [0, 1] 这种格式
            
            # 找到对应 locked_sequence 中的 Red 或 Blue 的索引
            # 注意：top4 是按 x 坐标排序的检测到的物体。
            # 如果检测稳定，top4[0] 对应 locked_sequence[0]，以此类推。
            
            mineral_flags = [] # 暂存所有板子的矿物状态 [False, True, False, False]
            for i in range(4):
                if i < len(top4):
                    mineral_flags.append(top4[i]['has_mineral'])
                else:
                    mineral_flags.append(False)

            # 根据需求：只发送两位数组表示：第几个红色/蓝色板上方有矿物
            # 这里的逻辑是：遍历锁定的序列，如果是R或B，则检查对应位置是否有矿物
            
            count_target_boards = 0
            for i, code in enumerate(self.locked_sequence): # type: ignore
                # 0代表Red或Blue (在 color_priority 中 Red=0, Blue=0)
                # 1代表Green
                if code == 0: # 是红色或蓝色板
                    if i < len(mineral_flags) and mineral_flags[i]:
                        mineral_status.append(1) # 有矿物
                    else:
                        mineral_status.append(0) # 无矿物
            
            # 补齐或截取为2位
            while len(mineral_status) < 2:
                mineral_status.append(0)
            mineral_status = mineral_status[:2]

            # 锁定后：TXT记录 [颜色序列][矿物状态]，串口只发送 [矿物状态]
            txt_log_content = f"{str(self.locked_sequence)}{str(mineral_status)}"
            serial_send_content = str(mineral_status)
            current_data_signature = txt_log_content # 用全量信息判断是否变化

        # --- 数据写入与发送 ---
        
        # 只有当数据状态发生变化时才操作
        if current_data_signature != self.last_sent_data:
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 写入 TXT
                with open(self.block_file, 'a') as f:
                    f.write(f"[{timestamp}] {txt_log_content}\n")
                print(f"Log Updated: {txt_log_content}")

                # 发送串口
                if self.ser and self.ser.is_open:
                    data_to_send_serial = f"{serial_send_content}\n"
                    self.ser.write(data_to_send_serial.encode('utf-8'))
                    print(f"Serial Sent: {serial_send_content}")
                
                self.last_sent_data = current_data_signature

            except Exception as e:
                print(f"Error updating block file/serial: {e}")

    def fast_color_detection(self, process_frame, original_dims):
        start_time = time.time()
        detected_areas = []
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 3)
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]
            
        # 计算图像总面积，用于排除过大的误识别（如背景）
        frame_area = process_frame.shape[0] * process_frame.shape[1]

        plate_colors = ['Red', 'Green', 'Blue', 'Gold', 'Silver']
        
        for color_name in plate_colors:
            mask = self.enhance_color_detection(hsv, color_name)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 400
            # 设定最大面积为屏幕的1/4，超过这个大小通常是背景误识别（例如图片中的蓝色大框）
            max_area = frame_area / 4 
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # 增加 max_area 判断，过滤掉那个巨大的蓝色背景框
                if area > min_area and area < max_area:
                    hull = cv2.convexHull(contour)
                    x, y, w, h = cv2.boundingRect(hull)
                    
                    center_x_proc = x + w // 2
                    center_y_proc = y + h // 2

                    has_mineral = False
                    
                    # 【修改重点】只有在已锁定的状态下，才进行矿物识别
                    if self.is_locked:
                        if color_name in ['Red', 'Blue']:
                            roi_hsv = hsv[y:y+h, x:x+w]
                            if self.check_mineral(roi_hsv):
                                has_mineral = True
                    
                    # 还原坐标
                    if self.processing_scale != 1.0:
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w = int(w * scale_x)
                        orig_h = int(h * scale_y)
                        hull_scaled = (hull * np.array([scale_x, scale_y])).astype(np.int32)
                        center_x = orig_x + orig_w // 2
                        center_y = orig_y + orig_h // 2
                    else:
                        hull_scaled = hull
                        center_x = center_x_proc
                        center_y = center_y_proc

                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(w, h),
                        'color': self.color_bgr[color_name],
                        'contour': hull_scaled,
                        'has_mineral': has_mineral
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
            
            if area.get('has_mineral', False):
                cv2.circle(result, area['center'], 20, (50, 50, 50), -1)
                text_pos = (area['center'][0] - 30, area['center'][1] + 10)
                cv2.putText(result, "MINERAL", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.is_locked:
             # 在屏幕上显示完整状态 [颜色][矿物]
             # 重新计算一下用于显示的 mineral_status
             mineral_disp = []
             color_priority = {'Red': 0, 'Blue': 0, 'Green': 1}
             # 这里只为了显示，简单筛选一下
             current_minerals = [1 if a['has_mineral'] else 0 for a in detected_areas if a['name'] in ['Red', 'Blue']]
             # 截取前两个用于显示
             while len(current_minerals) < 2: current_minerals.append(0)
             
             status_text = f"LOCKED: {self.locked_sequence} Min: {current_minerals[:2]}"
             cv2.putText(result, status_text, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    screenshot_count = 0
    try:
        while True:
            ret, frame = cap.read()
            #frame = cv2.flip(frame0, 1)
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
            elif key in [ord('r'), ord('R')]:
                detector.is_locked = False
                detector.locked_sequence = None
                detector.lock_counter = 0
                print("Reset Lock State")

    except KeyboardInterrupt:
        print("User interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector.ser and detector.ser.is_open:
            detector.ser.close()
            print("Serial port is closed.")
        print("Program exited")


if __name__ == "__main__":
    main()