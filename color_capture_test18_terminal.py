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
            # 【修改1】红色保持灵敏，覆盖橙色
            'Green': [
                # 绿色 H 范围可以适当放宽，S/V 下限同步降低
                ([30, 30, 30], [95, 255, 255])
            ],
            'Red': [
            ([0, 20, 20], [35, 255, 255]),   # 扩大亮红/橙红/暗红
            ([140, 20, 20], [180, 255, 255]) # 扩大深红范围
             ],

            'Blue': [
                ([75, 25, 25], [150, 255, 255])  # 扩大蓝色区间并降低阈值
            ],
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 30, 255])],
            
            # 【修改3】矿物灰色范围扩大：
            # 之前的 V<100 太黑了。现在允许 V=40-220 (支持浅灰)，但限制 S<50 (必须是灰色，不能有色彩)
            'Mineral_Gray': [([0, 0, 40], [180, 50, 220])] 
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
        
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) 
        
        self.processing_times = deque(maxlen=30)

        self.status_file = "color_status.txt"
        self.block_file = "block_states.txt"
        
        self.last_status = None
        self.last_sent_data = None 
        self.locked_sequence = [] 
        self.lock_counter = 0       
        self.is_locked = False      
        self.ser = None

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
        print("done")

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
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        return mask

    def detect_mineral_above(self, hsv_full, x, y, w, h):
        # 【修改4】限制矿物搜索距离
        # 只在色块上方 5像素 到 0.6倍高度 之间搜索
        # 这样即使有误识别，也不会跑到天上去
        
        roi_h = int(h * 0.6) 
        roi_y_start = max(0, y - roi_h - 5) 
        roi_y_end = max(0, y - 5) # 紧贴色块上方
        
        if roi_y_start >= roi_y_end:
            return False, None

        roi_hsv = hsv_full[roi_y_start:roi_y_end, x:x+w]
        
        ranges = self.color_ranges['Mineral_Gray']
        mask = None
        for lower, upper in ranges:
            temp_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
            if mask is None:
                mask = temp_mask
            else:
                mask = cv2.bitwise_or(mask, temp_mask)

        if mask is not None:
            count = cv2.countNonZero(mask)
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            # 阈值：像素占比 > 10% 且 绝对数量 > 50
            if count > (total_pixels * 0.10) and count > 50:
                center_x = x + w // 2
                # 简单的中心计算
                center_y = roi_y_start + (roi_y_end - roi_y_start) // 2
                return True, (center_x, center_y)
        
        return False, None

    def update_block_file(self, detected_areas):
        color_map = {'Red': 1, 'Blue': 1, 'Green': 0}
        valid_colors = ['Red', 'Blue', 'Green']

        filtered = [a for a in detected_areas if a['name'] in valid_colors]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]
        top4.sort(key=lambda a: a['center'][0]) 

        txt_log_content = ""
        serial_send_content = ""
        current_data_signature = ""

        if not self.is_locked:
            if len(top4) == 4:
                self.lock_counter += 1
                if self.lock_counter > 20:
                    self.is_locked = True
                    self.locked_sequence = [color_map[a['name']] for a in top4]
                    print(f"【SYSTEM LOCKED】Color Sequence: {self.locked_sequence}")
            else:
                self.lock_counter = 0
            
            current_colors = [color_map[a['name']] for a in top4]
            while len(current_colors) < 4:
                current_colors.append(-1)
            
            txt_log_content = str(current_colors)
            serial_send_content = str(current_colors)
            current_data_signature = str(current_colors)
            
        else:
            mineral_status = [] 
            current_mineral_flags = []
            # 将当前检测到的矿物状态映射到 top4
            # 注意：这里需要一个稳健的映射，假设位置大致不变
            # 更好的做法是按x坐标重新匹配，这里简化处理假设顺序一致
            for i in range(4):
                if i < len(top4):
                    current_mineral_flags.append(top4[i]['has_mineral'])
                else:
                    current_mineral_flags.append(False)

            for i, code in enumerate(self.locked_sequence):
                if code == 1: 
                    if i < len(current_mineral_flags) and current_mineral_flags[i]:
                        mineral_status.append(1) 
                    else:
                        mineral_status.append(0) 
            
            while len(mineral_status) < 2:
                mineral_status.append(0)
            mineral_status = mineral_status[:2]

            txt_log_content = f"{str(self.locked_sequence)}{str(mineral_status)}"
            serial_send_content = str(mineral_status)
            current_data_signature = txt_log_content 

        if current_data_signature != self.last_sent_data:
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.block_file, 'a') as f:
                    f.write(f"[{timestamp}] {txt_log_content}\n")
                print(f"Log Updated: {txt_log_content}")

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
        hsv_blur = cv2.medianBlur(hsv, 5) 
        
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]
            
        frame_area = process_frame.shape[0] * process_frame.shape[1]
        plate_colors = ['Red', 'Green', 'Blue', 'Gold', 'Silver']
        
        for color_name in plate_colors:
            mask = self.enhance_color_detection(hsv_blur, color_name)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = 600 
            max_area = frame_area / 3 
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area and area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # 【修改5：长宽比锁死】
                    # 真正的色块是正方形(1.0)。手机是长方形(2.0)。
                    # 限制在 0.8 到 1.3 之间，彻底过滤掉长方形的手机误识别！
                    if aspect_ratio < 0.8 or aspect_ratio > 1.3:
                        continue
                    
                    # 适当放宽多边形拟合，防止圆角被过滤
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    # 如果检测到是矩形（4点）或圆角矩形（>4点）都保留
                    # 只过滤明显的三角形(<4)
                    if len(approx) < 4: 
                        continue

                    hull = cv2.convexHull(contour)
                    
                    center_x_proc = x + w // 2
                    center_y_proc = y + h // 2

                    has_mineral = False
                    mineral_pos = None
                    
                    if self.is_locked and color_name in ['Red', 'Blue']:
                        has_mineral, mineral_pos = self.detect_mineral_above(hsv_blur, x, y, w, h)

                    if self.processing_scale != 1.0:
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w = int(w * scale_x)
                        orig_h = int(h * scale_y)
                        hull_scaled = (hull * np.array([scale_x, scale_y])).astype(np.int32)
                        center_x = orig_x + orig_w // 2
                        center_y = orig_y + orig_h // 2
                        if mineral_pos:
                            min_x = int((mineral_pos[0] - w//2) * scale_x) + orig_w//2
                            min_y = int(mineral_pos[1] * scale_y)
                            mineral_pos = (min_x, min_y)
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
            cv2.drawContours(result, [area['contour']], -1, area['color'], 3)
            cv2.putText(result, area['name'], area['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, area['color'], 2)
            
            if area.get('has_mineral', False):
                m_center = area.get('mineral_center', area['center'])
                cv2.circle(result, m_center, 15, (50, 50, 50), -1)
                cv2.putText(result, "MINERAL", (m_center[0]-30, m_center[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.arrowedLine(result, m_center, area['center'], (0,0,255), 2)

        if self.is_locked:
             current_minerals = []
             rb_areas = [a for a in detected_areas if a['name'] in ['Red', 'Blue']]
             rb_areas.sort(key=lambda x: x['center'][0]) 
             
             for a in rb_areas:
                 current_minerals.append(1 if a['has_mineral'] else 0)
             
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
                detector.locked_sequence = []
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