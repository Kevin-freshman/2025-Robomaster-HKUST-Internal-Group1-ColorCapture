import cv2
import numpy as np
import time
from collections import deque
import os
import serial
import datetime

class OptimizedColorDetector:
    def __init__(self):
        # 【修改1：大幅放宽颜色阈值】
        # 之前的阈值太严，导致蓝色无法识别。现在放宽S和V，依靠几何形状过滤噪点。
        self.color_ranges = {
            'Red': [
                # 红色范围扩大，S下限降至43
                ([0, 43, 46], [10, 255, 255]),
                ([156, 43, 46], [180, 255, 255])
            ],
            'Green': [
                # 绿色范围扩大
                ([35, 43, 46], [85, 255, 255])
            ],
            'Blue': [
                # 蓝色重点修改：S下限降至43，V下限降至46
                # 只要是蓝色调，哪怕有点偏灰或偏暗都能识别
                ([90, 43, 46], [130, 255, 255])
            ],
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 30, 255])],
            # 矿物颜色：深灰/黑色
            'Mineral_Gray': [([0, 0, 0], [180, 255, 100])] 
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
        
        # 形态学内核
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
        
        self.processing_times = deque(maxlen=30)

        self.status_file = "color_status.txt"
        self.block_file = "block_states.txt"
        
        self.last_status = None
        self.last_sent_data = None 
        self.locked_sequence = [] 
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

    # 【修改2：新增矿物检测函数】专门检测色块上方的区域
    def detect_mineral_above(self, hsv_full, x, y, w, h):
        """
        在给定矩形(x,y,w,h)的上方检测是否有灰色矿物
        """
        # 定义ROI：在色块上方
        # 宽度：与色块相同
        # 高度：取色块高度的0.6倍左右，向上偏移
        roi_h = int(h * 0.6) 
        roi_y_start = max(0, y - roi_h - 5) # 向上取区域，预留5像素间隙
        roi_y_end = max(0, y - 5)
        
        # 如果到达屏幕顶部，返回False
        if roi_y_start >= roi_y_end:
            return False, None

        # 截取上方区域的HSV数据
        roi_hsv = hsv_full[roi_y_start:roi_y_end, x:x+w]
        
        # 在ROI中检测灰色
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
            # 如果灰色像素占比超过 ROI 面积的 15%，则认为有矿物
            if count > (total_pixels * 0.15) and count > 50:
                # 返回 True 和 ROI 的中心坐标（用于画图）
                center_x = x + w // 2
                center_y = roi_y_start + (roi_y_end - roi_y_start) // 2
                return True, (center_x, center_y)
        
        return False, None

    def update_block_file(self, detected_areas):
        # 【修改3：更新颜色映射逻辑】
        # 红色/蓝色 -> 1, 绿色 -> 0
        color_map = {'Red': 1, 'Blue': 1, 'Green': 0}
        # 只关注这三种颜色
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
            # 锁定后，只检测矿物
            # 这里的逻辑：输出格式为 [矿物1, 矿物2]
            # 只针对“红色”或“蓝色”的板子检测矿物
            # 假设锁定的序列中，值为1的位置（红/蓝）才可能有矿物
            
            mineral_status = [] 
            
            # 将检测到的板子映射回锁定序列
            # 注意：这里假设摄像头不动，检测到的top4顺序与锁定顺序一致
            current_mineral_flags = []
            for i in range(4):
                if i < len(top4):
                    current_mineral_flags.append(top4[i]['has_mineral'])
                else:
                    current_mineral_flags.append(False)

            # 遍历锁定的序列
            for i, code in enumerate(self.locked_sequence):
                # code 1 代表 Red 或 Blue
                if code == 1: 
                    if i < len(current_mineral_flags) and current_mineral_flags[i]:
                        mineral_status.append(1) 
                    else:
                        mineral_status.append(0) 
            
            # 补齐或截取为2位（假设最多两个红/蓝板）
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
        # 适当模糊，平滑颜色
        hsv_blur = cv2.medianBlur(hsv, 5) 
        
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]
            
        frame_area = process_frame.shape[0] * process_frame.shape[1]
        plate_colors = ['Red', 'Green', 'Blue', 'Gold', 'Silver']
        
        for color_name in plate_colors:
            # 使用模糊后的HSV图进行检测
            mask = self.enhance_color_detection(hsv_blur, color_name)
            if mask is None:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = 1000 
            max_area = frame_area / 3 
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area and area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # 几何筛选：保留长宽比0.5到2.0的矩形
                    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                        continue

                    # 多边形拟合筛选
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    if len(approx) < 4 or len(approx) > 8:
                        continue

                    hull = cv2.convexHull(contour)
                    
                    # 计算中心点
                    center_x_proc = x + w // 2
                    center_y_proc = y + h // 2

                    has_mineral = False
                    mineral_pos = None
                    
                    # 只有在锁定后，且是红色或蓝色时，才调用上方检测函数
                    if self.is_locked and color_name in ['Red', 'Blue']:
                        has_mineral, mineral_pos = self.detect_mineral_above(hsv_blur, x, y, w, h)

                    # 坐标转换
                    if self.processing_scale != 1.0:
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w = int(w * scale_x)
                        orig_h = int(h * scale_y)
                        hull_scaled = (hull * np.array([scale_x, scale_y])).astype(np.int32)
                        center_x = orig_x + orig_w // 2
                        center_y = orig_y + orig_h // 2
                        # 如果检测到矿物，矿物坐标也要转换
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
                        'mineral_center': mineral_pos # 存储矿物位置用于绘制
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
            
            # 绘制矿物检测结果
            if area.get('has_mineral', False):
                # 在检测到的矿物位置画圆
                m_center = area.get('mineral_center', area['center'])
                cv2.circle(result, m_center, 15, (50, 50, 50), -1)
                cv2.putText(result, "MINERAL", (m_center[0]-30, m_center[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # 画一个箭头指向板子，表示归属关系
                cv2.arrowedLine(result, m_center, area['center'], (0,0,255), 2)

        if self.is_locked:
             mineral_disp = []
             # 获取当前红蓝板的矿物状态用于显示
             current_minerals = []
             # 过滤只看Red和Blue
             rb_areas = [a for a in detected_areas if a['name'] in ['Red', 'Blue']]
             rb_areas.sort(key=lambda x: x['center'][0]) # 按位置排序
             
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