import cv2
import numpy as np
import time
from collections import deque
import os
import serial
import datetime  # 新增：用于记录时间

class OptimizedColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),
                ([170, 120, 70], [180, 255, 255])
            ],
            'Green': [([35, 50, 50], [85, 255, 255])],
            # 【修改点1】修正蓝色阈值，提高S和V的下限，防止误识别头发/黑色物体
            'Blue': [([100, 110, 70], [130, 255, 255])], 
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 50, 230])],
            # 黑色/灰色用于矿物识别
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
            # 可以根据实际情况微调这个像素阈值
            return count > 50 
        return False

    def update_block_file(self, detected_areas):
        color_priority = {'Red': 0, 'Blue': 0, 'Green': 1}

        filtered = [a for a in detected_areas if a['name'] in color_priority]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]
        top4.sort(key=lambda a: a['center'][0]) 

        # --- 锁定逻辑与六元数组生成 ---
        if not self.is_locked:
            if len(top4) == 4:
                self.lock_counter += 1
                if self.lock_counter > 20:
                    self.is_locked = True
                    self.locked_sequence = [color_priority[a['name']] for a in top4]
                    print(f"Color Sequence Locked: {self.locked_sequence}")
            else:
                self.lock_counter = 0
            
            current_colors = [color_priority[a['name']] for a in top4]
            while len(current_colors) < 4:
                current_colors.append(-1)
            data_list = current_colors
            
        else:
            mineral_indices = []
            for i, area in enumerate(top4):
                if i >= 4: break 
                if area['has_mineral']:
                    mineral_indices.append(i + 1)

            while len(mineral_indices) < 2:
                mineral_indices.append(0)
            
            data_list = mineral_indices[:2]

        # --- 【修改点2】数据同步记录与发送 ---
        data_str = str(data_list)
        
        # 只有当数据发生变化时才记录和发送
        if data_str != self.last_sent_data:
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] {data_str}\n"
                
                # 使用 'a' (append) 模式，确保按时间顺序记录
                with open(self.block_file, 'a') as f:
                    f.write(log_entry)
                print(f"State Updated: {log_entry.strip()}")

                # 发送串口
                if self.ser and self.ser.is_open:
                    data_to_send_serial = f"{data_str}\n"
                    self.ser.write(data_to_send_serial.encode('utf-8'))
                    print(f"Serial sent: {data_str}")
                
                self.last_sent_data = data_str

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

        plate_colors = ['Red', 'Green', 'Blue', 'Gold', 'Silver']
        
        for color_name in plate_colors:
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
                    
                    center_x_proc = x + w // 2
                    center_y_proc = y + h // 2

                    has_mineral = False
                    if color_name in ['Red', 'Blue']:
                        roi_hsv = hsv[y:y+h, x:x+w]
                        if self.check_mineral(roi_hsv):
                            has_mineral = True

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
             cv2.putText(result, f"LOCKED: {self.locked_sequence}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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



# 出现了以下问题：
# 1.六元数组锁定逻辑好像没有在txt文件里体现（在识别成功四个颜色板后，如[1,0,0,1]便停止更新颜色板状态），之后只识别并更新数组最后两位，如果第一个蓝色或红色颜色板上有矿物，第二个蓝色或红色颜色板上没有矿物，则显示[1,0,0,1][1,0]，并利用串口发送[1,0]
# 2.颜色识别逻辑调整，只有在识别成功四个颜色板后才进行矿物的识别