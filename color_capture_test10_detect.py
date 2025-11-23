import cv2
import numpy as np
import time
from collections import deque
import os
#import flip from cv2
import serial

class OptimizedColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),
                ([170, 120, 70], [180, 255, 255])
            ],
            'Green': [([35, 50, 50], [85, 255, 255])],
            'Blue': [([90, 50, 50], [135, 255, 255])],
            'Gold': [([15, 80, 80], [35, 255, 255])],
            'Silver': [([0, 0, 100], [180, 50, 230])],
            'Black': [([0, 0, 0], [180, 255, 60])]
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

        self.ser = None

        try:

            serial_port = 'COM14'
            baud_rate = 9600
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)


            print("串口打开成功！")
            self.ser.write(b"hello_serial\r\n")
            print("发送测试信息：hello_serial")



            print(f"Serial port {serial_port} has been successfully opened, baud rate {baud_rate}")
        except serial.SerialException as e:
            print(f"An error occurred while opening the serial port: {e}")
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

    def update_status_file(self, status):
        if status != self.last_status:
            try:
                with open(self.status_file, 'w') as f:
                    f.write(str(status))
                self.last_status = status
            except Exception as e:
                print(f"file error: {e}")

    
    def update_block_file(self, detected_areas):
        
        color_priority = {'Red': 0, 'Blue': 0, 'Green': 1}

        filtered = [a for a in detected_areas if a['name'] in color_priority]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]
        top4.sort(key=lambda a: a['center'][0])

        
        block_states = [color_priority[a['name']] for a in top4]
        greens = [a for a in detected_areas if a['name'] == 'Green']
        silvers = [a for a in detected_areas if a['name'] == 'Silver']

        for i, area in enumerate(top4):
            if area['name'] == 'Green':
                gx, gy = area['center']

                # 判断是否有银色矿在绿色板“上方”区域
                silver_above = False
                for silver in silvers:
                    sx, sy = silver['center']

                    # 条件：银色矿在绿色板上方（sy < gy）且水平位置相近
                    if sy < gy and abs(sx - gx) < area['size'] * 1.2:
                        silver_above = True
                        break

                # 将 1 -> 2
                if silver_above:
                    block_states[i] = 2
        
        while len(block_states) < 4:
            block_states.append(-1)

        try:
            with open(self.block_file, 'w') as f:
                f.write(str(block_states))
            print(f"updated: {block_states}")

            if self.ser and self.ser.is_open:
                data_to_send = f"{str(block_states)}\n"
                self.ser.write(data_to_send.encode('utf-8'))
                print("serial data has been sent")
        except Exception as e:

            print(f"Error occurred while updating block file or sending serial port: {e}")

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
                    if self.processing_scale != 1.0:
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)
                        hull = (hull * np.array([scale_x, scale_y])).astype(np.int32)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(w, h),
                        'color': self.color_bgr[color_name],
                        'contour': hull
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
            print("Serial port is closed.")
        print("Program exited")


if __name__ == "__main__":
    main()