import cv2
import numpy as np
import time
from collections import deque

class OptimizedColorDetector:
    def __init__(self):
        # 扩大颜色检测范围 (HSV格式)
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),      # 红色范围1 - 扩大范围
                ([170, 120, 70], [180, 255, 255])     # 红色范围2 - 扩大范围
            ],
            'Green': [([35, 50, 50], [85, 255, 255])],  # 绿色范围扩大
            'Blue': [([90, 50, 50], [135, 255, 255])],  # 蓝色范围扩大
            'Gold': [([15, 80, 80], [35, 255, 255])],   # 金色范围扩大
            'Silver': [([0, 0, 100], [180, 50, 230])],  # 银色范围调整
            'Black': [([0, 0, 0], [180, 255, 60])]      # 黑色范围调整
        }
        
        self.color_bgr = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Gold': (0, 215, 255),  # 金色
            'Silver': (192, 192, 192),  # 银色
            'Black': (50, 50, 50)   # 使用深灰色而不是纯黑，便于显示
        }
        
        # 性能优化设置
        self.processing_scale = 0.4  
        self.target_width = 1920
        self.target_height = 1080
        
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()
        
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        self.kernel_gold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        self.processing_times = deque(maxlen=30)
        
        print("优化版高性能颜色识别系统初始化完成")
        print(f"目标颜色: 红, 绿, 蓝, 金, 银, 黑")
        print(f"处理比例: {self.processing_scale}")
        
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
            
            area_thresholds = {
                'Red': 400,      # 降低红色阈值
                'Green': 400,    # 降低绿色阈值
                'Blue': 400,     # 降低蓝色阈值
                'Gold': 300,     # 降低金色阈值
                'Silver': 500,   # 降低银色阈值
                'Black': 600     # 降低黑色阈值
            }
            
            min_area = area_thresholds.get(color_name, 400)
            
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
                    
                    perimeter = cv2.arcLength(hull, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        circularity_thresholds = {
                            'Red': 0.2,
                            'Green': 0.2,
                            'Blue': 0.2,
                            'Gold': 0.15,   
                            'Silver': 0.25, 
                            'Black': 0.15   
                        }
                        min_circularity = circularity_thresholds.get(color_name, 0.2)
                        if circularity < min_circularity:
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
        

        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return detected_areas
    
    def calculate_fps(self):

        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else fps
    
    def create_rounded_mask(self, contours, img_shape):

        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for contour in contours:
            if len(contour) > 2:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(mask, [approx], 255)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask
    
    def get_light_color(self, color_bgr, alpha=0.3):
        if color_bgr == (50, 50, 50):
            return (100, 100, 100)
        return tuple(int(c + (255 - c) * alpha) for c in color_bgr)
    
    def put_pretty_text(self, img, text, position, color, 
                       font_scale=1.0, thickness=2, shadow=True):
    
        font = cv2.FONT_HERSHEY_SIMPLEX  
        
        text = str(text)
        
        if shadow:
            shadow_color = (0, 0, 0)
            cv2.putText(img, text, 
                       (position[0] + 2, position[1] + 2), 
                       font, font_scale, shadow_color, thickness + 1, 
                       cv2.LINE_AA)
        
        cv2.putText(img, text, position, 
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    def process_frame(self, frame):
        
        display_frame, process_frame = self.optimize_frame(frame)
        
        display_frame = cv2.flip(display_frame, 1)
        process_frame = cv2.flip(process_frame, 1)
        
        original_dims = (display_frame.shape[1], display_frame.shape[0])
        detected_areas = self.fast_color_detection(process_frame, original_dims)
        
        result = display_frame.copy()
        
        color_layer = np.zeros_like(display_frame, dtype=np.uint8)
        
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            contour = area['contour']
            
            contour_mask = self.create_rounded_mask([contour], display_frame.shape)
            
            light_color = self.get_light_color(color, alpha=0.4)
            
            colored_area = np.full_like(display_frame, light_color, dtype=np.uint8)
            color_layer[contour_mask > 0] = colored_area[contour_mask > 0]
        
        result = cv2.addWeighted(display_frame, 0.7, color_layer, 0.3, 0)
        
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            
            font_scale = max(0.6, min(1.5, size / 150))
            thickness = max(1, int(size / 100))
            
            radius = max(25, size // 10)
            cv2.circle(result, center, radius, (40, 40, 40), -1)
            cv2.circle(result, center, radius, color, 2)
            
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            self.put_pretty_text(result, name, (text_x, text_y), color, 
                               font_scale, thickness)
        
        return result, len(detected_areas)
    
    def add_performance_info(self, frame, detected_count):
        h, w = frame.shape[:2]
        
        fps = self.calculate_fps()
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        title_bg = np.zeros((60, w, 3), dtype=np.uint8)
        title_bg[:, :] = (40, 40, 60)
        frame[0:60, 0:w] = cv2.addWeighted(frame[0:60, 0:w], 0.3, title_bg, 0.7, 0)
        
        title = "Color Detection System (Red, Green, Blue, Gold, Silver, Black)"
        self.put_pretty_text(frame, title, (20, 40), 
                           (255, 255, 255), 0.8, 2)
        
        status_bg = np.zeros((40, w, 3), dtype=np.uint8)
        status_bg[:, :] = (30, 30, 30)
        frame[h-40:h, 0:w] = cv2.addWeighted(frame[h-40:h, 0:w], 0.4, status_bg, 0.6, 0)
        
        status_text = f"Detected: {detected_count} colors | Q:Quit | S:Save | +/-:Scale"
        self.put_pretty_text(frame, status_text, (20, h-10), 
                           (200, 200, 200), 0.6, 1)
        
        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        self.put_pretty_text(frame, fps_text, (w - fps_size[0] - 20, 40), 
                           (0, 255, 255), 0.6, 1)
        
        time_text = f"Process: {avg_processing_time:.1f}ms"
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, time_text, (w - time_size[0] - 20, 70), 
                           (0, 200, 255), 0.5, 1)
        
        res_text = f"Resolution: {w}x{h}"
        res_size = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, res_text, (w - res_size[0] - 20, 95), 
                           (0, 200, 255), 0.5, 1)
        
        scale_text = f"Scale: {self.processing_scale}"
        scale_size = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, scale_text, (w - scale_size[0] - 20, 120), 
                           (0, 255, 0), 0.5, 1)

def main():
    detector = OptimizedColorDetector()
    
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    print(f"Processing scale: {detector.processing_scale}")
    print("Press '+' to increase scale (more accurate)")
    print("Press '-' to decrease scale (faster)")
    print("Press 'S' to save screenshot")
    print("Press 'Q' to quit")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break
            
            result, count = detector.process_frame(frame)
            
            detector.add_performance_info(result, count)
            
            cv2.imshow('Optimized Color Detection System', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
      
                screenshot_count += 1
                filename = f'color_detection_{screenshot_count}.png'
                cv2.imwrite(filename, result)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                detector.processing_scale = min(1.0, detector.processing_scale + 0.1)
                print(f"Scale increased to: {detector.processing_scale}")
            elif key == ord('-') or key == ord('_'):
                detector.processing_scale = max(0.2, detector.processing_scale - 0.1)
                print(f"Scale decreased to: {detector.processing_scale}")
                
    except KeyboardInterrupt:
        print("User interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program exited")

if __name__ == "__main__":
    main()