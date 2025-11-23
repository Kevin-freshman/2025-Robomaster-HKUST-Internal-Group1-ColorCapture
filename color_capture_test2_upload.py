import cv2
import numpy as np
import time
from collections import deque

class HighPerformanceColorDetector:
    def __init__(self):
        self.color_ranges = {
            'Red': [
                ([0, 150, 50], [10, 255, 255]),      # Red range 1
                ([170, 150, 50], [180, 255, 255])    # Red range 2
            ],
            'Blue': [([100, 150, 50], [130, 255, 255])],
            'Green': [([40, 80, 50], [80, 255, 255])],
            'Yellow': [([20, 100, 100], [30, 255, 255])],
            'Purple': [([130, 80, 50], [160, 255, 255])],
            'Orange': [([10, 150, 100], [20, 255, 255])],
            'Cyan': [([85, 100, 50], [100, 255, 255])]
        }
        
        self.color_bgr = {
            'Red': (0, 0, 255),
            'Blue': (255, 0, 0),
            'Green': (0, 255, 0),
            'Yellow': (0, 255, 255),
            'Purple': (255, 0, 255),
            'Orange': (0, 165, 255),
            'Cyan': (255, 255, 0)
        }
        
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()
        
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        self.target_width = 1920 
        self.target_height = 1080
        
        self.gradient_bg = self.create_gradient_bg(self.target_width, self.target_height)
        
    def create_gradient_bg(self, width, height):
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            ratio = i / height
            r = int(30 + ratio * 50)
            g = int(30 + ratio * 30)
            b = int(70 + ratio * 100)
            bg[i, :] = (b, g, r)
        return bg
    
    def get_light_color(self, color_bgr, alpha=0.3):
        return tuple(int(c + (255 - c) * alpha) for c in color_bgr)
    
    def put_pretty_text(self, img, text, position, color, 
                       font_scale=1.0, thickness=2, shadow=True):
        fonts = [
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_DUPLEX
        ]
        font = fonts[0] 
        
        if shadow:
            shadow_color = (0, 0, 0)
            cv2.putText(img, text, 
                       (position[0] + 2, position[1] + 2), 
                       font, font_scale, shadow_color, thickness + 1, 
                       cv2.LINE_AA)
        

        cv2.putText(img, text, position, 
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    def calculate_fps(self):

        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)

        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else fps
    
    def optimize_frame(self, frame):

        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        return frame
    
    def advanced_color_detection(self, frame, hsv):

        detected_areas = []
        
        for color_name, ranges in self.color_ranges.items():
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
                
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
       
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                min_area = 1500  
                
                if area > min_area:
                    hull = cv2.convexHull(contour)
                    
                    x, y, w, h = cv2.boundingRect(hull)
                    
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
        
        return detected_areas
    
    def create_rounded_mask(self, contours, img_shape):

        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for contour in contours:
            if len(contour) > 2:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(mask, [approx], 255)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask
    
    def detect_colors(self, frame):
        frame = self.optimize_frame(frame)
        
        frame = cv2.flip(frame, 1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        result = self.gradient_bg.copy()
        
        color_layer = np.zeros_like(frame, dtype=np.uint8)
        
        detected_areas = self.advanced_color_detection(frame, hsv)
        
        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']
            contour = area['contour']
            
            contour_mask = self.create_rounded_mask([contour], frame.shape)
            
            light_color = self.get_light_color(color, alpha=0.4)

            colored_area = np.full_like(frame, light_color, dtype=np.uint8)
            color_layer[contour_mask > 0] = colored_area[contour_mask > 0]

        result = cv2.addWeighted(frame, 0.7, color_layer, 0.3, 0)

        for area in detected_areas:
            name = area['name']
            center = area['center']
            size = area['size']
            color = area['color']

            font_scale = max(0.8, min(2.0, size / 150))
            thickness = max(1, int(size / 100))
            

            radius = max(30, size // 8)
            cv2.circle(result, center, radius, (40, 40, 40), -1)
            cv2.circle(result, center, radius, color, 2)
            

            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            self.put_pretty_text(result, name, (text_x, text_y), color, 
                               font_scale, thickness)
        
        return result, len(detected_areas)
    
    def add_ui_elements(self, frame, detected_count):

        h, w = frame.shape[:2]
        

        fps = self.calculate_fps()

        title_bg = np.zeros((60, w, 3), dtype=np.uint8)
        title_bg[:, :] = (40, 40, 60)
        frame[0:60, 0:w] = cv2.addWeighted(frame[0:60, 0:w], 0.3, title_bg, 0.7, 0)

        self.put_pretty_text(frame, "High-Performance Color Detection", (20, 40), 
                           (255, 255, 255), 1.2, 2)

        status_bg = np.zeros((40, w, 3), dtype=np.uint8)
        status_bg[:, :] = (30, 30, 30)
        frame[h-40:h, 0:w] = cv2.addWeighted(frame[h-40:h, 0:w], 0.4, status_bg, 0.6, 0)

        status_text = f"Detected: {detected_count} color areas | Press 'Q' to exit | 'S' to screenshot"
        self.put_pretty_text(frame, status_text, (20, h-10), 
                           (200, 200, 200), 0.6, 1)

        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
        self.put_pretty_text(frame, fps_text, (w - fps_size[0] - 20, 40), 
                           (0, 255, 255), 0.6, 1)

        res_text = f"Resolution: {w}x{h}"
        res_size = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
        self.put_pretty_text(frame, res_text, (w - res_size[0] - 20, 70), 
                           (0, 200, 255), 0.5, 1)

def main():

    detector = HighPerformanceColorDetector()
  
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera!")

    print("Setting camera to maximum resolution and frame rate...")
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    
 
    cap.set(cv2.CAP_PROP_FPS, 60)


    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")

    if actual_width != detector.target_width or actual_height != detector.target_height:
        detector.target_width = actual_width
        detector.target_height = actual_height
        detector.gradient_bg = detector.create_gradient_bg(actual_width, actual_height)
        print(f"Adjusted target resolution to: {actual_width}x{actual_height}")
    
    print("High-performance color detection started!")
    print("Press 'Q' to exit")
    print("Press 'S' to save screenshot")
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break
        
 
        result, count = detector.detect_colors(frame)
        detector.add_ui_elements(result, count)
        

        cv2.imshow('High-Performance Color Detection', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            
            screenshot_count += 1
            filename = f'color_detection_{screenshot_count}.png'
            cv2.imwrite(filename, result)
            print(f"Screenshot saved: {filename}")
    
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == "__main__":
    main()