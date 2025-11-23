"""
🎯 COLOR DETECTION & SERIAL COMMUNICATION SYSTEM 🤖
Author: Kevin

What this bad boy does:
- Uses your webcam to detect multiple colors in real-time (Red, Green, Blue, Gold, Silver, Black, Mineral Gray)
- Finds color blobs and draws fancy contours around them
- Detects minerals above red/blue blocks (because minerals are fancy like that)
- Saves detected states to files for other programs to read
- Talks to Arduino via serial port to send color data
- Makes you look like a computer vision wizard! 🧙‍♂️

Features:
- Optimized processing with scalable resolution
- Real-time FPS display (to impress your friends)
- Screenshot capability (for bragging rights)
- Morphological operations to clean up detection
- Mineral detection with arrows (pointing is caring)
- Serial communication with automatic port scanning

Pro tip: Press 'q' to quit, 's' for screenshot, '+'/- to adjust processing scale!
"""

import cv2
import numpy as np
import time
from collections import deque
import os
import serial

class OptimizedColorDetector:
    """
    The main brain behind all the color magic! 🎨
    This class does everything from detecting colors to talking with hardware.
    """
    
    def __init__(self):
        # 🎯 HSV color ranges for our color palette
        # Because we're artists, not just programmers!
        self.color_ranges = {
            'Red': [
                ([0, 120, 70], [10, 255, 255]),  # Red is complicated - it wraps around hue!
                ([170, 120, 70], [180, 255, 255])
            ],
            'Green': [([35, 50, 50], [95, 255, 255])],  # Green means go! 🚦
            'Blue': [([90, 50, 50], [135, 255, 255])],  # Feeling blue? We'll detect it!
            'Gold': [([15, 80, 80], [35, 255, 255])],   # Shiny! ✨
            'Silver': [([0, 0, 100], [180, 50, 230])],  # The robot's favorite
            'Black': [([0, 0, 0], [180, 255, 60])],     # The absence of light, but we see it!
            'Mineral_Gray': [([0, 0, 20], [180, 60, 210])]  # For those precious minerals! 💎
        }

        # 🎨 BGR colors for drawing (because OpenCV is backwards like that)
        self.color_bgr = {
            'Red': (0, 0, 255),        # Blue in RGB, Red in BGR - mind blown! 🤯
            'Green': (0, 255, 0),      # Actually green this time
            'Blue': (255, 0, 0),       # Red in RGB, Blue in BGR - still confused?
            'Gold': (0, 215, 255),     # Golden shower of pixels! 🌟
            'Silver': (192, 192, 192), # 50 shades of gray, but just one here
            'Black': (50, 50, 50),     # Not quite black, but close enough for government work
            'Mineral_Gray': (128, 128, 128) # Perfectly balanced, as all things should be
        }

        # ⚙️ Performance tuning parameters
        self.processing_scale = 0.4    # Process smaller frames for SPEED! 🚀
        self.target_width = 1920       # Camera resolution goals
        self.target_height = 1080

        # 📊 Performance monitoring
        self.fps_queue = deque(maxlen=30)      # Keep track of FPS like a hawk
        self.prev_time = time.time()           # For that sweet, sweet FPS calculation
        self.processing_times = deque(maxlen=30) # How long we take to think

        # 🧹 Morphological kernels for cleaning up our detections
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))    # Opening: break up small connections
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # Closing: fill small holes
        self.kernel_gold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))    # Special treatment for gold!

        # 💾 File I/O for communicating with other programs
        self.status_file = "color_status.txt"  # Where we write our findings
        self.block_file = "block_states.txt"   # Block states for the world to see
        self.last_status = None                # Don't repeat yourself!

        # 🔌 Serial communication setup
        self.ser = None
        baud_rate = 9600  # The classic baud rate - old but gold!
        
        print("🔍 Scanning for available serial ports among (COM1 - COM19)...")
        print("   Because guessing is half the fun! 🎲")

        # Try all the COM ports like trying keys on a keychain
        for i in range(10, 20):
            port_name = f'COM{i}'
            try:
                temp_ser = serial.Serial(port_name, baud_rate, timeout=1, write_timeout=1)
                self.ser = temp_ser
                print(f"--------------------------------")
                print(f"✅ Connected to the serial port: {port_name}")
                print(f"   Baud rate: {baud_rate} (fast enough for government work!)")
                print(f"--------------------------------")
                break  # Found one! No need to keep looking
            except serial.SerialException:
                continue  # Try the next one, no biggie

        # 😞 If we can't find any serial ports
        if self.ser is None:
            print("⚠️  Unable to locate the serial port among (COM1-COM19)")
            print("   Program will continue as the serial function is dead (lol) 🤷‍♂️")
            print("   Check your connections or prepare for manual mode!")

        print("🎉 Initialization complete! Ready to detect some colors! 🌈")
        print("🎯 Target colors: Red, Green, Blue, Gold, Silver, Black")

    def optimize_frame(self, frame):
        """
        Make frames smaller for faster processing! 🚀
        Because sometimes size DOES matter (smaller is faster!)
        """
        display_frame = frame.copy()  # Keep original for showing off
        
        # Shrink the frame if we're not processing at full size
        if self.processing_scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.processing_scale)   # Math magic! ✨
            new_h = int(h * self.processing_scale)
            process_frame = cv2.resize(frame, (new_w, new_h))
        else:
            process_frame = frame.copy()  # Full size, you brave soul!
            
        return display_frame, process_frame

    def enhance_color_detection(self, hsv, color_name):
        """
        The secret sauce for clean color detection! 🍝
        Takes HSV image and color name, returns a clean mask.
        This is where the magic happens! 🎩✨
        """
        ranges = self.color_ranges[color_name]
        mask = None
        
        # Handle multiple ranges for tricky colors like Red
        for lower, upper in ranges:
            lower_np = np.array(lower)  # Convert to numpy because computers like arrays
            upper_np = np.array(upper)
            temp_mask = cv2.inRange(hsv, lower_np, upper_np)  # The actual color detection!
            
            if mask is None:
                mask = temp_mask  # First range
            else:
                mask = cv2.bitwise_or(mask, temp_mask)  # Combine ranges
        
        if mask is None:
            return None  # No color found, sad! 😢

        # 🧹 Special cleaning for each color (because they're all special snowflakes)
        if color_name == 'Red':
            kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_red)   # Break up noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_red)  # Fill gaps
        elif color_name == 'Gold':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_gold)   # Gold deserves special treatment!
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_gold)
        elif color_name in ['Silver', 'Black', 'Mineral_Gray']:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)  # Basic cleaning
        elif color_name == 'Green':
            kernel_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_g)  # Green needs gentle touch
        else:
            # Standard cleaning for other colors
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
        return mask

    def detect_mineral_above(self, hsv_full, x, y, w, h):
        """
        Look for minerals above colored blocks! 💎
        Because minerals love to hang out above red and blue blocks!
        """
        roi_h = int(h * 0.8)  # Region of Interest height (80% of block height)
        roi_y_start = max(0, y - roi_h - 10)  # Don't go outside the image!
        roi_y_end = max(0, y - 5)
        
        # Make sure our ROI makes sense
        if roi_y_start >= roi_y_end:
            return False, None  # Can't look for minerals here! 😔

        # Extract the region above the block
        roi_hsv = hsv_full[roi_y_start:roi_y_end, x:x+w]
        
        # Look for both gray and black (minerals come in different flavors)
        gray_mask = self.enhance_color_detection(roi_hsv, 'Mineral_Gray')
        black_mask = self.enhance_color_detection(roi_hsv, 'Black')

        # Combine the masks like a boss! 🦸‍♂️
        mineral_mask = None
        if gray_mask is not None:
            mineral_mask = gray_mask
        if black_mask is not None:
            if mineral_mask is None:
                mineral_mask = black_mask
            else:
                mineral_mask = cv2.bitwise_or(mineral_mask, black_mask)  # Either gray OR black
        
        # Did we find anything interesting?
        if mineral_mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mineral_mask = cv2.morphologyEx(mineral_mask, cv2.MORPH_OPEN, kernel)  # Clean it up!
            
            # Count the mineral pixels
            count = cv2.countNonZero(mineral_mask)
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            
            # If we have enough mineral pixels, we found one! 🎉
            if count > (total_pixels * 0.15) and count > 50:
                M = cv2.moments(mineral_mask)  # Calculate center of mass
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    center_x_global = x + center_x    # Convert to global coordinates
                    center_y_global = roi_y_start + center_y
                    return True, (center_x_global, center_y_global)  # Success! 🏆
        
        return False, None  # No minerals today 😢

    def update_status_file(self, status):
        """
        Write status to file, but only if it changed!
        Because writing the same thing over and over is boring! 😴
        """
        if status != self.last_status:
            try:
                with open(self.status_file, 'w') as f:
                    f.write(str(status))
                self.last_status = status
            except Exception as e:
                print(f"💥 File error: {e}")

    def update_block_file(self, detected_areas):
        """
        Update the block states file and send data via serial! 📡
        This is where we talk to other programs and hardware!
        """
        # Priority system: Red and Blue are important, Green is... green
        color_priority = {'Red': 1, 'Blue': 1, 'Green': 0}
        
        # Filter and sort by size (bigger is better! 📏)
        filtered = [a for a in detected_areas if a['name'] in color_priority]
        filtered.sort(key=lambda a: a['size'], reverse=True)
        top4 = filtered[:4]  # Take the 4 biggest ones
        top4.sort(key=lambda a: a['center'][0])  # Sort by X position

        # Handle red and blue blocks separately for mineral detection
        rb_areas = [a for a in detected_areas if a['name'] in ['Red', 'Blue']]
        rb_areas.sort(key=lambda a: a['center'][0])  # Sort by X position

        # Build the state arrays
        block_states = [color_priority[a['name']] for a in top4]
        while len(block_states) < 4:
            block_states.append(2)  # Fill with 2 if we don't have enough blocks

        mineral_states = []
        for area in rb_areas[:2]:  # Check first two red/blue blocks
            mineral_states.append(1 if area.get('has_mineral', False) else 0)

        while len(mineral_states) < 2:
            mineral_states.append(0)  # Fill with 0 if no minerals

        # Combine everything into one happy array! 🎉
        combined_states = block_states + mineral_states

        try:
            # Write to file for other programs to read
            with open(self.block_file, 'w') as f:
                f.write(str(combined_states))
            print(f"📝 Updated block file: {combined_states}")
            
            # Buffer states for serial transmission (we're efficient like that! 💪)
            if not hasattr(self, "_state_buffer"):
                self._state_buffer = []

            self._state_buffer.append(tuple(combined_states))

            # When we have enough data, send the most common state via serial
            if len(self._state_buffer) >= 100 and self.ser and self.ser.is_open:
                # Find the most frequent state (democracy in action! 🗳️)
                freq = {}
                for s in self._state_buffer:
                    freq[s] = freq.get(s, 0) + 1
                most_common_states = max(freq.items(), key=lambda kv: kv[1])[0]

                # Send it over serial! 📡
                data_to_send = f"{list(most_common_states)}\n"
                self.ser.write(data_to_send.encode("utf-8"))
                print(f"📡 Sent to serial (most common in last 100 frames): {data_to_send.strip()}")

                self._state_buffer.clear()  # Clear buffer for next batch

        except Exception as e:
            print(f"💥 Error updating block file / serial transmission: {e}")

    def fast_color_detection(self, process_frame, original_dims):
        """
        The main color detection workhorse! 🐎
        Finds all the colors, draws contours, and makes everything look pretty!
        """
        start_time = time.time()  # Start the clock! ⏱️
        detected_areas = []
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV space
        hsv = cv2.medianBlur(hsv, 3)  # Smooth out the noise
        
        # Calculate scaling factors if we're processing at reduced size
        scale_x, scale_y = 1.0, 1.0
        if self.processing_scale != 1.0:
            scale_x = original_dims[0] / process_frame.shape[1]
            scale_y = original_dims[1] / process_frame.shape[0]

        # Check each color in our palette
        for color_name in self.color_ranges.keys():
            mask = self.enhance_color_detection(hsv, color_name)
            if mask is None:
                continue  # No pixels of this color found
                
            # Find contours (the shapes of our color blobs)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 400  # Ignore tiny blobs (they're probably noise!)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:  # Big enough to care about!
                    hull = cv2.convexHull(contour)  # Smooth out the shape
                    x, y, w, h = cv2.boundingRect(hull)  # Get bounding box
                    
                    # Check for minerals above red/blue blocks
                    has_mineral = False
                    mineral_pos = None
                    if color_name in ['Red', 'Blue']:
                        has_mineral, mineral_pos = self.detect_mineral_above(hsv, x, y, w, h)
                    
                    # Scale coordinates back to original size if needed
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
                    
                    # Calculate center of the detected area
                    center_x = orig_x + orig_w // 2
                    center_y = orig_y + orig_h // 2
                    
                    # Store all the info about this detection
                    detected_areas.append({
                        'name': color_name,
                        'center': (center_x, center_y),
                        'size': max(orig_w, orig_h),
                        'color': self.color_bgr[color_name],
                        'contour': hull,
                        'has_mineral': has_mineral,
                        'mineral_center': mineral_pos
                    })

        # Update the block file with our findings
        self.update_block_file(detected_areas)
        
        # Calculate how long this took us
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return detected_areas

    def calculate_fps(self):
        """
        Calculate FPS because everyone loves big numbers! 🔢
        """
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)  # Basic FPS calculation
        self.prev_time = current_time
        self.fps_queue.append(fps)
        # Return average FPS for smoother display
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else fps

    def process_frame(self, frame):
        """
        Process a single frame - the main pipeline! 🎬
        Takes a frame, returns annotated frame and detection count.
        """
        # Optimize frame for processing
        display_frame, process_frame = self.optimize_frame(frame)
        display_frame = cv2.flip(display_frame, 1)  # Mirror effect
        process_frame = cv2.flip(process_frame, 1)
        original_dims = (display_frame.shape[1], display_frame.shape[0])
        
        # Detect all the colors! 🌈
        detected_areas = self.fast_color_detection(process_frame, original_dims)
        result = display_frame.copy()  # Start with original frame
        
        # Draw all the detections
        for area in detected_areas:
            cv2.drawContours(result, [area['contour']], -1, area['color'], 2)  # Draw contour
            cv2.putText(result, area['name'], area['center'],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, area['color'], 2)  # Label it
            
            # If there's a mineral, draw it with an arrow! ➡️
            if area.get('has_mineral', False):
                mineral_center = area.get('mineral_center', area['center'])
                cv2.circle(result, mineral_center, 10, self.color_bgr['Mineral_Gray'], -1)  # Mineral dot
                cv2.putText(result, "MINERAL", (mineral_center[0]-30, mineral_center[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Mineral label
                cv2.arrowedLine(result, mineral_center, area['center'], 
                               self.color_bgr['Mineral_Gray'], 2)  # Point to block

        return result, len(detected_areas)

    def add_performance_info(self, frame, detected_count):
        """
        Add performance stats to the frame because data is beautiful! 📊
        """
        h, w = frame.shape[:2]
        fps = self.calculate_fps()
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # Add all the performance text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # FPS in yellow
        cv2.putText(frame, f"Detected: {detected_count}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Count in white
        cv2.putText(frame, f"Process: {avg_processing_time:.1f}ms",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)  # Time in orange


def main():
    """
    The main function where the magic begins! 🎪
    Sets up everything and runs the main loop.
    """
    print("🎬 Starting Color Detection System...")
    detector = OptimizedColorDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("💥 Cannot open camera! Check your connections! 🔌")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.target_height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Aim for 60 FPS because we're fancy! 💫

    screenshot_count = 0  # For naming screenshots
    
    try:
        print("🚀 Starting main loop... Press 'q' to quit, 's' for screenshot!")
        while True:
            # Read frame from camera
            ret, frame0 = cap.read()
            frame = cv2.flip(frame0, 1)  # Mirror the frame
            if not ret:
                print("💥 Failed to grab frame! Camera might be disconnected.")
                break
            
            # Process the frame and get results
            result, count = detector.process_frame(frame)
            detector.add_performance_info(result, count)  # Add performance stats
            
            # Show the result
            cv2.imshow('🎨 Optimized Color Detection System', result)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                print("👋 Quitting... Thanks for playing!")
                break
            elif key in [ord('s'), ord('S')]:
                screenshot_count += 1
                filename = f'color_detection_{screenshot_count}.png'
                cv2.imwrite(filename, result)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('+'):
                detector.processing_scale = min(1.0, detector.processing_scale + 0.1)
                print(f"🔍 Scale increased to: {detector.processing_scale}")
            elif key == ord('-'):
                detector.processing_scale = max(0.2, detector.processing_scale - 0.1)
                print(f"🔍 Scale decreased to: {detector.processing_scale}")

    except KeyboardInterrupt:
        print("🛑 User interrupted with Ctrl+C")

    finally:
        # Clean up everything like a good citizen! 🧹
        print("🧹 Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Close serial connection if open
        if detector.ser and detector.ser.is_open:
            detector.ser.close()
            print("🔌 Serial port closed")
        
        print("🎉 Program exited gracefully. Come back soon! 👋")


if __name__ == "__main__":
    main()