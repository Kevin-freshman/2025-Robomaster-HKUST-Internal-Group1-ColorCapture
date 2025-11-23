# 🎨 Optimized Color Detection System

> **Real-time color detection with mineral recognition and serial communication**  
> *Author: Kevin*  
> *Making computers see colors like artists!* 🎨

---

## 🚀 Overview

This Python application is a sophisticated real-time color detection system that uses your webcam to identify multiple colors, detect minerals above colored blocks, and communicate with external hardware via serial port. It's like giving your computer rainbow-colored glasses! 🌈

### Key Features:
- **7-Color Detection** (Red, Green, Blue, Gold, Silver, Black, Mineral Gray)
- **Real-time Processing** with performance optimization
- **Mineral Detection** above red/blue blocks with visual indicators
- **Serial Communication** for hardware integration
- **File I/O** for inter-process communication
- **Interactive Controls** for screenshots and performance tuning

---

## 🛠️ Technical Details

### Core Technologies Used:
- **OpenCV** - Computer vision and image processing
- **NumPy** - Mathematical operations and array handling
- **PySerial** - Serial communication with external devices
- **Collections.deque** - Efficient data buffering

### Key Parameters & Configuration:

#### 🎯 Color Detection Ranges (HSV Space):
```python
self.color_ranges = {
    'Red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
    'Green': [([35, 50, 50], [95, 255, 255])],
    'Blue': [([90, 50, 50], [135, 255, 255])],
    'Gold': [([15, 80, 80], [35, 255, 255])],
    'Silver': [([0, 0, 100], [180, 50, 230])],
    'Black': [([0, 0, 0], [180, 255, 60])],
    'Mineral_Gray': [([0, 0, 20], [180, 60, 210])]
}
```

#### ⚙️ Performance Parameters:
- `processing_scale = 0.4` - Scale factor for faster processing
- `target_width = 1920`, `target_height = 1080` - Camera resolution
- `fps_queue = deque(maxlen=30)` - FPS smoothing buffer
- `min_area = 400` - Minimum contour area to consider

#### 🧹 Morphological Kernels:
- **Opening Kernel**: 7x7 ellipse (removes noise)
- **Closing Kernel**: 10x10 ellipse (fills gaps)
- **Gold Kernel**: 5x5 ellipse (special treatment for gold)

---

## 📁 File Outputs

### Generated Files:
1. **`color_status.txt`** - Current detection status
2. **`block_states.txt`** - Array format: `[block1, block2, block3, block4, mineral1, mineral2]`
   - Block values: `0=Green, 1=Red/Blue, 2=Empty`
   - Mineral values: `0=No mineral, 1=Mineral detected`

### Serial Communication:
- **Baud Rate**: 9600
- **Format**: Python list as string
- **Port Scanning**: Automatically scans COM1-COM19
- **Data Buffering**: Sends most common state from last 100 frames

---

## 🎮 Controls

| Key | Action | Description |
|-----|--------|-------------|
| `Q` | Quit | Exit the application |
| `S` | Screenshot | Save current frame as PNG |
| `+` | Zoom In | Increase processing scale (up to 1.0) |
| `-` | Zoom Out | Decrease processing scale (down to 0.2) |

---

## 🔧 Installation & Setup

### Prerequisites:
```bash
pip install opencv-python numpy pyserial
```

### Hardware Requirements:
- Webcam (USB or built-in)
- Optional: Arduino/Serial device for hardware integration

### Running the Application:
```bash
python color_detector.py
```

---

## 🎯 Key Algorithms & Techniques

### 1. **Color Space Conversion**
- Converts BGR → HSV for better color separation
- Uses median blurring to reduce noise

### 2. **Morphological Operations**
- **Opening**: Erosion followed by dilation (removes small noise)
- **Closing**: Dilation followed by erosion (fills small holes)
- Different kernels for different colors based on detection characteristics

### 3. **Contour Detection**
- Finds external contours in binary masks
- Uses convex hull for smoother shapes
- Filters by area to remove noise

### 4. **Mineral Detection**
- Scans region above red/blue blocks
- Combines gray and black detection
- Uses moment calculations for center detection
- Requires >15% mineral pixels in ROI

### 5. **Performance Optimization**
- Frame scaling for faster processing
- FPS smoothing with deque buffers
- Efficient contour processing

---

## 📊 Performance Tips

### For Better Detection:
- **Good Lighting**: Ensure consistent, bright lighting
- **Contrast**: Use high-contrast backgrounds
- **Camera Quality**: Higher resolution cameras work better
- **Color Calibration**: Adjust HSV ranges for your environment

### For Higher FPS:
- **Lower `processing_scale`**: 0.2-0.4 for maximum speed
- **Reduce Resolution**: Lower `target_width` and `target_height`
- **Simplify Detection**: Reduce number of colors being tracked

---

## 🐛 Troubleshooting

### Common Issues:

1. **"Cannot open camera!"**
   - Check camera connections
   - Ensure no other applications are using the camera

2. **Poor Color Detection**
   - Adjust HSV ranges for your lighting conditions
   - Increase `min_area` to filter noise
   - Check camera white balance

3. **Serial Port Not Found**
   - Verify Arduino is connected and powered
   - Check device manager for correct COM port
   - Update PySerial installation

4. **Low FPS**
   - Reduce processing scale with `-` key
   - Lower camera resolution in code
   - Close other resource-intensive applications

---

## 🎓 Learning Resources

### OpenCV Concepts:
- [HSV Color Space](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Contour Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)

### Serial Communication:
- [PySerial Documentation](https://pyserial.readthedocs.io/)
- [Arduino-Python Communication](https://create.arduino.cc/projecthub/ansh2919/serial-communication-between-python-and-arduino-7e7c6f)

---

## 📝 License & Contributions

This project is open for educational purposes. Feel free to:
- Modify and adapt for your needs
- Improve detection algorithms
- Add new color profiles
- Enhance serial communication

---

## 🎨 Pro Tips

1. **Gold Detection**: Works best in warm, consistent lighting
2. **Mineral Detection**: Requires good vertical separation between blocks and minerals
3. **Serial Data**: Implements majority voting to reduce noise in communications
4. **Performance**: The system automatically scales processing based on your hardware

---

*Happy color detecting! May your minerals always be found and your FPS always be high!* 🚀🎯
