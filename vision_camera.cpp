#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

// 定义颜色范围 (HSV格式)
map<string, Scalar> colorRangesLower = {
    {"红色", Scalar(0, 120, 70)},
    {"蓝色", Scalar(100, 120, 70)},
    {"绿色", Scalar(40, 120, 70)},
    {"黄色", Scalar(20, 120, 70)},
    {"紫色", Scalar(130, 120, 70)}
};

map<string, Scalar> colorRangesUpper = {
    {"红色", Scalar(10, 255, 255)},
    {"蓝色", Scalar(140, 255, 255)},
    {"绿色", Scalar(80, 255, 255)},
    {"黄色", Scalar(40, 255, 255)},
    {"紫色", Scalar(170, 255, 255)}
};

// 生成浅色版本
Scalar getLightColor(const Scalar& color) {
    return Scalar(
        min(color[0] + 100, 255.0),
        min(color[1] + 100, 255.0),
        min(color[2] + 100, 255.0)
    );
}

// 绘制美观的文字
void putPrettyText(Mat& image, const string& text, Point position, Scalar color, double fontScale = 1.0) {
    // 文字阴影效果
    putText(image, text, Point(position.x + 2, position.y + 2), 
            FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), 3, LINE_AA);
    
    // 主文字
    putText(image, text, position, 
            FONT_HERSHEY_SIMPLEX, fontScale, color, 2, LINE_AA);
}

int main() {
    cout<<"hello"


    // 打开前置摄像头（通常是索引1）
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头!" << endl;
        return -1;
    }
    
    // 设置摄像头分辨率
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    
    cout << "颜色识别程序启动成功!" << endl;
    cout << "按 'q' 键退出程序" << endl;
    
    Mat frame, hsv, mask, result;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "无法获取帧!" << endl;
            break;
        }
        
        // 水平翻转，让镜像看起来更自然
        flip(frame, frame, 1);
        
        // 转换为HSV颜色空间
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        
        // 创建结果图像
        result = frame.clone();
        
        // 用于存储检测到的颜色信息
        vector<pair<string, vector<Point>>> detectedColors;
        
        // 检测每种颜色
        for (const auto& color : colorRangesLower) {
            string colorName = color.first;
            Scalar lower = colorRangesLower[colorName];
            Scalar upper = colorRangesUpper[colorName];
            
            // 创建颜色掩码
            inRange(hsv, lower, upper, mask);
            
            // 形态学操作，去除噪声
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
            morphologyEx(mask, mask, MORPH_OPEN, kernel);
            morphologyEx(mask, mask, MORPH_CLOSE, kernel);
            
            // 查找轮廓
            vector<vector<Point>> contours;
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            // 处理每个轮廓
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                
                // 只处理大面积的颜色区域
                if (area > 5000) {
                    // 获取轮廓的边界矩形
                    Rect boundRect = boundingRect(contour);
                    
                    // 获取该颜色的浅色版本
                    Scalar baseColor;
                    if (colorName == "红色") baseColor = Scalar(0, 0, 255);
                    else if (colorName == "蓝色") baseColor = Scalar(255, 0, 0);
                    else if (colorName == "绿色") baseColor = Scalar(0, 255, 0);
                    else if (colorName == "黄色") baseColor = Scalar(0, 255, 255);
                    else if (colorName == "紫色") baseColor = Scalar(255, 0, 255);
                    
                    Scalar lightColor = getLightColor(baseColor);
                    
                    // 创建ROI的掩码
                    Mat roiMask = Mat::zeros(frame.size(), CV_8UC1);
                    vector<vector<Point>> contourArray = {contour};
                    fillPoly(roiMask, contourArray, Scalar(255));
                    
                    // 用浅色涂抹区域
                    Mat coloredArea(frame.size(), CV_8UC3, lightColor);
                    coloredArea.copyTo(result, roiMask);
                    
                    // 在区域中心添加文字标签
                    Point center(boundRect.x + boundRect.width/2, 
                                boundRect.y + boundRect.height/2);
                    
                    // 计算文字大小
                    int textHeight = max(30, boundRect.height / 8);
                    double fontScale = textHeight / 30.0;
                    
                    putPrettyText(result, colorName, center, baseColor, fontScale);
                    
                    detectedColors.push_back({colorName, contour});
                }
            }
        }
        
        // 添加状态信息
        string status = "检测到 " + to_string(detectedColors.size()) + " 种颜色";
        putPrettyText(result, status, Point(20, 40), Scalar(255, 255, 255), 0.8);
        putPrettyText(result, "按 'q' 退出", Point(20, 80), Scalar(200, 200, 200), 0.6);
        
        // 显示结果
        imshow("颜色识别 - 浅色涂抹模式", result);
        
        // 退出条件
        char key = waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    cout << "程序已退出" << endl;
    return 0;
}