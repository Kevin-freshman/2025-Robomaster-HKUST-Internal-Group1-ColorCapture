#include <iostream>
#include <windows.h>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <array>
#include <cctype>  // 用于 std::isspace

std::string find_first_com_port() {
    for (int i = 1; i <= 256; ++i) {
        std::string portName = "\\\\.\\COM" + std::to_string(i);
        HANDLE hCom = CreateFileA(portName.c_str(), GENERIC_READ | GENERIC_WRITE,
                                  0, NULL, OPEN_EXISTING, 0, NULL);
        if (hCom != INVALID_HANDLE_VALUE) {
            CloseHandle(hCom);
            return "COM" + std::to_string(i);
        }
    }
    return "";
}

HANDLE open_serial_port(const std::string &portName) {
    std::string fullName = "\\\\.\\" + portName;
    HANDLE hCom = CreateFileA(fullName.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (hCom == INVALID_HANDLE_VALUE) {
        std::cerr << "无法打开串口 " << portName << std::endl;
        return INVALID_HANDLE_VALUE;
    }

    DCB dcb = {0};
    GetCommState(hCom, &dcb);
    dcb.BaudRate = CBR_9600;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    if (!SetCommState(hCom, &dcb)) {
        std::cerr << "设置串口参数失败。" << std::endl;
        CloseHandle(hCom);
        return INVALID_HANDLE_VALUE;
    }

    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutConstant = 50;
    SetCommTimeouts(hCom, &timeouts);

    std::cout << "串口 " << portName << " 已打开 (9600 8N1)" << std::endl;
    return hCom;
}

void send_loop(HANDLE hCom, const std::array<double, 4>& quaternion) {
    std::stringstream ss;
    ss.precision(6);  // 设置精度以避免科学计数法
    ss << quaternion[0] << " " << quaternion[1] << " " << quaternion[2] << " " << quaternion[3];
    const std::string msg = ss.str();
    DWORD bytesWritten = 0;

    while (true) {
        BOOL success = WriteFile(hCom, msg.c_str(), msg.size(), &bytesWritten, NULL);
        if (success && bytesWritten == msg.size()) {
            std::cout << "发送成功: " << msg << std::endl;
        } else {
            std::cerr << "发送失败。" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    std::string port = find_first_com_port();
    if (port.empty()) {
        std::cerr << "未检测到任何可用的 COM 端口。" << std::endl;
        return 0;
    }

    HANDLE hCom = open_serial_port(port);
    if (hCom == INVALID_HANDLE_VALUE)
        return 0;

    // 读取 101.txt 中的四元数（格式如 [1, 1, -1, -1]）
    std::array<double, 4> quaternion = {0.0, 0.0, 0.0, 0.0};
    std::ifstream file("101.txt");
    if (!file) {
        std::cerr << "无法打开文件 101.txt" << std::endl;
        CloseHandle(hCom);
        return 0;
    }

    std::string line;
    if (std::getline(file, line)) {
        // 去除前后空格
        size_t start = line.find_first_not_of(" \t");
        size_t end = line.find_last_not_of(" \t");
        if (start == std::string::npos || end == std::string::npos) {
            std::cerr << "文件内容为空。" << std::endl;
            CloseHandle(hCom);
            return 0;
        }
        line = line.substr(start, end - start + 1);

        // 检查并去除方括号
        if (line.front() == '[' && line.back() == ']') {
            line = line.substr(1, line.size() - 2);
        } else {
            std::cerr << "文件格式错误：缺少方括号。" << std::endl;
            CloseHandle(hCom);
            return 0;
        }

        // 使用 stringstream 解析逗号分隔的数字（允许逗号后有空格）
        std::stringstream ss(line);
        char comma;
        if (!(ss >> quaternion[0] >> comma >> quaternion[1] >> comma >> quaternion[2] >> comma >> quaternion[3]) || comma != ',') {
            std::cerr << "文件格式错误，无法解析四个数字（预期格式：[x, y, z, w]）。" << std::endl;
            CloseHandle(hCom);
            return 0;
        }
    } else {
        std::cerr << "文件为空或读取失败。" << std::endl;
        CloseHandle(hCom);
        return 0;
    }

    send_loop(hCom, quaternion);

    CloseHandle(hCom);
    return 0;
}
    
    //hi