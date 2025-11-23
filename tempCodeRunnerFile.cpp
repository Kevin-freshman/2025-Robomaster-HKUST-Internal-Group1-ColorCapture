#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <cctype>

// 工具函数：去除字符串中的空格和中括号
std::string cleanString(const std::string &input) {
    std::string result;
    for (char c : input) {
        if (std::isdigit(c) || c == ',' || c == '-') {
            result.push_back(c);
        }
    }
    return result;
}

// 从文件中读取 block_states.txt
std::vector<int> readBlockStates(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<int> states;

    if (!file.is_open()) {
        std::cerr << "[警告] 无法打开文件: " << filename << std::endl;
        return states;
    }

    std::string content;
    std::getline(file, content);
    file.close();

    if (content.empty()) return states;

    // 清理字符串，只保留数字、逗号、负号
    content = cleanString(content);

    // 按逗号分割
    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            states.push_back(std::stoi(token));
        } catch (...) {
            // 跳过非数字
        }
    }

    return states;
}

// 主循环
int main() {
    const std::string filename = "block_states.txt";
    std::vector<int> lastStates;

    std::cout << "🟢 Block Reader Started." << std::endl;
    std::cout << "实时读取 " << filename << " 并判断矿物状态..." << std::endl;
    //std::cout << "hi"<<std::endl;
    while (true) {
        std::vector<int> currentStates = readBlockStates(filename);

        if (!currentStates.empty() && currentStates != lastStates) {
            std::cout << "\n检测到新状态: [ ";
            for (int s : currentStates) std::cout << s << " ";
            std::cout << "]" << std::endl;

            for (size_t i = 0; i < currentStates.size(); ++i) {
                int state = currentStates[i];
                std::cout << "方块 " << i + 1 << "：";
                if (state == 1)
                    std::cout << "🟢 有矿物" << std::endl;
                else if (state == 0)
                    std::cout << "⚪ 没有矿物" << std::endl;
                else
                    std::cout << "❌ 未检测" << std::endl;
            }

            lastStates = currentStates;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 每0.5秒检查一次
    }

    return 0;
}
