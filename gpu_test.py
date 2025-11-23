import subprocess
import sys

def check_and_install_opencv_with_cuda():
    """检查并安装支持CUDA的OpenCV"""
    try:
        # 检查当前OpenCV版本
        import cv2
        print(f"当前OpenCV版本: {cv2.__version__}")
        
        # 检查CUDA支持
        if hasattr(cv2, 'cuda'):
            print("OpenCV已编译CUDA支持")
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"检测到 {cuda_count} 个CUDA设备")
            return True
        else:
            print("当前OpenCV版本不支持CUDA")
            
    except ImportError:
        print("OpenCV未安装")
    
    # 询问用户是否要安装支持CUDA的版本
    response = input("是否要尝试安装支持CUDA的OpenCV? (y/n): ")
    if response.lower() == 'y':
        try:
            # 卸载当前版本
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
            
            # 安装支持CUDA的版本
            print("正在安装支持CUDA的OpenCV...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-contrib-python"])
            
            print("安装完成，请重新运行程序")
            return True
        except Exception as e:
            print(f"安装失败: {e}")
            return False
    return False

# 运行检查
if __name__ == "__main__":
    check_and_install_opencv_with_cuda()