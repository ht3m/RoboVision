# 文件名：get_current_pose.py （独立专用，单独运行）
import socket
import struct
import numpy as np

# 你的机器人静态IP（不变）
HOST = "169.254.162.96"
PORT = 30003

def get_robot_current_pose():
    """
    快捷获取机器人当前位姿
    返回格式：[x, y, z, rx, ry, rz]
    单位：米(m)、弧度(rad)
    """
    try:
        # 创建TCP连接（UR实时端口）
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PORT))
        
        # 接收机器人实时数据（固定1108字节）
        data = s.recv(1108)
        
        # 核心：444偏移量 → 读取6个双精度浮点数 = TCP位姿
        # x, y, z, rx, ry, rz
        pose = struct.unpack('!6d', data[444:444+48])
        s.close()
        
        # 转为numpy数组，保留4位小数（整洁好看）
        pose = np.round(pose, 4)
        return pose.tolist()
    
    except Exception as e:
        return f"连接失败！错误：{str(e)}"

# ==================== 主程序：实时显示位姿 ====================
if __name__ == '__main__':
    print("="*60)
    print(" UR5 实时位姿读取器 (格式：x,y,z,rx,ry,rz) ")
    print(" 移动示教器，数值自动刷新 ")
    print("="*60)
    
    # 循环实时读取（按 Ctrl+C 停止）
    while True:
        current_pose = get_robot_current_pose()
        print(f"\r当前位姿：{current_pose}", end="")
