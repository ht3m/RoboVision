"""
robot_arm.py — UR5 机械臂实时位姿读取模块
=========================================
通过 TCP 连接读取 UR5 的当前 TCP 位姿。
返回格式: [x, y, z, rx, ry, rz] (单位: 米, 弧度)
并附带齐次变换矩阵 (4x4)。
"""

import socket
import struct
import numpy as np


def get_current_pose(host: str, port: int = 30003) -> np.ndarray:
    """读取机械臂当前 TCP 位姿。

    Args:
        host: 机器人 IP 地址
        port: UR 实时数据端口 (默认 30003)

    Returns:
        pose: shape (6,) → [x, y, z, rx, ry, rz]
              单位: 米 (m)、弧度 (rad)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect((host, port))

    data = s.recv(1108)
    s.close()

    # 偏移 444 字节，读取 6 个 double (大端序)
    pose = struct.unpack('!6d', data[444:444 + 48])
    return np.array(pose, dtype=np.float64)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """将 [x, y, z, rx, ry, rz] 转换为 4x4 齐次变换矩阵。

    旋转部分使用 UR 的旋转向量 (rx, ry, rz) 经 Rodrigues 公式得到 3x3 旋转矩阵。
    注意: UR 中 rx, ry, rz 是旋转向量 (axis-angle)，不是欧拉角。

    Args:
        pose: (6,) → [x, y, z, rx, ry, rz]

    Returns:
        T: (4, 4) 齐次矩阵
    """
    x, y, z = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]

    # Rotation vector → rotation matrix (Rodrigues)
    theta = np.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        R = np.eye(3)
    else:
        k = np.array([rx, ry, rz]) / theta
        K = np.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ])
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    return T
