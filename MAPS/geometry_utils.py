"""
几何计算工具函数

提供MAPS算法所需的各种几何计算功能：
- 向量角度和长度计算
- 三角形面积计算（2D/3D）
- 点在三角形内的判断
- 重心坐标计算
- 顶点曲率估算
- 距离和投影计算

这些函数为共形映射、三角化和简化算法提供基础支持。
"""

import numpy as np
import math
from typing import List, Tuple


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角（弧度）
    
    Args:
        v1, v2: 三维向量
    
    Returns:
        角度（弧度）
    """
    # 计算向量的模长
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算点积并限制在[-1, 1]范围内
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return math.acos(cos_angle)


def triangle_area_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三维空间中三角形的面积
    
    Args:
        p1, p2, p3: 三角形的三个顶点坐标
    
    Returns:
        三角形面积
    """
    # 使用叉积计算面积
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross)


def triangle_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算二维平面中三角形的面积
    
    Args:
        p1, p2, p3: 三角形的三个顶点坐标 (2D)
    
    Returns:
        三角形面积
    """
    # 使用向量叉积的z分量
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])


def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两点之间的欧几里得距离"""
    return np.linalg.norm(p2 - p1)


def compute_ring_angles(center: np.ndarray, ring_vertices: List[np.ndarray]) -> List[float]:
    """
    计算环形邻域中相邻边之间的角度
    
    Args:
        center: 中心顶点坐标
        ring_vertices: 环形邻域顶点坐标列表
    
    Returns:
        角度列表（弧度）
    """
    angles = []
    n = len(ring_vertices)
    
    for i in range(n):
        prev_vertex = ring_vertices[i - 1] if i > 0 else ring_vertices[n - 1]
        curr_vertex = ring_vertices[i]
        
        # 计算从中心到相邻顶点的向量
        v_prev = prev_vertex - center
        v_curr = curr_vertex - center
        
        # 计算角度
        angle = vector_angle(v_prev, v_curr)
        angles.append(angle)
    
    return angles


def compute_vertex_curvature(center: np.ndarray, ring_vertices: List[np.ndarray]) -> float:
    """
    计算顶点的近似高斯曲率
    
    Args:
        center: 中心顶点坐标
        ring_vertices: 环形邻域顶点坐标列表
    
    Returns:
        近似高斯曲率
    """
    if len(ring_vertices) < 3:
        return 0.0
    
    # 计算环形邻域的角度
    angles = compute_ring_angles(center, ring_vertices)
    
    # 计算总角度
    total_angle = sum(angles)
    
    # 计算面积（使用环形邻域三角形的总面积）
    total_area = 0.0
    n = len(ring_vertices)
    
    for i in range(n):
        next_vertex = ring_vertices[(i + 1) % n]
        curr_vertex = ring_vertices[i]
        area = triangle_area_3d(center, curr_vertex, next_vertex)
        total_area += area
    
    if total_area == 0:
        return 0.0
    
    # 近似高斯曲率 = (2π - 总角度) / (总面积 / 3)
    curvature = (2 * math.pi - total_angle) / (total_area / 3.0)
    return curvature


def point_in_triangle_2d(point: np.ndarray, triangle: List[np.ndarray]) -> bool:
    """
    检查点是否在二维三角形内
    使用重心坐标方法
    
    Args:
        point: 要检查的点 (2D)
        triangle: 三角形的三个顶点 (2D)
    
    Returns:
        True如果点在三角形内
    """
    if len(triangle) != 3:
        return False
    
    p1, p2, p3 = triangle
    
    # 计算重心坐标
    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    
    if abs(denom) < 1e-10:
        return False
    
    alpha = ((p2[1] - p3[1]) * (point[0] - p3[0]) + (p3[0] - p2[0]) * (point[1] - p3[1])) / denom
    beta = ((p3[1] - p1[1]) * (point[0] - p3[0]) + (p1[0] - p3[0]) * (point[1] - p3[1])) / denom
    gamma = 1 - alpha - beta
    
    # 检查重心坐标是否都非负
    return alpha >= 0 and beta >= 0 and gamma >= 0


def compute_barycentric_coordinates(point: np.ndarray, triangle: List[np.ndarray]) -> Tuple[float, float, float]:
    """
    计算点在三角形中的重心坐标
    
    Args:
        point: 目标点 (2D)
        triangle: 三角形的三个顶点 (2D)
    
    Returns:
        重心坐标 (alpha, beta, gamma)
    """
    if len(triangle) != 3:
        return (0.0, 0.0, 0.0)
    
    p1, p2, p3 = triangle
    
    # 计算面积
    area_total = triangle_area_2d(p1, p2, p3)
    
    if area_total == 0:
        return (1.0, 0.0, 0.0)
    
    # 计算各个子三角形的面积
    area1 = triangle_area_2d(point, p2, p3)  # 对应顶点1的权重
    area2 = triangle_area_2d(p1, point, p3)  # 对应顶点2的权重
    area3 = triangle_area_2d(p1, p2, point)  # 对应顶点3的权重
    
    # 归一化
    alpha = area1 / area_total
    beta = area2 / area_total
    gamma = area3 / area_total
    
    return (alpha, beta, gamma)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def compute_vertex_normal(center: np.ndarray, ring_vertices: List[np.ndarray]) -> np.ndarray:
    """
    计算顶点的法向量
    使用相邻三角形法向量的加权平均
    
    Args:
        center: 中心顶点坐标
        ring_vertices: 环形邻域顶点坐标列表
    
    Returns:
        归一化的法向量
    """
    if len(ring_vertices) < 3:
        return np.array([0.0, 0.0, 1.0])
    
    normal = np.zeros(3)
    n = len(ring_vertices)
    
    for i in range(n):
        v1 = ring_vertices[i] - center
        v2 = ring_vertices[(i + 1) % n] - center
        
        # 计算三角形法向量
        tri_normal = np.cross(v1, v2)
        normal += tri_normal
    
    return normalize_vector(normal)


def project_to_plane(points: List[np.ndarray], normal: np.ndarray, center: np.ndarray) -> List[np.ndarray]:
    """
    将3D点投影到指定平面上
    
    Args:
        points: 要投影的3D点列表
        normal: 平面法向量
        center: 平面上的一个点
    
    Returns:
        投影后的3D点列表
    """
    normal = normalize_vector(normal)
    projected_points = []
    
    for point in points:
        # 计算点到平面的距离
        to_point = point - center
        distance = np.dot(to_point, normal)
        
        # 投影到平面
        projected = point - distance * normal
        projected_points.append(projected)
    
    return projected_points


def find_closest_point_on_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    找到点到线段的最近点
    
    Args:
        point: 目标点
        line_start, line_end: 线段的两个端点
    
    Returns:
        最近点坐标和距离
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return line_start, np.linalg.norm(point_vec)
    
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    
    # 限制投影点在线段范围内
    proj_length = np.clip(proj_length, 0.0, line_len)
    
    closest_point = line_start + proj_length * line_unitvec
    distance = np.linalg.norm(point - closest_point)
    
    return closest_point, distance