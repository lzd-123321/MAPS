"""
共形映射算法模块

实现MAPS算法的核心共形映射功能：
- 将3D顶点环形邻域映射到2D平面
- 使用复数 z^a 映射保持角度关系
- 支持边界和内部顶点的不同处理策略
- 在映射的2D环中进行点插值

共形映射是MAPS算法保持网格几何特性的关键技术。
"""

import numpy as np
import math
from typing import List, Dict, Tuple

# 修复导入问题
try:
    from .geometry_utils import (
        vector_angle, point_distance, compute_ring_angles, 
        normalize_vector, project_to_plane, compute_vertex_normal
    )
except ImportError:
    from geometry_utils import (
        vector_angle, point_distance, compute_ring_angles, 
        normalize_vector, project_to_plane, compute_vertex_normal
    )


def compute_conformal_mapping(center: np.ndarray, 
                            ring_vertices: List[np.ndarray], 
                            is_boundary: bool = False) -> Tuple[Dict[int, np.ndarray], float]:
    """
    计算顶点邻域的共形映射（完全模拟Unity实现）
    使用 z^a 映射将3D环形邻域展平到2D平面
    
    Args:
        center: 中心顶点坐标
        ring_vertices: 环形邻域顶点坐标列表 
        is_boundary: 是否在边界上
    
    Returns:
        mapped_ring: 映射后的2D坐标字典 {vertex_index: 2D_coord}
        exponent_a: 映射指数参数
    """
    if len(ring_vertices) < 3:
        return {}, 0.0
    
    # 计算角度
    angles = []
    n = len(ring_vertices)
    
    for i in range(n):
        prev_idx = (i - 1) % n
        curr_idx = i
        
        v_prev = ring_vertices[prev_idx] - center
        v_curr = ring_vertices[curr_idx] - center
        
        angle_degrees = vector_angle_degrees(v_prev, v_curr)
        angle_radians = angle_degrees * math.pi / 180.0
        angles.append(angle_radians)
    
    total_angle = sum(angles)
    
    # 计算映射指数 a
    if is_boundary:
        total_angle_for_a = total_angle - angles[-1]
        exponent_a = math.pi / total_angle_for_a if total_angle_for_a > 0 else 1.0
    else:
        exponent_a = (2 * math.pi) / total_angle if total_angle > 0 else 1.0
    
    # 构建映射后的环形邻域
    mapped_ring = {}
    cumulative_angle = 0.0
    
    for i, vertex in enumerate(ring_vertices):
        distance = point_distance(center, vertex)
        mapped_radius = math.pow(distance, exponent_a)
        
        if is_boundary:
            mapped_angle = cumulative_angle * exponent_a
            cumulative_angle += angles[i]
        else:
            cumulative_angle += angles[i]
            mapped_angle = cumulative_angle * exponent_a
        
        # 转换为2D笛卡尔坐标，缩放100倍
        scale = 100.0
        x = mapped_radius * scale * math.cos(mapped_angle)
        y = mapped_radius * scale * math.sin(mapped_angle)
        
        mapped_ring[i] = np.array([x, y])
    
    return mapped_ring, exponent_a


def vector_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个3D向量之间的角度（度数），完全模拟Unity的Vector3.Angle
    
    Args:
        v1, v2: 3D向量
    
    Returns:
        角度（度数）
    """
    # 计算向量的模长
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0
    
    # 计算余弦值
    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
    
    # 限制在有效范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 计算角度（弧度转度数）
    angle_radians = math.acos(cos_angle)
    angle_degrees = angle_radians * 180.0 / math.pi
    
    return angle_degrees


def compute_ring_mapping_simple(center: np.ndarray, 
                               ring_vertices: List[np.ndarray],
                               vertex_indices: List[int],
                               is_boundary: bool = False) -> Dict[int, np.ndarray]:
    """
    简化版本的环形邻域映射
    直接将3D环投影到2D平面
    
    Args:
        center: 中心顶点坐标
        ring_vertices: 环形邻域顶点坐标列表
        vertex_indices: 顶点索引列表
        is_boundary: 是否在边界上
    
    Returns:
        映射后的2D坐标字典
    """
    if not ring_vertices or len(ring_vertices) != len(vertex_indices):
        return {}
    
    # 计算法向量
    normal = compute_vertex_normal(center, ring_vertices)
    
    # 将环形邻域投影到与法向量垂直的平面
    projected_vertices = project_to_plane(ring_vertices, normal, center)
    
    # 构建局部坐标系
    if len(projected_vertices) == 0:
        return {}
    
    # 第一个投影顶点作为x轴方向
    x_axis = normalize_vector(projected_vertices[0] - center)
    y_axis = normalize_vector(np.cross(normal, x_axis))
    
    # 转换为2D坐标
    mapped_ring = {}
    for i, (vertex, index) in enumerate(zip(projected_vertices, vertex_indices)):
        relative_pos = vertex - center
        
        # 投影到局部2D坐标系
        x = np.dot(relative_pos, x_axis)
        y = np.dot(relative_pos, y_axis)
        
        mapped_ring[index] = np.array([x, y])
    
    return mapped_ring


def interpolate_point_in_mapped_ring(target_point_2d: np.ndarray,
                                   mapped_ring: Dict[int, np.ndarray],
                                   original_bijection: Dict[int, float]) -> np.ndarray:
    """
    在映射后的环形邻域中插值目标点
    
    Args:
        target_point_2d: 目标点的2D坐标
        mapped_ring: 映射后的环形邻域
        original_bijection: 原始双射映射
    
    Returns:
        插值后的2D坐标
    """
    if len(original_bijection) == 1:
        # 单点映射，直接返回原点
        return np.array([0.0, 0.0])
    
    elif len(original_bijection) == 2:
        # 两点映射，线性插值
        points = []
        weights = []
        for index, weight in original_bijection.items():
            if index in mapped_ring:
                points.append(mapped_ring[index])
                weights.append(weight)
        
        if len(points) == 2:
            return points[0] * weights[0] + points[1] * weights[1]
    
    elif len(original_bijection) == 3:
        # 三点映射，寻找包含目标点的三角形
        indices = list(original_bijection.keys())
        if all(idx in mapped_ring for idx in indices):
            # 构建三角形并进行重心坐标插值
            triangle_2d = [mapped_ring[idx] for idx in indices]
            weights = [original_bijection[idx] for idx in indices]
            
            # 加权平均
            result = np.zeros(2)
            for point, weight in zip(triangle_2d, weights):
                result += point * weight
            return result
    
    # 默认情况：加权平均
    result = np.zeros(2)
    total_weight = 0.0
    
    for index, weight in original_bijection.items():
        if index in mapped_ring:
            result += mapped_ring[index] * weight
            total_weight += weight
    
    if total_weight > 0:
        result /= total_weight
    
    return result


def compute_center_offset(mapped_ring: Dict[int, np.ndarray]) -> np.ndarray:
    """
    计算映射环的中心偏移量
    用于将映射后的环形邻域居中
    
    Args:
        mapped_ring: 映射后的2D坐标字典
    
    Returns:
        中心点坐标
    """
    if not mapped_ring:
        return np.array([0.0, 0.0])
    
    points = list(mapped_ring.values())
    center = np.mean(points, axis=0)
    return center


def refine_mapping_to_center(target_point: np.ndarray, 
                           center_offset: np.ndarray,
                           max_iterations: int = 10) -> np.ndarray:
    """
    将目标点逐步调整到环的中心
    
    Args:
        target_point: 目标点2D坐标
        center_offset: 环的中心偏移
        max_iterations: 最大迭代次数
    
    Returns:
        调整后的点坐标
    """
    current_point = target_point.copy()
    
    for _ in range(max_iterations):
        # 向中心移动10%的距离
        direction = center_offset - current_point
        current_point += 0.1 * direction
        
        # 如果足够接近中心，停止迭代
        if np.linalg.norm(direction) < 1e-6:
            break
    
    return current_point


def validate_conformal_mapping(original_ring: List[np.ndarray],
                             mapped_ring: Dict[int, np.ndarray],
                             center: np.ndarray) -> Dict[str, float]:
    """
    验证共形映射的质量
    
    Args:
        original_ring: 原始3D环形邻域
        mapped_ring: 映射后的2D环形邻域
        center: 中心点坐标
    
    Returns:
        映射质量指标字典
    """
    if len(original_ring) != len(mapped_ring):
        return {'valid': False, 'angle_preservation': 0.0, 'area_ratio': 0.0}
    
    # 计算角度保持性
    original_angles = compute_ring_angles(center, original_ring)
    
    # 计算映射后的角度（围绕原点）
    mapped_points = list(mapped_ring.values())
    if len(mapped_points) < 3:
        return {'valid': False, 'angle_preservation': 0.0, 'area_ratio': 0.0}
    
    mapped_angles = []
    n = len(mapped_points)
    origin = np.array([0.0, 0.0])
    
    for i in range(n):
        v1 = mapped_points[i - 1] - origin
        v2 = mapped_points[i] - origin
        
        # 计算2D向量夹角
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        if angle < 0:
            angle += 2 * math.pi
        mapped_angles.append(angle)
    
    # 计算角度保持性（相关系数）
    if len(original_angles) == len(mapped_angles):
        angle_correlation = np.corrcoef(original_angles, mapped_angles)[0, 1]
        if np.isnan(angle_correlation):
            angle_correlation = 0.0
    else:
        angle_correlation = 0.0
    
    return {
        'valid': True,
        'angle_preservation': abs(angle_correlation),
        'num_vertices': len(mapped_ring),
        'original_angle_sum': sum(original_angles),
        'mapped_angle_sum': sum(mapped_angles)
    }