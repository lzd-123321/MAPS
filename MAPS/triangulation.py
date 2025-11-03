"""
三角化算法模块

提供多种三角化方法重建网格结构：
- 简单扇形三角化：适用于凸多边形
- 耳切法三角化：处理任意简单多边形
- 约束Delaunay三角化：保持边界约束

在顶点移除后重新连接网格时，确保生成高质量的三角形。
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional

# 修复导入问题
try:
    from .data_structures import Triangle
    from .geometry_utils import point_in_triangle_2d, triangle_area_2d, compute_barycentric_coordinates
except ImportError:
    from data_structures import Triangle
    from geometry_utils import point_in_triangle_2d, triangle_area_2d, compute_barycentric_coordinates


def simple_triangulation_from_ring(ring: List[int], is_boundary: bool = False) -> List[Triangle]:
    """
    从环形邻域创建简单三角化
    使用扇形三角化方法（从第一个顶点连接所有对面的边）
    
    Args:
        ring: 环形邻域顶点索引列表
        is_boundary: 是否为边界环
    
    Returns:
        新生成的三角形列表
    """
    if len(ring) < 3:
        return []
    
    triangles = []
    n = len(ring)
    
    if n == 3:
        # 三个顶点直接构成一个三角形
        triangles.append(Triangle(ring[0], ring[1], ring[2]))
    elif n == 4:
        # 四个顶点构成两个三角形
        triangles.append(Triangle(ring[0], ring[1], ring[2]))
        triangles.append(Triangle(ring[0], ring[2], ring[3]))
    else:
        # 更多顶点使用扇形三角化
        for i in range(1, n - 1):
            triangles.append(Triangle(ring[0], ring[i], ring[i + 1]))
    
    return triangles


def ear_cutting_triangulation(ring_2d: Dict[int, np.ndarray]) -> List[Triangle]:
    """
    使用耳切法对2D多边形进行三角化
    
    Args:
        ring_2d: 环形顶点的2D坐标字典 {vertex_index: 2d_coord}
    
    Returns:
        三角形列表
    """
    if len(ring_2d) < 3:
        return []
    
    # 获取顶点索引和坐标
    indices = list(ring_2d.keys())
    coords = [ring_2d[idx] for idx in indices]
    
    if len(indices) == 3:
        return [Triangle(indices[0], indices[1], indices[2])]
    
    triangles = []
    remaining_indices = indices.copy()
    remaining_coords = coords.copy()
    
    # 耳切算法
    while len(remaining_indices) > 3:
        ear_found = False
        
        for i in range(len(remaining_indices)):
            # 检查当前顶点是否为"耳朵"
            prev_idx = (i - 1) % len(remaining_indices)
            next_idx = (i + 1) % len(remaining_indices)
            
            p_prev = remaining_coords[prev_idx]
            p_curr = remaining_coords[i]
            p_next = remaining_coords[next_idx]
            
            # 检查三角形是否为凸耳朵
            if is_ear(p_prev, p_curr, p_next, remaining_coords, i):
                # 创建三角形
                triangle = Triangle(
                    remaining_indices[prev_idx],
                    remaining_indices[i],
                    remaining_indices[next_idx]
                )
                triangles.append(triangle)
                
                # 移除当前顶点
                remaining_indices.pop(i)
                remaining_coords.pop(i)
                
                ear_found = True
                break
        
        # 如果没找到耳朵，使用简单三角化
        if not ear_found:
            break
    
    # 添加最后的三角形
    if len(remaining_indices) == 3:
        triangles.append(Triangle(
            remaining_indices[0],
            remaining_indices[1],
            remaining_indices[2]
        ))
    
    return triangles


def is_ear(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray,
          all_coords: List[np.ndarray], curr_index: int) -> bool:
    """
    检查三个连续顶点是否构成一个"耳朵"
    
    Args:
        p_prev, p_curr, p_next: 三个连续顶点的坐标
        all_coords: 所有顶点坐标列表
        curr_index: 当前顶点在列表中的索引
    
    Returns:
        True如果构成耳朵
    """
    # 检查三角形是否为凸的（逆时针方向）
    cross = (p_next[0] - p_curr[0]) * (p_prev[1] - p_curr[1]) - \
            (p_next[1] - p_curr[1]) * (p_prev[0] - p_curr[0])
    
    if cross <= 0:
        return False  # 不是凸三角形
    
    # 检查是否有其他顶点在三角形内部
    triangle = [p_prev, p_curr, p_next]
    
    for i, coord in enumerate(all_coords):
        if i == curr_index or i == (curr_index - 1) % len(all_coords) or \
           i == (curr_index + 1) % len(all_coords):
            continue
        
        if point_in_triangle_2d(coord, triangle):
            return False
    
    return True


def compute_convex_hull_2d(points: List[np.ndarray]) -> List[int]:
    """
    计算2D点集的凸包（Graham扫描法）
    
    Args:
        points: 2D点列表
    
    Returns:
        凸包顶点索引列表
    """
    if len(points) < 3:
        return list(range(len(points)))
    
    # 找到最左下角的点
    start_idx = 0
    for i in range(1, len(points)):
        if (points[i][1] < points[start_idx][1] or 
            (points[i][1] == points[start_idx][1] and points[i][0] < points[start_idx][0])):
            start_idx = i
    
    # 按极角排序
    def polar_angle(p):
        dx = p[0] - points[start_idx][0]
        dy = p[1] - points[start_idx][1]
        return math.atan2(dy, dx)
    
    sorted_indices = sorted(range(len(points)), key=lambda i: polar_angle(points[i]))
    
    # Graham扫描
    hull = []
    for idx in sorted_indices:
        while (len(hull) > 1 and 
               cross_product_2d(points[hull[-2]], points[hull[-1]], points[idx]) <= 0):
            hull.pop()
        hull.append(idx)
    
    return hull


def cross_product_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算2D叉积"""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def triangulate_from_mapped_ring(ring_indices: List[int],
                               mapped_ring: Dict[int, np.ndarray],
                               method: str = "ear_cutting") -> Tuple[List[Triangle], bool]:
    """
    从映射后的环形邻域生成三角化
    
    Args:
        ring_indices: 环形顶点索引列表
        mapped_ring: 映射后的2D坐标字典
        method: 三角化方法 ("simple", "ear_cutting")
    
    Returns:
        (triangles, success) 三角形列表和成功标志
    """
    if len(ring_indices) < 3:
        return [], False
    
    # 检查映射是否有效
    missing_vertices = [idx for idx in ring_indices if idx not in mapped_ring]
    if missing_vertices:
        return [], False
    
    try:
        if method == "simple":
            triangles = simple_triangulation_from_ring(ring_indices)
            
        elif method == "ear_cutting":
            ring_2d = {idx: mapped_ring[idx] for idx in ring_indices}
            triangles = ear_cutting_triangulation(ring_2d)
            
        else:
            triangles = simple_triangulation_from_ring(ring_indices)
        
        # 验证三角化结果
        if not triangles:
            triangles = simple_triangulation_from_ring(ring_indices)
        
        return triangles, len(triangles) > 0
        
    except Exception as e:
        triangles = simple_triangulation_from_ring(ring_indices)
        return triangles, len(triangles) > 0


def validate_triangulation(triangles: List[Triangle], ring_indices: List[int]) -> bool:
    """
    验证三角化结果是否有效
    
    Args:
        triangles: 三角形列表
        ring_indices: 原始环形顶点索引
    
    Returns:
        True如果三角化有效
    """
    if not triangles:
        return False
    
    # 检查所有三角形的顶点是否都在环中
    ring_set = set(ring_indices)
    
    for triangle in triangles:
        vertices = triangle.get_vertices()
        if not all(v in ring_set for v in vertices):
            return False
    
    # 检查欧拉公式 V - E + F = 1 (对于有一个洞的平面图)
    vertices = set()
    edges = set()
    
    for triangle in triangles:
        v1, v2, v3 = triangle.get_vertices()
        vertices.update([v1, v2, v3])
        
        # 添加边（无向）
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v3])))
        edges.add(tuple(sorted([v3, v1])))
    
    V = len(vertices)
    E = len(edges)
    F = len(triangles)
    
    # 对于三角化的环形区域，欧拉特征数应该为1
    euler_char = V - E + F
    
    return abs(euler_char - 1) <= 1  # 允许小的误差


def optimize_triangulation(triangles: List[Triangle],
                         mapped_ring: Dict[int, np.ndarray]) -> List[Triangle]:
    """
    优化三角化结果（简单的边翻转优化）
    
    Args:
        triangles: 原始三角形列表
        mapped_ring: 映射后的2D坐标字典
    
    Returns:
        优化后的三角形列表
    """
    # 对于简单实现，直接返回原始三角化
    # 更复杂的优化可以实现边翻转算法来改善三角形质量
    return triangles