"""
网格简化算法模块

实现MAPS算法的网格简化核心功能：
- 基于几何重要性的顶点优先级计算
- 独立集提取确保顶点移除的安全性
- 顶点星形邻域和环形邻域的提取
- 顶点移除操作的验证和执行

结合面积和曲率信息，选择最适合移除的顶点集合。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# 修复导入问题
try:
    from .maps_mesh import MapsMesh
    from .data_structures import Triangle
    from .geometry_utils import triangle_area_3d, compute_vertex_curvature
except ImportError:
    from maps_mesh import MapsMesh
    from data_structures import Triangle
    from geometry_utils import triangle_area_3d, compute_vertex_curvature


def compute_vertex_priorities(mesh: MapsMesh, candidates: List[int], 
                            lambda_weight: float = 0.5) -> Dict[int, float]:
    """
    计算候选顶点的移除优先级
    优先级 = λ * (面积/最大面积) + (1-λ) * (曲率/最大曲率)
    
    Args:
        mesh: MAPS网格对象
        candidates: 候选顶点列表
        lambda_weight: 面积权重参数
    
    Returns:
        优先级字典 {vertex_index: priority}
    """
    priorities = {}
    areas = []
    curvatures = []
    
    # 计算每个候选顶点的面积和曲率
    for vertex_index in candidates:
        # 获取顶点的星形邻域
        star = mesh.get_vertex_star(vertex_index)
        
        if not star:
            continue
        
        # 计算顶点周围的总面积
        total_area = 0.0
        vertex_coords = []
        
        for triangle in star:
            # 获取三角形的三个顶点坐标
            ordered_vertices = triangle.get_ordered_vertices(vertex_index)
            if len(ordered_vertices) >= 3:
                p0 = mesh.P[ordered_vertices[0]]  # 中心顶点
                p1 = mesh.P[ordered_vertices[1]]
                p2 = mesh.P[ordered_vertices[2]]
                
                area = triangle_area_3d(p0, p1, p2)
                total_area += area
        
        areas.append(total_area)
        
        # 计算顶点的近似高斯曲率
        ring, _ = mesh.get_vertex_ring(vertex_index)
        if ring:
            ring_coords = [mesh.P[i] for i in ring]
            center_coord = mesh.P[vertex_index]
            curvature = compute_vertex_curvature(center_coord, ring_coords)
        else:
            curvature = 0.0
        
        curvatures.append(abs(curvature))  # 使用绝对值
    
    # 找到最大值用于归一化
    max_area = max(areas) if areas else 1.0
    max_curvature = max(curvatures) if curvatures else 1.0
    
    # 防止除零
    max_area = max(max_area, 1e-8)
    max_curvature = max(max_curvature, 1e-8)
    
    # 计算归一化的优先级
    for i, vertex_index in enumerate(candidates):
        if i < len(areas) and i < len(curvatures):
            area_term = areas[i] / max_area
            curvature_term = curvatures[i] / max_curvature
            
            priority = lambda_weight * area_term + (1.0 - lambda_weight) * curvature_term
            priorities[vertex_index] = priority
    
    return priorities


def extract_vertex_stars(mesh: MapsMesh, candidates: List[int]) -> Dict[int, List[Triangle]]:
    """
    提取候选顶点的星形邻域
    
    Args:
        mesh: MAPS网格对象
        candidates: 候选顶点列表
    
    Returns:
        星形邻域字典 {vertex_index: star_triangles}
    """
    stars = {}
    
    for vertex_index in candidates:
        star = mesh.get_vertex_star(vertex_index)
        stars[vertex_index] = star
    
    return stars


def extract_independent_set(priorities: Dict[int, float], 
                          stars: Dict[int, List[Triangle]]) -> List[int]:
    """
    提取独立集（不相邻的顶点集合）
    按优先级从低到高选择顶点，确保选中的顶点互不相邻
    
    Args:
        priorities: 顶点优先级字典
        stars: 顶点星形邻域字典
    
    Returns:
        独立集顶点列表
    """
    # 按优先级升序排序（优先级低的先移除）
    sorted_vertices = sorted(priorities.items(), key=lambda x: x[1])
    
    independent_set = []
    excluded_vertices = set()
    
    for vertex_index, priority in sorted_vertices:
        if vertex_index in excluded_vertices:
            continue
        
        # 将当前顶点加入独立集
        independent_set.append(vertex_index)
        
        # 排除当前顶点星形邻域中的所有顶点
        star = stars.get(vertex_index, [])
        for triangle in star:
            excluded_vertices.add(triangle.v1)
            excluded_vertices.add(triangle.v2)
            excluded_vertices.add(triangle.v3)
    
    return independent_set


def validate_vertex_removal(mesh: MapsMesh, vertex_index: int) -> Tuple[bool, str]:
    """
    验证顶点是否可以安全移除
    
    Args:
        mesh: MAPS网格对象
        vertex_index: 要验证的顶点索引
    
    Returns:
        (is_valid, reason) 验证结果和原因
    """
    # 获取星形邻域
    star = mesh.get_vertex_star(vertex_index)
    
    # 检查星形邻域大小
    if len(star) < 3:
        return False, "星形邻域太小（少于3个三角形）"
    
    if len(star) > 12:
        return False, "星形邻域太大（超过12个三角形）"
    
    # 获取环形邻域
    ring, is_boundary = mesh.get_vertex_ring(vertex_index)
    
    # 检查环形邻域有效性
    if len(ring) < 3:
        return False, "环形邻域无效（少于3个顶点）"
    
    # 检查是否有重复顶点
    if len(set(ring)) != len(ring):
        return False, "环形邻域包含重复顶点"
    
    # 检查环形邻域大小
    if len(ring) >= 12:
        return False, "环形邻域太大（超过12个顶点）"
    
    return True, "可以安全移除"


def find_ring_from_star(star: List[Triangle], center_vertex: int) -> Tuple[List[int], bool, bool]:
    """
    从星形邻域构建环形邻域
    
    Args:
        star: 星形邻域三角形列表
        center_vertex: 中心顶点索引
    
    Returns:
        (ring, is_boundary, is_invalid) 环形邻域、是否边界、是否无效
    """
    if not star:
        return [], False, True
    
    # 检查重复的三角形或坏的拓扑
    edge_count = {}
    for triangle in star:
        vertices = triangle.get_vertices()
        if center_vertex not in vertices:
            continue
        
        # 获取不包含中心顶点的边
        other_vertices = [v for v in vertices if v != center_vertex]
        if len(other_vertices) == 2:
            edge = tuple(sorted(other_vertices))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    # 如果有边出现超过2次，说明拓扑有问题
    bad_edges = [edge for edge, count in edge_count.items() if count > 2]
    if bad_edges:
        return [], False, True
    
    # 构建环形邻域
    ring = []
    used_triangles = set()
    
    # 从第一个三角形开始
    current_triangle = star[0]
    used_triangles.add(id(current_triangle))
    
    # 获取除中心顶点外的两个顶点
    vertices = current_triangle.get_ordered_vertices(center_vertex)
    if len(vertices) >= 3:
        ring.append(vertices[1])
        ring.append(vertices[2])
    else:
        return [], False, True
    
    # 继续查找相邻的三角形
    while len(used_triangles) < len(star):
        last_vertex = ring[-1]
        found = False
        
        for triangle in star:
            if id(triangle) in used_triangles:
                continue
            
            if triangle.contains_edge(center_vertex, last_vertex):
                used_triangles.add(id(triangle))
                vertices = triangle.get_ordered_vertices(center_vertex)
                
                if len(vertices) >= 3:
                    # 找到下一个顶点
                    if vertices[1] == last_vertex:
                        ring.append(vertices[2])
                    elif vertices[2] == last_vertex:
                        ring.append(vertices[1])
                    else:
                        # 拓扑错误
                        return [], False, True
                
                found = True
                break
        
        if not found:
            # 可能是边界情况
            break
    
    # 检查是否在边界
    is_boundary = len(ring) < 3 or (ring and ring[0] != ring[-1])
    
    # 如果不在边界，移除最后一个重复顶点
    if not is_boundary and ring and ring[0] == ring[-1]:
        ring.pop()
    
    # 检查环的有效性
    is_invalid = len(set(ring)) != len(ring) or len(ring) < 3
    
    return ring, is_boundary, is_invalid


def perform_vertex_removal(mesh: MapsMesh, vertex_index: int) -> Tuple[bool, str]:
    """
    执行顶点移除操作
    
    Args:
        mesh: MAPS网格对象
        vertex_index: 要移除的顶点索引
    
    Returns:
        (success, message) 操作结果和消息
    """
    # 验证是否可以移除
    is_valid, reason = validate_vertex_removal(mesh, vertex_index)
    if not is_valid:
        return False, f"无法移除顶点 {vertex_index}: {reason}"
    
    # 获取星形邻域
    star = mesh.get_vertex_star(vertex_index)
    
    # 从网格中移除星形邻域的所有三角形
    mesh.remove_triangles(star)
    
    # 从拓扑结构中移除顶点
    mesh.remove_vertex(vertex_index)
    
    return True, f"成功移除顶点 {vertex_index}"


def compute_simplification_statistics(mesh: MapsMesh, 
                                    removed_vertices: List[int]) -> Dict:
    """
    计算简化统计信息
    
    Args:
        mesh: MAPS网格对象
        removed_vertices: 已移除的顶点列表
    
    Returns:
        统计信息字典
    """
    current_info = mesh.get_mesh_info()
    
    return {
        'current_vertices': current_info['vertices'],
        'current_triangles': current_info['triangles'],
        'removed_vertices': len(removed_vertices),
        'feature_points': current_info['feature_points'],
        'removal_rate': len(removed_vertices) / (current_info['vertices'] + len(removed_vertices))
        if (current_info['vertices'] + len(removed_vertices)) > 0 else 0.0
    }