"""
MAPS主算法类

MAPS (Multi-resolution Adaptive Parameterization of Surfaces) 算法的核心实现。
整合所有子模块，提供完整的网格简化流程：

主要功能：
- 管理网格简化的整个流程
- 协调共形映射、三角化和简化模块
- 提供简化统计和进度跟踪
- 支持迭代式网格简化

核心方法：
- level_down(): 执行一次简化迭代
- get_current_mesh(): 获取当前网格状态
- get_statistics(): 获取简化统计信息
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# 修复导入问题
try:
    from .maps_mesh import MapsMesh
    from .data_structures import Triangle
    from .simplification import (
        compute_vertex_priorities, extract_vertex_stars, extract_independent_set,
        validate_vertex_removal, find_ring_from_star
    )
    from .conformal_mapping import (
        compute_conformal_mapping, interpolate_point_in_mapped_ring, 
        compute_center_offset, refine_mapping_to_center
    )
    from .triangulation import triangulate_from_mapped_ring
    from .geometry_utils import compute_barycentric_coordinates, point_in_triangle_2d
except ImportError:
    from maps_mesh import MapsMesh
    from data_structures import Triangle
    from simplification import (
        compute_vertex_priorities, extract_vertex_stars, extract_independent_set,
        validate_vertex_removal, find_ring_from_star
    )
    from conformal_mapping import (
        compute_conformal_mapping, interpolate_point_in_mapped_ring, 
        compute_center_offset, refine_mapping_to_center
    )
    from triangulation import triangulate_from_mapped_ring
    from geometry_utils import compute_barycentric_coordinates, point_in_triangle_2d


class MAPS:
    """MAPS算法主类"""
    
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, lambda_weight: float = 0.8):
        """
        初始化MAPS算法
        
        Args:
            vertices: 顶点坐标数组 (N, 3)
            triangles: 三角形索引数组 (M, 3)
            lambda_weight: 权重参数，控制面积和曲率的平衡
        """
        self.mesh = MapsMesh(vertices, triangles)
        self.removed_vertices = []
        self.unremovable_vertices = []
        self.lambda_weight = lambda_weight  # 保存权重参数
        
        print(f"MAPS初始化完成:")
        print(f"  顶点数: {len(vertices)}")
        print(f"  三角形数: {len(triangles)}")
        print(f"  权重参数: {lambda_weight} (面积权重: {lambda_weight:.1%}, 曲率权重: {(1-lambda_weight):.1%})")
    
    def level_down(self) -> bool:
        """
        执行一次网格简化步骤
        
        Returns:
            True如果成功移除了顶点
        """
        # 获取候选顶点
        candidates = self.mesh.get_candidate_vertices(
            self.removed_vertices, self.unremovable_vertices
        )
        
        if not candidates:
            return False
        
        # 计算优先级和星形邻域
        priorities = compute_vertex_priorities(self.mesh, candidates, self.lambda_weight)
        stars = extract_vertex_stars(self.mesh, candidates)
        
        if not priorities:
            return False
        
        # 提取独立集
        independent_set = extract_independent_set(priorities, stars)
        
        if not independent_set:
            return False
        
        # 尝试移除独立集中的每个顶点
        removed_count = 0
        
        for vertex_index in independent_set:
            success = self._remove_single_vertex(vertex_index)
            if success:
                removed_count += 1
                self.removed_vertices.append(vertex_index)
            else:
                self.unremovable_vertices.append(vertex_index)
        
        return removed_count > 0
    
    def _remove_single_vertex(self, vertex_index: int) -> bool:
        """
        移除单个顶点的完整流程
        
        Args:
            vertex_index: 要移除的顶点索引
        
        Returns:
            True如果成功移除
        """
        # 验证顶点是否可以移除
        is_valid, reason = validate_vertex_removal(self.mesh, vertex_index)
        if not is_valid:
            return False
        
        # 获取星形邻域和环形邻域
        star = self.mesh.get_vertex_star(vertex_index)
        ring, is_boundary = self.mesh.get_vertex_ring(vertex_index)
        
        if len(ring) < 3 or len(ring) >= 12:
            return False
        
        # 执行共形映射
        ring_coordinates = [self.mesh.P[i] for i in ring]
        center_coordinate = self.mesh.P[vertex_index]
        
        mapped_ring_temp, exponent_a = compute_conformal_mapping(
            center_coordinate, ring_coordinates, is_boundary
        )
        
        if not mapped_ring_temp:
            return False
        
        # 将临时索引映射到真实的顶点索引
        mapped_ring = {}
        for i, vertex_idx in enumerate(ring):
            if i in mapped_ring_temp:
                mapped_ring[vertex_idx] = mapped_ring_temp[i]
        
        # 执行三角化
        triangles, triangulation_success = triangulate_from_mapped_ring(
            ring, mapped_ring, method="ear_cutting"
        )
        
        if not triangulation_success or not triangles:
            return False
        
        # 更新双射映射
        self._update_bijection_mappings(vertex_index, ring, mapped_ring, triangles)
        
        # 从网格中移除星形邻域和顶点
        self.mesh.remove_triangles(star)
        self.mesh.remove_vertex(vertex_index)
        
        # 添加新的三角形
        self.mesh.add_triangles(triangles)
        
        return True
    
    def _update_bijection_mappings(self, removed_vertex: int, ring: List[int],
                                 mapped_ring: Dict[int, np.ndarray],
                                 new_triangles: List[Triangle]):
        """
        更新受影响顶点的双射映射
        
        Args:
            removed_vertex: 被移除的顶点索引
            ring: 环形邻域
            mapped_ring: 映射后的环形邻域
            new_triangles: 新生成的三角形
        """
        # 遍历所有双射映射，更新包含被移除顶点的映射
        for i, bijection_map in enumerate(self.mesh.bijection):
            if removed_vertex not in bijection_map:
                continue
            
            # 计算目标点在映射环中的位置
            target_point_2d = interpolate_point_in_mapped_ring(
                np.array([0.0, 0.0]), mapped_ring, bijection_map
            )
            
            # 在新三角形中查找包含目标点的三角形
            new_mapping = self._find_containing_triangle_mapping(
                target_point_2d, mapped_ring, new_triangles
            )
            
            if new_mapping:
                self.mesh.update_bijection(i, new_mapping)
            else:
                # 如果找不到包含的三角形，使用加权平均
                self._create_fallback_mapping(i, bijection_map, ring, removed_vertex)
    
    def _find_containing_triangle_mapping(self, target_point: np.ndarray,
                                        mapped_ring: Dict[int, np.ndarray],
                                        triangles: List[Triangle]) -> Optional[Dict[int, float]]:
        """
        找到包含目标点的三角形并计算重心坐标
        
        Args:
            target_point: 目标点2D坐标
            mapped_ring: 映射后的环形邻域
            triangles: 候选三角形列表
        
        Returns:
            新的双射映射字典
        """
        center_offset = compute_center_offset(mapped_ring)
        current_point = target_point.copy()
        
        # 尝试多次，逐步向中心调整
        for iteration in range(10):
            for triangle in triangles:
                vertices = triangle.get_vertices()
                
                # 检查三角形的所有顶点是否都在映射环中
                if not all(v in mapped_ring for v in vertices):
                    continue
                
                # 获取三角形的2D坐标
                triangle_2d = [mapped_ring[v] for v in vertices]
                
                # 检查点是否在三角形内
                if point_in_triangle_2d(current_point, triangle_2d):
                    # 计算重心坐标
                    alpha, beta, gamma = compute_barycentric_coordinates(
                        current_point, triangle_2d
                    )
                    
                    # 创建新的双射映射
                    new_mapping = {}
                    weights = [alpha, beta, gamma]
                    
                    for i, (vertex, weight) in enumerate(zip(vertices, weights)):
                        if weight > 1e-6:  # 忽略很小的权重
                            new_mapping[vertex] = weight
                    
                    # 归一化权重
                    total_weight = sum(new_mapping.values())
                    if total_weight > 0:
                        for vertex in new_mapping:
                            new_mapping[vertex] /= total_weight
                        return new_mapping
            
            # 向中心移动
            current_point = refine_mapping_to_center(current_point, center_offset, 1)
        
        return None
    
    def _create_fallback_mapping(self, vertex_index: int, 
                               original_mapping: Dict[int, float],
                               ring: List[int], removed_vertex: int):
        """
        创建备用的双射映射（当无法找到包含三角形时）
        
        Args:
            vertex_index: 要更新映射的顶点索引
            original_mapping: 原始双射映射
            ring: 环形邻域
            removed_vertex: 被移除的顶点
        """
        # 移除被删除顶点，重新归一化权重
        new_mapping = {}
        total_weight = 0.0
        
        for vertex, weight in original_mapping.items():
            if vertex != removed_vertex and vertex in ring:
                new_mapping[vertex] = weight
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for vertex in new_mapping:
                new_mapping[vertex] /= total_weight
        else:
            # 极端情况：平均分配给环中的顶点
            if ring:
                weight_per_vertex = 1.0 / len(ring)
                new_mapping = {v: weight_per_vertex for v in ring}
        
        self.mesh.update_bijection(vertex_index, new_mapping)
    
    def get_current_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前简化后的网格
        
        Returns:
            (vertices, triangles) 当前网格的顶点和三角形
        """
        return self.mesh.export_mesh()
    
    def get_statistics(self) -> Dict:
        """获取算法统计信息"""
        current_info = self.mesh.get_mesh_info()
        
        return {
            'original_vertices': len(self.mesh.P),
            'current_vertices': current_info['vertices'],
            'current_triangles': current_info['triangles'],
            'removed_vertices': len(self.removed_vertices),
            'unremovable_vertices': len(self.unremovable_vertices),
            'simplification_ratio': len(self.removed_vertices) / len(self.mesh.P)
        }
    
    def reset_to_original(self):
        """重置到原始网格状态"""
        # 重新初始化双射映射
        self.mesh.bijection = self.mesh._initialize_bijection()
        
        # 重建原始拓扑
        triangles = []
        for i in range(0, len(self.mesh.P) - 2, 3):
            if i + 2 < len(self.mesh.P):
                triangles.append([i, i + 1, i + 2])
        
        if triangles:
            triangles = np.array(triangles)
            self.mesh.topology = self.mesh._build_topology(triangles)
        
        # 清空移除记录
        self.removed_vertices = []
        self.unremovable_vertices = []
        
        print("已重置到原始网格状态")