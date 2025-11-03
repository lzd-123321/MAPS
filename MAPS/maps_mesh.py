"""
MAPS网格管理类

MapsMesh类负责管理3D网格的所有数据，包括：
- 顶点位置数组 (P)
- 拓扑连接关系 (topology)
- 双射映射维护 (bijection)
- 网格验证和统计

这是MAPS算法的核心数据结构，提供网格操作的基础功能。
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional

# 修复导入问题
try:
    from .data_structures import Vertex, Triangle, Edge, Topology
except ImportError:
    from data_structures import Vertex, Triangle, Edge, Topology


class MapsMesh:
    """MAPS网格类，实现多分辨率自适应参数化"""
    
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        初始化MAPS网格
        
        Args:
            vertices: 顶点坐标数组 (N, 3)
            triangles: 三角形索引数组 (M, 3) 
        """
        self.P = vertices.copy()  # 原始顶点位置
        self.topology = self._build_topology(triangles)
        self.bijection = self._initialize_bijection()
    
    def _build_topology(self, triangles: np.ndarray) -> Topology:
        """从三角形数组构建拓扑结构"""
        vertices = [Vertex(i) for i in range(len(self.P))]
        triangle_list = []
        
        for tri in triangles:
            triangle_list.append(Triangle(tri[0], tri[1], tri[2]))
        
        return Topology(vertices=vertices, triangles=triangle_list)
    
    def _initialize_bijection(self) -> List[Dict[int, float]]:
        """
        初始化双射映射
        每个顶点初始时映射到自己，权重为1.0
        """
        bijection = []
        for i in range(len(self.P)):
            bijection.append({i: 1.0})
        return bijection
    
    def get_candidate_vertices(self, removed_indices: List[int], 
                             unremovable_indices: List[int]) -> List[int]:
        """
        获取可以移除的候选顶点
        排除已移除顶点和不可移除顶点
        """
        candidates = []
        
        for vertex in self.topology.vertices:
            index = vertex.index
            if (index not in removed_indices and 
                index not in unremovable_indices):
                candidates.append(index)
        
        return candidates
    
    def get_projected_points(self, indices: List[int]) -> np.ndarray:
        """
        获取顶点的投影位置
        使用双射映射计算当前顶点的实际位置
        """
        projected_points = []
        
        for index in indices:
            point = np.zeros(3)
            bijection_map = self.bijection[index]
            
            for original_index, weight in bijection_map.items():
                point += self.P[original_index] * weight
            
            projected_points.append(point)
        
        return np.array(projected_points)
    
    def remove_vertex(self, vertex_index: int):
        """从拓扑结构中移除顶点"""
        self.topology.remove_vertex(vertex_index)
    
    def remove_triangles(self, triangles: List[Triangle]):
        """从拓扑结构中移除三角形列表"""
        for triangle in triangles:
            self.topology.remove_triangle(triangle)
    
    def add_triangles(self, triangles: List[Triangle]):
        """向拓扑结构中添加三角形列表"""
        for triangle in triangles:
            self.topology.add_triangle(triangle)
    
    def get_vertex_star(self, vertex_index: int) -> List[Triangle]:
        """获取顶点的星形邻域"""
        return self.topology.get_vertex_star(vertex_index)
    
    def get_vertex_ring(self, vertex_index: int) -> Tuple[List[int], bool]:
        """
        获取顶点的环形邻域
        
        Returns:
            ring: 环形邻域顶点列表
            is_boundary: 是否在边界上
        """
        star = self.get_vertex_star(vertex_index)
        if not star:
            return [], False
        
        # 构建环形邻域
        ring = []
        used_triangles = set()
        
        # 从第一个三角形开始
        current_triangle = star[0]
        used_triangles.add(id(current_triangle))
        
        # 获取除中心顶点外的两个顶点
        vertices = current_triangle.get_ordered_vertices(vertex_index)
        if len(vertices) >= 3:
            ring.append(vertices[1])
            ring.append(vertices[2])
        
        # 继续查找相邻三角形
        while len(used_triangles) < len(star):
            last_vertex = ring[-1]
            found = False
            
            for triangle in star:
                if id(triangle) in used_triangles:
                    continue
                
                if triangle.contains_edge(vertex_index, last_vertex):
                    used_triangles.add(id(triangle))
                    vertices = triangle.get_ordered_vertices(vertex_index)
                    
                    # 找到下一个顶点
                    if len(vertices) >= 3:
                        if vertices[1] == last_vertex:
                            ring.append(vertices[2])
                        elif vertices[2] == last_vertex:
                            ring.append(vertices[1])
                    
                    found = True
                    break
            
            if not found:
                break
        
        # 检查是否在边界
        is_boundary = len(ring) < 3 or (ring and ring[0] != ring[-1])
        
        # 如果不在边界，移除最后一个重复顶点
        if not is_boundary and ring and ring[0] == ring[-1]:
            ring.pop()
        
        return ring, is_boundary
    
    def update_bijection(self, vertex_index: int, new_mapping: Dict[int, float]):
        """更新顶点的双射映射"""
        if vertex_index < len(self.bijection):
            self.bijection[vertex_index] = new_mapping.copy()
    
    def is_valid_for_removal(self, vertex_index: int) -> bool:
        """检查顶点是否可以安全移除"""
        star = self.get_vertex_star(vertex_index)
        
        # 检查星形邻域大小
        if len(star) < 3 or len(star) > 12:
            return False
        
        # 检查环形邻域是否有效
        ring, is_boundary = self.get_vertex_ring(vertex_index)
        if len(set(ring)) != len(ring):  # 检查是否有重复顶点
            return False
        
        return True
    
    def get_mesh_info(self) -> Dict:
        """获取网格信息"""
        return {
            'vertices': len(self.topology.vertices),
            'triangles': len(self.topology.triangles),
            'bounds': {
                'min': np.min(self.P, axis=0),
                'max': np.max(self.P, axis=0)
            }
        }
    
    def export_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        导出当前网格数据
        
        Returns:
            vertices: 顶点坐标数组
            triangles: 三角形索引数组
        """
        # 获取当前活跃的顶点
        active_vertices = [v.index for v in self.topology.vertices]
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_vertices)}
        
        # 重新映射顶点坐标
        vertices = self.get_projected_points(active_vertices)
        
        # 重新映射三角形索引
        triangles = []
        for triangle in self.topology.triangles:
            if (triangle.v1 in vertex_map and 
                triangle.v2 in vertex_map and 
                triangle.v3 in vertex_map):
                triangles.append([
                    vertex_map[triangle.v1],
                    vertex_map[triangle.v2],
                    vertex_map[triangle.v3]
                ])
        
        return vertices, np.array(triangles)