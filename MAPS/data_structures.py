"""
MAPS算法基础数据结构
包含顶点、边、三角形和拓扑结构的定义
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Vertex:
    """顶点结构，存储顶点索引"""
    index: int
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.index == other.index
    
    def __hash__(self):
        return hash(self.index)


@dataclass
class Edge:
    """边结构，连接两个顶点"""
    v1: int
    v2: int
    
    def is_equal(self, other: 'Edge') -> bool:
        """检查两条边是否相等（考虑方向不敏感）"""
        return (self.v1 == other.v1 and self.v2 == other.v2) or \
               (self.v1 == other.v2 and self.v2 == other.v1)
    
    def contains_vertex(self, vertex_index: int) -> bool:
        """检查边是否包含指定顶点"""
        return self.v1 == vertex_index or self.v2 == vertex_index
    
    def get_other_vertex(self, vertex_index: int) -> Optional[int]:
        """给定一个顶点，返回边上的另一个顶点"""
        if self.v1 == vertex_index:
            return self.v2
        elif self.v2 == vertex_index:
            return self.v1
        return None


@dataclass
class Triangle:
    """三角形结构，由三个顶点索引组成"""
    v1: int
    v2: int
    v3: int
    
    def get_vertices(self) -> List[int]:
        """获取三角形的所有顶点"""
        return [self.v1, self.v2, self.v3]
    
    def contains_vertex(self, vertex_index: int) -> int:
        """检查三角形是否包含指定顶点，返回位置(1,2,3)或0"""
        if vertex_index == self.v1:
            return 1
        elif vertex_index == self.v2:
            return 2
        elif vertex_index == self.v3:
            return 3
        return 0
    
    def contains_edge(self, v1: int, v2: int) -> bool:
        """检查三角形是否包含指定的边"""
        vertices = [self.v1, self.v2, self.v3]
        return v1 in vertices and v2 in vertices
    
    def get_ordered_vertices(self, center_vertex: int) -> List[int]:
        """以指定顶点为中心，返回有序的顶点列表"""
        if center_vertex == self.v1:
            return [self.v1, self.v2, self.v3]
        elif center_vertex == self.v2:
            return [self.v2, self.v3, self.v1]
        elif center_vertex == self.v3:
            return [self.v3, self.v1, self.v2]
        return []
    
    def is_equal(self, other: 'Triangle') -> bool:
        """检查两个三角形是否相等（考虑顶点顺序不敏感）"""
        vertices_self = {self.v1, self.v2, self.v3}
        vertices_other = {other.v1, other.v2, other.v3}
        return vertices_self == vertices_other
    
    def get_opposite_vertex(self, other_triangle: 'Triangle') -> Optional[int]:
        """找到相对于另一个三角形的对立顶点"""
        if other_triangle.contains_edge(self.v1, self.v2):
            return self.v3
        elif other_triangle.contains_edge(self.v1, self.v3):
            return self.v2
        elif other_triangle.contains_edge(self.v2, self.v3):
            return self.v1
        return None
    
    def get_edges(self) -> List[Edge]:
        """获取三角形的所有边"""
        return [
            Edge(self.v1, self.v2),
            Edge(self.v2, self.v3),
            Edge(self.v3, self.v1)
        ]


@dataclass
class Topology:
    """拓扑结构，包含顶点、边和三角形的集合"""
    vertices: List[Vertex]
    edges: List[Edge]
    triangles: List[Triangle]
    
    def __init__(self, vertices: List[Vertex] = None, 
                 edges: List[Edge] = None, 
                 triangles: List[Triangle] = None):
        self.vertices = vertices or []
        self.edges = edges or []
        self.triangles = triangles or []
    
    def add_vertex(self, vertex: Vertex):
        """添加顶点"""
        if vertex not in self.vertices:
            self.vertices.append(vertex)
    
    def add_triangle(self, triangle: Triangle):
        """添加三角形"""
        self.triangles.append(triangle)
    
    def remove_triangle(self, triangle: Triangle):
        """移除三角形"""
        for i, t in enumerate(self.triangles):
            if t.is_equal(triangle):
                self.triangles.pop(i)
                break
    
    def remove_vertex(self, vertex_index: int):
        """移除顶点"""
        self.vertices = [v for v in self.vertices if v.index != vertex_index]
    
    def get_vertex_star(self, vertex_index: int) -> List[Triangle]:
        """获取顶点的星形邻域（包含该顶点的所有三角形）"""
        star = []
        for triangle in self.triangles:
            if triangle.contains_vertex(vertex_index):
                star.append(triangle)
        return star
    
    def get_vertex_ring(self, vertex_index: int) -> List[int]:
        """获取顶点的环形邻域（相邻顶点的有序列表）"""
        star = self.get_vertex_star(vertex_index)
        if not star:
            return []
        
        # 从星形邻域构建环形邻域
        ring = []
        used_triangles = set()
        
        # 从第一个三角形开始
        current_triangle = star[0]
        used_triangles.add(id(current_triangle))
        
        # 获取第一个三角形中除中心顶点外的两个顶点
        vertices = current_triangle.get_ordered_vertices(vertex_index)
        ring.append(vertices[1])
        ring.append(vertices[2])
        
        # 继续查找相邻的三角形
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
                    if vertices[1] == last_vertex:
                        ring.append(vertices[2])
                    elif vertices[2] == last_vertex:
                        ring.append(vertices[1])
                    
                    found = True
                    break
            
            if not found:
                # 可能是边界情况
                break
        
        return ring
    
    def is_boundary_vertex(self, vertex_index: int) -> bool:
        """检查顶点是否在边界上"""
        ring = self.get_vertex_ring(vertex_index)
        if not ring:
            return False
        
        # 如果环形邻域的第一个和最后一个顶点不相同，说明在边界上
        return len(ring) < 3 or ring[0] != ring[-1]