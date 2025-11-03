"""
OBJ文件加载和保存模块

提供3D网格文件的输入输出功能：
- 加载OBJ格式的3D模型文件
- 解析顶点坐标和面片索引
- 保存处理后的网格到OBJ文件
- 支持标准OBJ文件格式

为MAPS算法提供网格数据的读取和结果保存功能。
"""

import numpy as np
import os
from typing import Tuple, List


def load_obj_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载OBJ文件
    
    Args:
        filepath: OBJ文件路径
    
    Returns:
        (vertices, triangles) 顶点坐标和三角形索引
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OBJ文件不存在: {filepath}")
    
    vertices = []
    triangles = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('v '):
                # 顶点坐标
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
            
            elif line.startswith('f '):
                # 面（三角形）
                parts = line.split()
                if len(parts) >= 4:
                    # OBJ文件索引从1开始，转换为从0开始
                    face_indices = []
                    for part in parts[1:]:
                        # 处理可能的纹理坐标和法向量索引 (vertex/texture/normal)
                        vertex_index = int(part.split('/')[0]) - 1
                        face_indices.append(vertex_index)
                    
                    # 如果是四边形，分解为两个三角形
                    if len(face_indices) == 3:
                        triangles.append(face_indices)
                    elif len(face_indices) == 4:
                        # 分解四边形为两个三角形
                        triangles.append([face_indices[0], face_indices[1], face_indices[2]])
                        triangles.append([face_indices[0], face_indices[2], face_indices[3]])
    
    if not vertices:
        raise ValueError(f"OBJ文件中没有顶点数据: {filepath}")
    
    if not triangles:
        raise ValueError(f"OBJ文件中没有面数据: {filepath}")
    
    vertices = np.array(vertices, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.int32)
    
    print(f"加载OBJ文件: {os.path.basename(filepath)}")
    print(f"  顶点数: {len(vertices)}")
    print(f"  三角形数: {len(triangles)}")
    
    return vertices, triangles


def get_available_meshes(assets_dir: str = "assets/meshes") -> List[str]:
    """
    获取可用的网格文件列表
    
    Args:
        assets_dir: 资源目录路径
    
    Returns:
        OBJ文件名列表
    """
    if not os.path.exists(assets_dir):
        return []
    
    obj_files = []
    for filename in os.listdir(assets_dir):
        if filename.lower().endswith('.obj'):
            obj_files.append(filename)
    
    return sorted(obj_files)


def save_obj_file(vertices: np.ndarray, triangles: np.ndarray, filepath: str):
    """
    保存网格为OBJ文件
    
    Args:
        vertices: 顶点坐标数组
        triangles: 三角形索引数组
        filepath: 输出文件路径
    """
    with open(filepath, 'w') as f:
        f.write(f"# Mesh exported from MAPS\n")
        f.write(f"# {len(vertices)} vertices, {len(triangles)} triangles\n\n")
        
        # 写入顶点
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        f.write("\n")
        
        # 写入面（OBJ文件索引从1开始）
        for triangle in triangles:
            f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
    
    print(f"网格已保存到: {filepath}")


if __name__ == "__main__":
    # 测试加载器
    available_meshes = get_available_meshes()
    print("可用的网格文件:")
    for i, mesh in enumerate(available_meshes):
        print(f"  {i+1}. {mesh}")
    
    if available_meshes:
        # 测试加载第一个文件
        test_file = os.path.join("assets/meshes", available_meshes[0])
        try:
            vertices, triangles = load_obj_file(test_file)
            print(f"\n成功加载测试文件: {available_meshes[0]}")
        except Exception as e:
            print(f"加载测试文件失败: {e}")