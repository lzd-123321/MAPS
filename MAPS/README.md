# MAPS - 多分辨率自适应曲面参数化算法

基于共形映射的MAPS网格简化算法Python实现。

## 概述

MAPS（Multi-resolution Adaptive Parameterization of Surfaces，多分辨率自适应曲面参数化）是一种网格简化算法，能够在保持3D网格几何和拓扑特性的同时降低其复杂度。算法使用共形映射在简化过程中保持局部角度关系。

## 功能特性

- **共形映射**：在简化过程中保持局部角度关系
- **自适应简化**：基于几何重要性的迭代顶点移除
- **OBJ文件支持**：加载和处理3D模型文件
- **多种三角化方法**：支持扇形三角化和耳切算法
- **渐进式简化**：可控制的细节层次降低

## 安装要求

依赖环境：
- Python 3.7+
- NumPy

## 快速开始

### 处理OBJ文件

```bash
python main.py
```

这将处理 `assets/meshes/stanford-bunny.obj` 中的兔子网格，并将简化版本保存到 `assets/out/`。

### 使用API

```python
from maps_algorithm import MAPS

# 加载网格（顶点和三角形作为numpy数组）
maps = MAPS(vertices, triangles)

# 执行一次简化迭代
success = maps.level_down()

# 获取当前网格
vertices, triangles = maps.get_current_mesh()

# 获取统计信息
stats = maps.get_statistics()
print(f"顶点数: {stats['num_vertices']}")
print(f"三角形数: {stats['num_triangles']}")
```

### 使用方法

直接运行主程序：

```bash
python main.py
```

或使用网格生成器创建测试网格：

```bash
python mesh_generator.py
```

## 算法原理

MAPS算法的工作流程如下：

1. **顶点优先级计算**：基于局部几何特征计算移除优先级
2. **独立集选择**：选择互不相邻的顶点集合进行同时移除
3. **共形映射**：将3D顶点邻域映射到2D平面
4. **重新三角化**：为简化区域生成新的三角化
5. **网格更新**：更新网格结构并保持连通性

## 项目结构与文件职责

```
MAPS/
├── main.py                  # 主程序入口
├── mesh_loader.py          # OBJ文件加载和保存
├── maps_algorithm.py       # MAPS主算法协调器
├── maps_mesh.py            # 网格数据管理
├── data_structures.py      # 基础数据结构定义
├── simplification.py       # 顶点简化算法
├── conformal_mapping.py    # 共形映射实现
├── triangulation.py        # 三角化算法
├── geometry_utils.py       # 几何计算工具
├── README.md               # 项目文档（可选）
└── assets/                 # 资源文件夹
    ├── meshes/             # 输入网格文件
    └── out/                # 输出结果文件
```

### 核心文件详细说明

**main.py** - 程序入口点
- 解析命令行参数和配置
- 协调整个网格简化流程
- 管理输入输出文件路径
- 显示进度和统计信息

**mesh_loader.py** - 网格文件I/O处理
- load_obj_file(): 解析OBJ文件格式，提取顶点坐标和面片索引
- save_obj_file(): 将网格数据保存为标准OBJ格式
- 处理文件路径和错误异常

**maps_algorithm.py** - MAPS算法核心协调器
- MAPS类: 整合所有子算法模块
- level_down(): 执行一次完整的简化迭代
- get_current_mesh(): 获取当前网格状态
- get_statistics(): 提供详细的简化统计信息
- 管理算法参数和配置

**maps_mesh.py** - 网格数据结构管理
- MapsMesh类: 核心网格数据容器
- 维护顶点位置数组（P）和拓扑连接关系
- 管理双射映射，追踪原始顶点到当前顶点的对应关系
- 提供网格验证和完整性检查功能

**data_structures.py** - 基础几何数据类型
- Vertex: 顶点结构，存储索引信息
- Triangle: 三角形结构，定义三个顶点的连接
- Edge: 边结构，连接两个顶点
- Topology: 拓扑管理器，维护所有几何元素的集合和关系

**simplification.py** - 顶点简化策略
- compute_vertex_priorities(): 基于面积和曲率计算顶点重要性
- extract_independent_set(): 选择互不相邻的顶点集合进行安全移除
- extract_vertex_stars(): 提取顶点的星形邻域结构
- find_ring_from_star(): 从星形邻域构建环形边界
- validate_vertex_removal(): 验证顶点移除的几何合法性

**conformal_mapping.py** - 保角映射算法
- compute_conformal_mapping(): 核心共形映射，将3D环形邻域映射到2D圆盘
- 使用复数z^a变换保持局部角度关系
- interpolate_point_in_mapped_ring(): 在2D映射中插值新点位置
- 处理边界顶点和内部顶点的不同映射策略

**triangulation.py** - 网格重建算法
- triangulate_from_mapped_ring(): 主要三角化接口
- simple_triangulation_from_ring(): 扇形三角化，适用于凸多边形
- ear_cutting_triangulation(): 耳切法，处理任意简单多边形
- 生成高质量三角形，避免细长三角形

**geometry_utils.py** - 基础几何计算库
- 向量运算：vector_angle(), normalize_vector(), point_distance()
- 三角形计算：triangle_area_2d(), triangle_area_3d()
- 几何判断：point_in_triangle_2d(), compute_barycentric_coordinates()
- 曲率估算：compute_vertex_curvature()
- 投影变换：project_to_plane(), compute_vertex_normal()

## 性能表现

- **斯坦福兔子**（2503个顶点）：在10次迭代中减少到87个顶点（96.5%简化率）
- **处理时间**：复杂网格约30-40秒
