"""
MAPS算法主程序
功能:
- 从assets/meshes/目录加载OBJ文件
- 执行MAPS算法进行网格简化
- 将简化结果保存到assets/out/目录
"""
import os
import time

# ============================================================

# 网格文件名（assets/meshes/目录下的.obj文件）
MESH_NAME = 'bunny'

# MAPS算法参数
MAX_ITERATIONS = 10     # 最大简化迭代次数
LAMBDA_WEIGHT = 0.5     # 权重参数：面积 vs 曲率 (0.0=纯曲率, 1.0=纯面积)

# ============================================================
# 导入所有需要的模块
import mesh_loader
import maps_algorithm
load_obj_file = mesh_loader.load_obj_file
save_obj_file = mesh_loader.save_obj_file
MAPS = maps_algorithm.MAPS

# 设置路径常量
RAW_MESHES = 'assets/meshes/'
OUTPUT_DIR = 'assets/out/'

def main():
    """主函数"""
    print("="*50)
    
    name = MESH_NAME 
    #================================================
    
    path = f'{RAW_MESHES}{name}.obj'
    print(f"正在加载网格文件: {path}")
    
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        print(f"可用的网格文件:")
        if os.path.exists(RAW_MESHES):
            for file in os.listdir(RAW_MESHES):
                if file.endswith('.obj'):
                    print(f"  - {file[:-4]}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载网格
    try:
        vertices, triangles = load_obj_file(path)
    except Exception as e:
        print(f"加载网格失败: {e}")
        return
    
    # 保存原始网格到输出目录
    save_obj_file(vertices, triangles, f"{OUTPUT_DIR}{name}_original.obj")
    
    print(f"参数设置: 最大迭代={MAX_ITERATIONS}, 权重参数={LAMBDA_WEIGHT}")
    
    # 初始化MAPS
    try:
        maps = MAPS(vertices, triangles, lambda_weight=LAMBDA_WEIGHT)
    except Exception as e:
        print(f"MAPS初始化失败: {e}")
        return
    
    # 执行简化
    print(f"\n开始执行最多 {MAX_ITERATIONS} 次简化迭代...")
    
    iteration = 0
    total_start_time = time.time()
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        start_time = time.time()
        success = maps.level_down()
        end_time = time.time()
        
        if not success:
            print("无法继续简化，停止迭代")
            break
        
        # 获取统计信息
        stats = maps.get_statistics()
        print(f"第 {iteration} 轮简化完成: {stats['current_vertices']} 顶点, {stats['current_triangles']} 三角形 (简化率: {stats['simplification_ratio']:.1%})")
        
        # 保存中间结果
        current_vertices, current_triangles = maps.get_current_mesh()
        if len(current_triangles) > 0:
            save_obj_file(current_vertices, current_triangles, 
                         f"{OUTPUT_DIR}{name}_simplified_{iteration}.obj")
    
    total_end_time = time.time()
    
    # 最终统计
    print("\n" + "="*50)
    print("简化完成！")
    print("="*50)
    
    final_stats = maps.get_statistics()
    print(f"总耗时: {total_end_time - total_start_time:.3f}s")
    print(f"总迭代次数: {iteration}")
    print(f"原始顶点数: {final_stats['original_vertices']}")
    print(f"最终顶点数: {final_stats['current_vertices']}")
    print(f"最终三角形数: {final_stats['current_triangles']}")
    print(f"移除顶点数: {final_stats['removed_vertices']}")
    print(f"无法移除顶点数: {final_stats['unremovable_vertices']}")
    print(f"总简化率: {final_stats['simplification_ratio']:.1%}")
    
    # 保存最终结果
    final_vertices, final_triangles = maps.get_current_mesh()
    if len(final_triangles) > 0:
        save_obj_file(final_vertices, final_triangles, f"{OUTPUT_DIR}{name}_final.obj")
        print(f"网格已保存到: {OUTPUT_DIR}{name}_final.obj")
        
        print(f"\n输出文件:")
        print(f"  - {OUTPUT_DIR}{name}_original.obj (原始网格)")
        for i in range(1, iteration + 1):
            print(f"  - {OUTPUT_DIR}{name}_simplified_{i}.obj (第{i}次简化)")
        print(f"  - {OUTPUT_DIR}{name}_final.obj (最终结果)")


if __name__ == "__main__":
    main()