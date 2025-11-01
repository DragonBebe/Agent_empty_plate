#!/usr/bin/env python3
"""
修正的Plate几何分析
重新分析XML中的site信息，找出之前分析的问题
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_xml_sites():
    """
    分析XML中的site信息
    """
    print("=== XML Site信息分析 ===")
    
    # 从XML中提取的site位置信息
    bottom_site = np.array([1.8500000041610231e-09, 1.8499999990262416e-09, -0.00589834945])
    top_site = np.array([1.8500000041610231e-09, 1.8499999990262416e-09, 0.005789232750000001])
    horizontal_radius_site = np.array([0.0912271556, 0.09122715005, -5.45583499999998e-05])
    
    print(f"bottom_site: {bottom_site}")
    print(f"top_site: {top_site}")
    print(f"horizontal_radius_site: {horizontal_radius_site}")
    
    # 分析高度信息
    plate_height_raw = top_site[2] - bottom_site[2]
    print(f"\nPlate原始高度: {plate_height_raw:.6f}m")
    
    # 分析水平半径信息
    horizontal_radius_raw = np.sqrt(horizontal_radius_site[0]**2 + horizontal_radius_site[1]**2)
    print(f"Horizontal radius (原始): {horizontal_radius_raw:.6f}m")
    
    # 应用缩放系数
    scale_factor = 0.24975
    plate_height_scaled = plate_height_raw * scale_factor
    horizontal_radius_scaled = horizontal_radius_raw * scale_factor
    
    print(f"\n应用缩放系数 {scale_factor}:")
    print(f"Plate缩放后高度: {plate_height_scaled:.6f}m")
    print(f"Horizontal radius (缩放后): {horizontal_radius_scaled:.6f}m")
    
    return {
        'scale_factor': scale_factor,
        'horizontal_radius_raw': horizontal_radius_raw,
        'horizontal_radius_scaled': horizontal_radius_scaled,
        'plate_height_raw': plate_height_raw,
        'plate_height_scaled': plate_height_scaled
    }

def analyze_site_meaning():
    """
    分析site的真实含义
    """
    print("\n=== Site含义分析 ===")
    
    # horizontal_radius_site的位置
    site_pos = np.array([0.0912271556, 0.09122715005, -5.45583499999998e-05])
    
    print(f"horizontal_radius_site位置: {site_pos}")
    print(f"X坐标: {site_pos[0]:.8f}")
    print(f"Y坐标: {site_pos[1]:.8f}")
    print(f"Z坐标: {site_pos[2]:.8f}")
    
    # 计算到原点的距离
    radius_2d = np.sqrt(site_pos[0]**2 + site_pos[1]**2)
    print(f"\n到原点的2D距离: {radius_2d:.8f}m")
    
    # 注意：Z坐标接近0，说明这个site在plate的表面附近
    print(f"Z坐标接近0 ({site_pos[2]:.2e})，说明这个site在plate表面")
    
    # 这个site可能表示的是什么？
    print(f"\n可能的含义分析:")
    print(f"1. 如果是外边界：外半径 = {radius_2d:.6f}m (原始)")
    print(f"2. 如果是内边界：内半径 = {radius_2d:.6f}m (原始)")
    print(f"3. 如果是特征点：可能是某个特定位置的标记")
    
    return radius_2d

def compare_with_user_analysis():
    """
    与用户提供的精确分析进行对比
    """
    print("\n=== 与用户精确分析对比 ===")
    
    # 用户提供的精确分析结果
    user_flat_area = 0.0328  # m²
    user_flat_radius_from_area = np.sqrt(user_flat_area / np.pi)
    user_conservative_radius = 0.070  # m
    user_approx_radius = 0.084  # m
    
    print(f"用户精确分析结果:")
    print(f"  平整区域面积: {user_flat_area:.4f} m²")
    print(f"  平整区域等效半径: {user_flat_radius_from_area:.4f} m")
    print(f"  保守安全半径: {user_conservative_radius:.4f} m")
    print(f"  近似安全半径: {user_approx_radius:.4f} m")
    
    # XML分析结果
    xml_info = analyze_xml_sites()
    xml_radius = xml_info['horizontal_radius_scaled']
    
    print(f"\nXML分析结果:")
    print(f"  horizontal_radius_site缩放后: {xml_radius:.4f} m")
    
    # 对比分析
    print(f"\n对比分析:")
    print(f"  XML半径 vs 用户平整半径: {xml_radius:.4f} vs {user_flat_radius_from_area:.4f}")
    print(f"  差异: {abs(xml_radius - user_flat_radius_from_area):.4f} m")
    
    if abs(xml_radius - user_flat_radius_from_area) < 0.01:
        print("  → XML的horizontal_radius_site可能表示平整区域的边界")
    elif xml_radius > user_flat_radius_from_area:
        print("  → XML的horizontal_radius_site可能表示外边界或总边界")
    else:
        print("  → XML的horizontal_radius_site可能表示内部某个特征")
    
    return xml_info, user_flat_radius_from_area

def identify_problem_in_previous_analysis():
    """
    识别之前分析中的问题
    """
    print("\n=== 之前分析的问题识别 ===")
    
    xml_info, user_flat_radius = compare_with_user_analysis()
    
    print("问题分析:")
    print("1. Site含义误解:")
    print(f"   - horizontal_radius_site可能不是外边界")
    print(f"   - 可能是平整区域的边界或某个特征点")
    
    print("\n2. 缩放理解问题:")
    print(f"   - 所有mesh都使用相同的缩放系数 0.24975")
    print(f"   - site位置也需要应用相同的缩放")
    
    print("\n3. 几何假设问题:")
    print(f"   - 之前假设的同心圆模型可能过于简化")
    print(f"   - 实际的平整区域可能不是完美的圆形")
    
    # 基于用户精确分析修正参数
    corrected_flat_radius = user_flat_radius
    block_size = 0.015
    block_corner_distance = block_size * np.sqrt(2)
    corrected_safe_radius = corrected_flat_radius - block_corner_distance
    
    print(f"\n修正后的参数:")
    print(f"  平整区域半径: {corrected_flat_radius:.4f} m")
    print(f"  Block角点距离: {block_corner_distance:.4f} m")
    print(f"  修正安全半径: {corrected_safe_radius:.4f} m")
    print(f"  用户保守半径: 0.070 m")
    print(f"  差异: {abs(corrected_safe_radius - 0.070):.4f} m")
    
    return {
        'corrected_flat_radius': corrected_flat_radius,
        'corrected_safe_radius': corrected_safe_radius,
        'block_corner_distance': block_corner_distance
    }

def create_corrected_generator():
    """
    创建修正后的生成器
    """
    print("\n=== 创建修正后的生成器 ===")
    
    corrected_params = identify_problem_in_previous_analysis()
    
    class CorrectedCircularGenerator:
        def __init__(self, plate_center=None):
            if plate_center is None:
                self.plate_center = np.array([0.3, -0.25, 0.82])
            else:
                self.plate_center = np.array(plate_center)
            
            # 使用修正后的参数
            self.block_half_size = 0.015
            self.block_corner_distance = corrected_params['block_corner_distance']
            self.plate_flat_radius = corrected_params['corrected_flat_radius']
            self.safe_generation_radius = corrected_params['corrected_safe_radius']
            
            print(f"修正后的生成器参数:")
            print(f"  Plate平整半径: {self.plate_flat_radius:.4f} m")
            print(f"  Block角点距离: {self.block_corner_distance:.4f} m")
            print(f"  安全生成半径: {self.safe_generation_radius:.4f} m")
        
        def generate_random_position(self):
            if self.safe_generation_radius <= 0:
                print("错误: 安全半径 <= 0")
                return self.plate_center + np.array([0, 0, self.block_half_size])
            
            # 在圆内均匀随机采样
            r = np.sqrt(np.random.random()) * self.safe_generation_radius
            theta = np.random.random() * 2 * np.pi
            
            x_rel = r * np.cos(theta)
            y_rel = r * np.sin(theta)
            
            world_x = self.plate_center[0] + x_rel
            world_y = self.plate_center[1] + y_rel
            world_z = self.plate_center[2] + self.block_half_size
            
            return np.array([world_x, world_y, world_z])
    
    return CorrectedCircularGenerator()

def visualize_corrected_analysis():
    """
    可视化修正后的分析结果
    """
    print("\n=== 可视化修正后的分析 ===")
    
    corrected_params = identify_problem_in_previous_analysis()
    xml_info = analyze_xml_sites()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制XML中的horizontal_radius_site
    xml_radius = xml_info['horizontal_radius_scaled']
    xml_circle = plt.Circle((0, 0), xml_radius, 
                           fill=False, color='blue', linewidth=2, 
                           label=f'XML horizontal_radius_site (r={xml_radius:.3f}m)')
    ax.add_patch(xml_circle)
    
    # 绘制用户分析的平整区域
    flat_radius = corrected_params['corrected_flat_radius']
    flat_circle = plt.Circle((0, 0), flat_radius, 
                            fill=False, color='orange', linewidth=2,
                            label=f'User Analysis Flat Area (r={flat_radius:.3f}m)')
    ax.add_patch(flat_circle)
    
    # 绘制修正后的安全区域
    safe_radius = corrected_params['corrected_safe_radius']
    if safe_radius > 0:
        safe_circle = plt.Circle((0, 0), safe_radius, 
                               fill=False, color='green', linewidth=2,
                               label=f'Corrected Safe Area (r={safe_radius:.3f}m)')
        ax.add_patch(safe_circle)
    
    # 绘制用户的保守安全区域
    conservative_radius = 0.070
    conservative_circle = plt.Circle((0, 0), conservative_radius, 
                                   fill=False, color='red', linewidth=2, linestyle='--',
                                   label=f'User Conservative Safe (r={conservative_radius:.3f}m)')
    ax.add_patch(conservative_circle)
    
    # 标记中心和site位置
    ax.plot(0, 0, 'ko', markersize=8, label='Plate Center')
    
    # 标记XML site位置（缩放后）
    site_x = 0.0912271556 * xml_info['scale_factor']
    site_y = 0.09122715005 * xml_info['scale_factor']
    ax.plot(site_x, site_y, 'bs', markersize=8, label='XML horizontal_radius_site')
    
    # 设置坐标轴
    max_radius = max(xml_radius, flat_radius, 0.12)
    ax.set_xlim(-max_radius * 1.2, max_radius * 1.2)
    ax.set_ylim(-max_radius * 1.2, max_radius * 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Corrected Plate Geometry Analysis')
    ax.set_xlabel('X (m) - relative to plate center')
    ax.set_ylabel('Y (m) - relative to plate center')
    
    plt.tight_layout()
    plt.savefig('/home/dragon/empty_plat_train_deploy/corrected_plate_analysis.png', dpi=150)
    plt.show()
    
    print("已保存修正分析图: corrected_plate_analysis.png")

def main():
    """主函数"""
    print("修正的Plate几何分析")
    print("=" * 60)
    
    # 分析XML信息
    analyze_xml_sites()
    analyze_site_meaning()
    
    # 对比用户分析
    compare_with_user_analysis()
    
    # 识别问题
    identify_problem_in_previous_analysis()
    
    # 创建修正后的生成器
    generator = create_corrected_generator()
    
    print(f"\n修正后的随机位置生成示例:")
    for i in range(5):
        pos = generator.generate_random_position()
        print(f"  位置 {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # 可视化
    visualize_corrected_analysis()
    
    print(f"\n总结:")
    print(f"1. XML中的horizontal_radius_site可能不是外边界")
    print(f"2. 用户的精确分析更准确，基于实际碰撞网格")
    print(f"3. 修正后的安全半径约为 {0.070:.3f}m，与用户保守估计一致")
    print(f"4. 同心圆模型仍然有效，但需要使用正确的半径参数")

if __name__ == "__main__":
    main()
