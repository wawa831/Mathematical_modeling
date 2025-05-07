import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import rcParams
from collections import defaultdict

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

def on_legend_click(event):
    """优化点击灵敏度的图例交互"""
    legend_line = event.artist
    main_line = legend_line.get_label().split('（')[0]  # 适配新图例格式
    related_lines = [k for k in lined.keys() if k.startswith(main_line)]
    new_visibility = not lined[related_lines[0]][0].get_visible()
    for line_name in related_lines:
        for line in lined[line_name]:
            line.set_visible(new_visibility)
    plt.draw()

def get_user_input():
    print("请输入线路（格式示例：8号线：橘园十一舍，国际学院；支线：外国语学院），空行结束：")
    lines = []
    while True:
        entry = input().strip()
        if not entry: break
        parts = entry.split(';')
        main_part = parts[0].split(':')
        main_name = main_part[0].strip()
        main_stations = [s.strip() for s in main_part[1].split(',')]
        lines.append((main_name, main_stations))
        for part in parts[1:]:
            branch_name = f"{main_name}-支线"
            branch_stations = [s.strip() for s in part.split(':')[1].split(',')]
            lines.append((branch_name, branch_stations))
    return lines

if __name__ == "__main__":
    lines = get_user_input()
    
    # 构建网络图
    G = nx.Graph()
    edge_to_lines = defaultdict(list)
    for name, stations in lines:
        for i in range(len(stations)-1):
            u, v = stations[i], stations[i+1]
            edge = tuple(sorted([u, v]))
            edge_to_lines[edge].append(name)
            G.add_edge(u, v)

    # 增强型布局算法
    pos = nx.spring_layout(G, k=0.5, iterations=200)  # 关键参数调整

    plt.figure(figsize=(16, 12), dpi=100)
    ax = plt.gca()
    
    # 节点可视化优化
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=800, 
        node_color='white', 
        edgecolors='#666666',
        linewidths=1.2
    )
    
    # 标签防重叠策略
    label_pos = {k: (v[0], v[1]+0.03) for k, v in pos.items()}  # 垂直偏移
    nx.draw_networkx_labels(
        G, label_pos, 
        font_size=9, 
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2')
    )

    # 专业配色方案
    color_palette = [
        '#2E86C1', '#E74C3C', '#2ECC71', '#8E44AD', '#F39C12',
        '#16A085', '#C0392B', '#7D3C98', '#229954'
    ]
    
    # 线路绘制优化
    lined = defaultdict(list)
    for idx, (name, stations) in enumerate(lines):
        main_line = name.split('-')[0]
        color = color_palette[idx % len(color_palette)]
        
        for i in range(len(stations)-1):
            u, v = stations[i], stations[i+1]
            edge = tuple(sorted([u, v]))
            
            start = pos[u]
            end = pos[v]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            edge_length = np.hypot(dx, dy)
            position = edge_to_lines[edge].index(name)
            
            # 动态曲率控制
            curvature = 0.1 * edge_length * (1 + position*0.3)  # 曲率更平缓
            direction = (-1)**position  # 严格交替方向
            
            if position == 0:
                vertices = [start, end]
                codes = [Path.MOVETO, Path.LINETO]
            else:
                angle = np.arctan2(dy, dx)
                ctrl_angle = np.pi/3.5  # 约60度
                ctrl1 = [
                    start[0] + curvature * np.cos(angle - ctrl_angle * direction),
                    start[1] + curvature * np.sin(angle - ctrl_angle * direction)
                ]
                ctrl2 = [
                    end[0] - curvature * np.cos(angle + ctrl_angle * direction),
                    end[1] - curvature * np.sin(angle + ctrl_angle * direction)
                ]
                vertices = [start, ctrl1, ctrl2, end]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

            path = Path(vertices, codes)
            patch = patches.PathPatch(
                path,
                lw=2.2 if position==0 else 1.8,
                edgecolor=color,
                facecolor='none',
                alpha=0.85,
                linestyle='-' 
            )
            lined[name].append(ax.add_patch(patch))

    # 强化图例交互
    main_lines = list({line[0].split('-')[0] for line in lines})
    legend_handles = [
        plt.Line2D([], [], color=color_palette[i], lw=2.5, 
                  label=f"{name}（{sum(1 for l in lines if l[0].startswith(name))}条支线）")
        for i, name in enumerate(main_lines)
    ]
    
    legend = ax.legend(
        handles=legend_handles,
        loc='upper right',
        title='线路图例',
        framealpha=0.9,
        title_fontsize=12,
        fontsize=10
    )
    for legend_line in legend.get_lines():
        legend_line.set_picker(True)
        legend_line.set_pickradius(10)  # 增大点击敏感区域

    plt.connect('pick_event', on_legend_click)
    
    # 界面美化
    ax.set_facecolor('#F8F9F9')
    plt.title("校车线路图\n", fontsize=14, loc='left', color='#2C3E50')
    plt.box(False)
    plt.tight_layout()
    plt.show()