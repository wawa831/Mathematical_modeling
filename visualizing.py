import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import rcParams
from collections import defaultdict

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def on_legend_click(event):
    """处理图例点击事件"""
    legend_line = event.artist
    main_line = legend_line.get_label().split("-")[0].split("（")[0].strip()

    # 切换相关线路可见性
    for line_name in lined:
        if line_name.startswith(main_line):
            for line in lined[line_name]:
                line.set_visible(not line.get_visible())
    plt.draw()

def parse_lines(input_data):
    """解析输入数据"""
    lines = []
    for line in input_data.strip().split('\n'):
        if not line.strip():
            continue
        main_part, *branches = line.split(';')
        main_name, stations = main_part.split(':', 1)
        main_name = main_name.strip()

        # 处理主线
        lines.append((main_name, [s.strip() for s in stations.split(',')]))

        # 处理支线
        for i, branch in enumerate(branches, 1):
            branch_name, branch_stations = branch.split(':', 1)
            branch_name = f"{main_name}-{branch_name.strip()}"
            lines.append((branch_name, [s.strip() for s in branch_stations.split(',')]))
    return lines

# 预置输入数据
input_data = """
1号线:二号门,楠园,8教(李园),9教,国际学院
2号线:二号门,楠园,8教(李园) 
3号线:二号门,楠园,外国语学院,27教(梅园),橘园食堂,桃园,9教,国际学院;支线:二号门,竹园;支线:竹园,楠园
4号线:二号门,楠园,外国语学院,27教(梅园),橘园十一舍,橘园食堂
5号线:二号门,楠园,8教(李园),国际学院
6号线:二号门,楠园,外国语学院,27教(梅园),橘园十一舍,橘园食堂,桃园,8教(李园);支线:二号门,竹园;支线:竹园,楠园
7号线:二号门,楠园,外国语学院,27教(梅园),橘园十一舍,橘园食堂,桃园,9教,8教(李园)
8号线:8教(李园),外国语学院,27教(梅园),橘园十一舍,橘园食堂,桃园,9教
支线(经管专线):二号门,楠园,经管院
"""

if __name__ == "__main__":
    # 解析输入数据
    lines = parse_lines(input_data)

    # 构建网络图
    G = nx.Graph()
    for name, stations in lines:
        for i in range(len(stations) - 1):
            G.add_edge(stations[i], stations[i + 1])

    # 自动布局
    pos = nx.kamada_kawai_layout(G, dim=2)

    # 创建图表
    plt.figure(figsize=(14, 10), facecolor='#f4f4f4')  # 设置背景颜色
    ax = plt.gca()

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='white', edgecolors='gray', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='SimHei', ax=ax)

    # 颜色分配，使用更柔和的颜色
    color_map = plt.cm.Set2.colors
    main_lines = list(set([name.split("-")[0].split("（")[0].strip() for name, _ in lines]))
    color_dict = {ml: color_map[i % len(color_map)] for i, ml in enumerate(main_lines)}

    # 绘制直线
    lined = defaultdict(list)
    for name, stations in lines:
        main_name = name.split("-")[0].split("（")[0].strip()
        color = color_dict[main_name]

        # 绘制线段
        for i in range(len(stations) - 1):
            u, v = stations[i], stations[i + 1]
            line, = ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=color,
                lw=3 if name == main_name else 2,
                alpha=0.7,
                solid_capstyle='round'
            )
            lined[name].append(line)

    # 创建图例
    legend_handles = [
        plt.Line2D([], [], color=color_dict[name], lw=3, label=name)
        for name in main_lines
    ]
    legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='gray')

    # 绑定点击事件
    for legend_line in legend.get_lines():
        legend_line.set_picker(True)
        legend_line.set_pickradius(10)
    plt.connect('pick_event', on_legend_click)

    # 显示表格
    df = pd.DataFrame([(name, '→'.join(stations)) for name, stations in lines],
                      columns=['线路名称', '站点列表'])
    print("\n线路信息表：")
    print(df.to_string(index=False))

    plt.title("校车线路图（点击图例切换显示）", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

