def visualize_routes(G):
    # 创建一个更大的图形
    plt.figure(figsize=(24, 16), facecolor='#f0f0f0')
    
    # 使用更合理的节点布局，模拟真实地理位置
    pos = {
        # 北区
        "竹园": (8, 9),
        "国际学院(外办)": (6, 9),
        
        # 中北区
        "楠园": (7, 7),
        "外国语学院": (5, 7),
        
        # 中区
        "二十七教(梅园)": (6, 5),
        "八教(李园)": (4, 5),
        
        # 中南区
        "橘园十一舍": (2, 4),
        "橘园食堂": (3, 4),
        "九教(芸文楼)": (5, 4),
        
        # 南区
        "桃园": (2, 2),
        "经管院": (7, 2),
    }
    
    # 使用更优雅的配色方案
    route_styles = {
        1: {'color': '#E41A1C', 'name': '1号线', 'linestyle': '-'},    # 深红色
        2: {'color': '#377EB8', 'name': '2号线', 'linestyle': '--'},   # 深蓝色
        3: {'color': '#4DAF4A', 'name': '3号线', 'linestyle': '-'},    # 深绿色
        4: {'color': '#984EA3', 'name': '4号线', 'linestyle': '-.'},   # 紫色
        5: {'color': '#FF7F00', 'name': '5号线', 'linestyle': ':'},    # 橙色
        6: {'color': '#FFFF33', 'name': '6号线', 'linestyle': '--'},   # 黄色
        7: {'color': '#A65628', 'name': '7号线', 'linestyle': '-'},    # 棕色
        8: {'color': '#F781BF', 'name': '8号线', 'linestyle': '-.'},   # 粉色
        9: {'color': '#999999', 'name': '支线', 'linestyle': ':'}      # 灰色
    }
    
    # 添加底图网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制路线，使用平滑的曲线
    for route_id, style in route_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['route'] == route_id]
        if edges:
            nx.draw_networkx_edges(G, pos, 
                                 edgelist=edges,
                                 edge_color=style['color'],
                                 style=style['linestyle'],
                                 width=3,
                                 alpha=0.8,
                                 arrows=True,
                                 arrowsize=20,
                                 connectionstyle='arc3,rad=0.2',
                                 label=style['name'])
    
    # 绘制站点，使用渐变色填充 
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if "教" in node:
            node_colors.append('#B3E5FC')  # 浅蓝色
            node_sizes.append(3000)
        elif "园" in node:
            node_colors.append('#C8E6C9')  # 浅绿色
            node_sizes.append(3500)
        else:
            node_colors.append('#FFECB3')  # 浅黄色
            node_sizes.append(3000)
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 edgecolors='white',
                                 linewidths=2,
                                 alpha=0.9)
    nodes.set_zorder(20)  # 确保节点在边之上
    
    # 添加节点标签，使用阴影效果
    labels = nx.draw_networkx_labels(G, pos,
                                   font_size=12,
                                   font_family='SimHei',
                                   font_weight='bold')
    
    # 创建精美的图例
    legend = plt.legend(loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       title='校车路线图例',
                       title_fontsize=14,
                       fontsize=12,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       borderpad=1,
                       labelspacing=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    
    # 添加标题
    plt.title("西南大学北碚校区校车路线示意图",
              fontsize=20,
              pad=20,
              fontweight='bold',
              fontfamily='SimHei')
    
    # 添加路线详细信息
    route_info = [
        "1号线：经管院 ↔ 楠园 ↔ 八教 ↔ 九教 ↔ 国际学院",
        "2号线：经管院 ↔ 楠园 ↔ 八教",
        "3号线：竹园 ↔ 楠园 ↔ 外院 ↔ 二七教 ↔ 橘食堂 ↔ 桃园 ↔ 九教 ↔ 国际学院",
        "4号线：楠园 ↔ 外院 ↔ 二七教 ↔ 橘十一 ↔ 橘食堂",
        "5号线：楠园 ↔ 八教 ↔ 国际学院",
        "6号线：竹园 ↔ 楠园 ↔ 外院 ↔ 二七教 ↔ 橘十一 ↔ 橘食堂 ↔ 桃园 ↔ 八教",
        "7号线：楠园 ↔ 外院 ↔ 二七教 ↔ 橘十一 ↔ 橘食堂 ↔ 桃园 ↔ 九教 ↔ 八教",
        "8号线：八教 ↔ 外院 ↔ 二七教 ↔ 橘十一 ↔ 橘食堂 ↔ 桃园 ↔ 九教",
        "支线：竹园 ↔ 楠园 ↔ 外院 ↔ 二七教 ↔ 国际学院"
    ]
    
    # 使用精美的文本框显示路线信息
    plt.figtext(1.02, 0.95, '\n'.join(route_info),
                fontsize=11,
                fontfamily='SimHei',
                bbox=dict(facecolor='white',
                         edgecolor='#CCCCCC',
                         boxstyle='round,pad=1',
                         alpha=0.9))
    
    # 调整布局
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # 为右侧说明留出空间
    
    # 添加版权信息
    plt.figtext(0.99, 0.01, 
                '© 2024 西南大学数学建模协会',
                ha='right',
                fontsize=8,
                alpha=0.5)
    
    plt.show()
