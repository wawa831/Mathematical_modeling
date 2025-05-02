import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 站点间行驶时间信息
travel_times = {
    ("竹园", "楠园"): 3.60,
    ("楠园", "经管院"): 5.20,
    ("楠园", "外国语学院"): 4.60,
    ("楠园", "八教(李园)", "顺"): 12.00,
    ("楠园", "八教(李园)", "逆"): 4.00,
    ("外国语学院", "八教(李园)"): 7.30,
    ("外国语学院", "二十七教(梅园)"): 1.04,
    ("八教(李园)", "九教(芸文楼)"): 2.80,
    ("八教(李园)", "国际学院(外办)"): 3.76,
    ("八教(李园)", "桃园"): 3.12,
    ("国际学院(外办)", "九教(芸文楼)"): 1.84,
    ("国际学院(外办)", "二十七教(梅园)"): 5.64,
    ("二十七教(梅园)", "橘园十一舍"): 1.12,
    ("二十七教(梅园)", "橘园食堂"): 2.08,
    ("橘园十一舍", "橘园食堂"): 0.88,
    ("橘园食堂", "桃园"): 1.04,
    ("桃园", "九教(芸文楼)"): 0.37
}

# ===================== 数据结构定义 =====================

class Bus:
    def __init__(self, bus_id, route_id, capacity, bus_type):
        self.bus_id = bus_id
        self.route_id = route_id
        self.capacity = capacity
        self.bus_type = bus_type  # 普通/高峰/支线
        self.location = None
        self.passengers = 0
        self.total_distance = 0   # 新增：记录总行驶距离
        self.last_route_change_time = 0  # 新增：记录高峰车换线时间
        self.current_route_stops = []  # 新增：当前路线已经过站点

    def can_change_route(self, current_time):
        """判断高峰车是否可以换线"""
        if "高峰" not in self.bus_type:
            return False
        if not self.current_route_stops:
            return True
        return set(self.current_route_stops) == set(self.get_route_stops())


class Station:
    def __init__(self, name, demand_dict):
        self.name = name
        self.demand_dict = demand_dict  # {目标站点: 人数}


# ===================== 数据初始化 =====================

def load_station_data():
    # 早高峰时期的需求(7:30)
    morning_demands = {
        "竹园": {
            "国际学院(外办)": 120, "八教(李园)": 120, "九教(芸文楼)": 120,
            "二十七教(梅园)": 120, "外国语学院": 120
        },
        "楠园": {
            "国际学院(外办)": 120, "八教(李园)": 120, "九教(芸文楼)": 120,
            "二十七教(梅园)": 120, "经管院": 120
        },
        "二十七教(梅园)": {"八教(李园)": 200, "九教(芸文楼)": 200},
        "八教(李园)": {"九教(芸文楼)": 250, "二十七教(梅园)": 250},
        "橘园十一舍": {"八教(李园)": 100, "外国语学院": 100},
        "橘园食堂": {"八教(李园)": 133, "二十七教(梅园)": 133, "外国语学院": 134},
        "桃园": {"八教(李园)": 100, "二十七教(梅园)": 100, "外国语学院": 100}
    }

    # 下课时期的需求(9:45)
    afternoon_demands = {
        "国际学院": {"竹园": 100},
        "二十七教": {
            "竹园": 70, "橘园": 70, "楠园": 70,
            "桃园": 70, "二号门": 70
        },
        "九教": {"八教": 57, "梅园": 57, "楠园": 56},
        "八教": {"楠园": 175, "竹园": 175},
        "外国语学院": {
            "楠园": 38, "竹园": 38, "桃园": 37, "橘园": 37
        },
        "经济管理学院": {"楠园": 80},
        "三十教": {"楠园": 50, "二十五教": 50}
    }

    # 创建所有站点的对象，包括目标站点
    all_stations = set()

    # 收集所有站点（包括起点和终点）
    for source, demands in morning_demands.items():
        all_stations.add(source)
        all_stations.update(demands.keys())

    for source, demands in afternoon_demands.items():
        all_stations.add(source)
        all_stations.update(demands.keys())

    # 创建站点对象
    stations = []
    for station_name in all_stations:
        demands = {}
        if station_name in morning_demands:
            demands = morning_demands[station_name]
        stations.append(Station(station_name, demands))

    return stations, morning_demands, afternoon_demands


def load_bus_data():
    buses = []
    # 根据题目给定的车辆数据创建校车对象
    bus_configs = [
        (1, 3, 13, "普通"),  # (路线号, 数量, 容量, 类型)
        (2, 2, 13, "普通"),
        (3, 5, 13, "普通"),
        (4, 8, 26, "普通"),
        (5, 6, 13, "普通"),
        (6, 5, 26, "普通"),
        (7, 6, 26, "普通"),
        (8, 4, 13, "普通"),
        (9, 2, 13, "支线"),
        (10, 4, 13, "高峰车1"),
        (11, 6, 26, "高峰车2")
    ]

    bus_id = 1
    for route_id, count, capacity, bus_type in bus_configs:
        for _ in range(count):
            buses.append(Bus(bus_id, route_id, capacity, bus_type))
            bus_id += 1

    return buses


def load_routes():
    G = nx.DiGraph()

    # 添加站点
    stations = [
        "竹园", "楠园", "经管院", "八教(李园)", "九教(芸文楼)",
        "国际学院(外办)", "外国语学院", "二十七教(梅园)",
        "橘园十一舍", "橘园食堂", "桃园"
    ]

    # 先添加所有节点
    G.add_nodes_from(stations)

    # 添加路线（按题目给定路线）
    routes = [
        # 1号线
        ("经管院", "楠园", {"route": 1, "distance": 0.5}),
        ("楠园", "八教(李园)", {"route": 1, "distance": 0.4}),
        ("八教(李园)", "九教(芸文楼)", {"route": 1, "distance": 0.3}),
        ("九教(芸文楼)", "国际学院(外办)", {"route": 1, "distance": 0.4}),

        # 2号线
        ("经管院", "楠园", {"route": 2, "distance": 0.5}),
        ("楠园", "八教(李园)", {"route": 2, "distance": 0.4}),

        # 3号线
        ("竹园", "楠园", {"route": 3, "distance": 0.6}),
        ("楠园", "外国语学院", {"route": 3, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 3, "distance": 0.3}),
        ("二十七教(梅园)", "橘园食堂", {"route": 3, "distance": 0.5}),
        ("橘园食堂", "桃园", {"route": 3, "distance": 0.3}),
        ("桃园", "九教(芸文楼)", {"route": 3, "distance": 0.4}),
        ("九教(芸文楼)", "国际学院(外办)", {"route": 3, "distance": 0.4}),

        # 4号线
        ("楠园", "外国语学院", {"route": 4, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 4, "distance": 0.3}),
        ("二十七教(梅园)", "橘园十一舍", {"route": 4, "distance": 0.4}),
        ("橘园十一舍", "橘园食堂", {"route": 4, "distance": 0.3}),

        # 5号线
        ("楠园", "八教(李园)", {"route": 5, "distance": 0.4}),
        ("八教(李园)", "国际学院(外办)", {"route": 5, "distance": 0.5}),

        # 6号线
        ("竹园", "楠园", {"route": 6, "distance": 0.6}),
        ("楠园", "外国语学院", {"route": 6, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 6, "distance": 0.3}),
        ("二十七教(梅园)", "橘园十一舍", {"route": 6, "distance": 0.4}),
        ("橘园十一舍", "橘园食堂", {"route": 6, "distance": 0.3}),
        ("橘园食堂", "桃园", {"route": 6, "distance": 0.3}),
        ("桃园", "八教(李园)", {"route": 6, "distance": 0.5}),

        # 7号线
        ("楠园", "外国语学院", {"route": 7, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 7, "distance": 0.3}),
        ("二十七教(梅园)", "橘园十一舍", {"route": 7, "distance": 0.4}),
        ("橘园十一舍", "橘园食堂", {"route": 7, "distance": 0.3}),
        ("橘园食堂", "桃园", {"route": 7, "distance": 0.3}),
        ("桃园", "九教(芸文楼)", {"route": 7, "distance": 0.4}),
        ("九教(芸文楼)", "八教(李园)", {"route": 7, "distance": 0.3}),

        # 8号线
        ("八教(李园)", "外国语学院", {"route": 8, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 8, "distance": 0.3}),
        ("二十七教(梅园)", "橘园十一舍", {"route": 8, "distance": 0.4}),
        ("橘园十一舍", "橘园食堂", {"route": 8, "distance": 0.3}),
        ("橘园食堂", "桃园", {"route": 8, "distance": 0.3}),
        ("桃园", "九教(芸文楼)", {"route": 8, "distance": 0.4}),

        # 支线
        ("竹园", "楠园", {"route": 9, "distance": 0.6}),
        ("楠园", "外国语学院", {"route": 9, "distance": 0.4}),
        ("外国语学院", "二十七教(梅园)", {"route": 9, "distance": 0.3}),
        ("二十七教(梅园)", "国际学院(外办)", {"route": 9, "distance": 0.4})
    ]

    # 添加边
    for start, end, attr in routes:
        G.add_edge(start, end, **attr)
        G.add_edge(end, start, **attr)  # 添加反向边

    return G


# ===================== 调度与仿真核心 =====================

def calculate_route_demands(stations, G):
    """计算每条路线的总需求量"""
    route_demands = {}
    for station in stations:
        for dest, demand in station.demand_dict.items():
            try:
                path = nx.shortest_path(G, station.name, dest, weight='distance')
                for i in range(len(path) - 1):
                    edge = G[path[i]][path[i + 1]]
                    route = edge['route']
                    route_demands[route] = route_demands.get(route, 0) + demand
            except nx.NodeNotFound:
                print(f"警告: 找不到从 {station.name} 到 {dest} 的路径")
                continue
    return route_demands


def allocate_buses(buses, route_demands):
    """根据路线需求量分配车辆"""
    # 给高峰车分配需求量最大的路线
    peak_buses = [bus for bus in buses if "高峰" in bus.bus_type]
    normal_buses = [bus for bus in buses if "高峰" not in bus.bus_type]

    sorted_routes = sorted(route_demands.items(), key=lambda x: x[1], reverse=True)

    # 分配高峰车
    for bus in peak_buses:
        if sorted_routes:
            bus.route_id = sorted_routes[0][0]
            sorted_routes.pop(0)

    # 分配普通车和支线车
    for bus in normal_buses:
        if bus.bus_type != "支线":  # 保持原有路线
            continue
        if sorted_routes:
            bus.route_id = sorted_routes[0][0]
            sorted_routes.pop(0)

    return buses


def assign_buses_to_routes(buses, stations, G, demand_type="morning"):
    """车辆调度分配算法"""
    route_demands = calculate_route_demands(stations, G)
    return allocate_buses(buses, route_demands)


def optimize_peak_bus_allocation(buses, stations, G):
    """高峰车专用优化算法"""
    peak_buses = [bus for bus in buses if "高峰" in bus.bus_type]
    route_demands = calculate_route_demands(stations, G)
    
    def calculate_route_load(route_id):
        """计算路线负载情况"""
        total_demand = route_demands.get(route_id, 0)
        current_capacity = sum(bus.capacity for bus in buses 
                             if bus.route_id == route_id)
        return total_demand / (current_capacity + 1)  # 避免除零
    
    # 动态分配高峰车
    for bus in peak_buses:
        # 找出当前负载最高的路线
        route_loads = {rid: calculate_route_load(rid) 
                      for rid in range(1, 10)}
        best_route = max(route_loads.items(), key=lambda x: x[1])[0]
        bus.route_id = best_route


def simulate_transport(buses, stations, G, speed=15, load_time=3):
    """改进的校车运输仿真"""
    total_time = 0
    remaining_passengers = {station.name: station.demand_dict.copy() for station in stations}
    
    # 新增：拥堵系数计算
    def get_congestion_factor(time, location):
        base_factor = 1.0
        if time < 60:  # 早高峰前60分钟
            base_factor *= 1.3
        if any(loc in location for loc in ["八教", "九教", "二十七教"]):
            base_factor *= 1.2  # 教学区拥堵加成
        return base_factor
    
    while any(bool(demands) for demands in remaining_passengers.values()):
        for bus in buses:
            route_edges = [(u, v) for u, v, d in G.edges(data=True) if d['route'] == bus.route_id]
            if not route_edges:
                continue

            current_passengers = 0
            bus.location = route_edges[0][0]

            for start, end in route_edges:
                if start not in remaining_passengers:
                    remaining_passengers[start] = {}
                if end not in remaining_passengers:
                    remaining_passengers[end] = {}

                # 获取拥堵系数
                congestion = get_congestion_factor(total_time, start)

                # 计算行驶时间
                if (start, end) in travel_times:
                    travel_time = travel_times[(start, end)] * congestion
                elif (start, end, "顺") in travel_times:
                    travel_time = travel_times[(start, end, "顺")] * congestion
                elif (start, end, "逆") in travel_times:
                    travel_time = travel_times[(start, end, "逆")] * congestion
                else:
                    distance = G[start][end]['distance']
                    travel_time = (distance / (speed / congestion)) * 60

                # 上下车时间
                if remaining_passengers[start]:
                    boarding_count = min(
                        bus.capacity - current_passengers,
                        sum(remaining_passengers[start].values())
                    )
                    boarding_time = boarding_count * (2 + random.random()) / 60
                    travel_time += boarding_time * congestion

                    # 更新乘客
                    for dest in list(remaining_passengers[start].keys()):
                        if current_passengers < bus.capacity:
                            passengers_to_board = min(
                                remaining_passengers[start][dest],
                                bus.capacity - current_passengers
                            )
                            current_passengers += passengers_to_board
                            remaining_passengers[start][dest] -= passengers_to_board
                            if remaining_passengers[start][dest] == 0:
                                del remaining_passengers[start][dest]

                total_time = max(total_time, travel_time)
                bus.location = end

                # 高峰车换线等待时间
                if "高峰" in bus.bus_type and bus.route_id != bus.last_route_change_time // 30:
                    total_time += 0.5  # 30秒转换时间
                    bus.last_route_change_time = total_time

    return total_time


def init_genetic_algorithm(buses, route_count, population_size):
    """初始化遗传算法的种群"""
    population = []
    for _ in range(population_size):
        # 随机生成一个解，每个解是车辆到路线的映射
        solution = []
        peak_buses = [bus for bus in buses if "高峰" in bus.bus_type]
        normal_buses = [bus for bus in buses if "高峰" not in bus.bus_type]

        # 优先分配高峰车
        peak_demands = [random.randint(1, route_count) for _ in peak_buses]
        solution.extend(peak_demands)

        # 分配普通车
        normal_solution = [random.randint(1, route_count) for _ in normal_buses]
        solution.extend(normal_solution)

        # 检查解的合法性
        while not is_solution_valid(solution, route_count):
            solution = [random.randint(1, route_count) for _ in buses]

        population.append(solution)
    return population


def is_solution_valid(solution, route_count):
    """检查解的合法性，确保每条路线至少有一辆车"""
    covered_routes = set(solution)
    return len(covered_routes) == route_count


def fitness_function(solution, buses, stations, G):
    """多目标优化的适应度函数"""
    temp_buses = buses.copy()
    for i, bus in enumerate(temp_buses):
        bus.route_id = solution[i]
    
    time = simulate_transport(temp_buses, stations, G)
    coverage = calculate_route_coverage(temp_buses, G)
    balance = evaluate_route_balance(temp_buses, G)
    peak_efficiency = evaluate_peak_bus_efficiency(temp_buses)
    
    # 多目标权重
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    
    return -(time * w1 - coverage * w2 - balance * w3 - peak_efficiency * w4)


def calculate_route_coverage(buses, G):
    """计算路线覆盖率"""
    covered_routes = set()
    for bus in buses:
        covered_routes.add(bus.route_id)
    return len(covered_routes) / 9  # 9 条路线


def optimize_bus_allocation(buses, stations, G, generations=100, population_size=200):
    """车辆自由调配优化算法，使用遗传算法"""
    route_count = 9
    population = init_genetic_algorithm(buses, route_count, population_size)

    # 遗传算法主循环
    for generation in range(generations):
        # 计算适应度
        fitness_scores = [fitness_function(solution, buses, stations, G) for solution in population]

        # 选择优秀个体
        elite_size = population_size // 4
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i] for i in elite_indices]

        # 交叉和变异
        while len(new_population) < population_size:
            # 选择父代
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # 交叉
            crossover_point = random.randint(0, len(buses) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            # 变异
            if random.random() < 0.1:  # 变异概率
                mutation_point = random.randint(0, len(buses) - 1)
                child[mutation_point] = random.randint(1, route_count)

            # 检查解的合法性
            if is_solution_valid(child, route_count):
                new_population.append(child)
            else:
                # 若不合法，重新生成变异点
                while not is_solution_valid(child, route_count):
                    mutation_point = random.randint(0, len(buses) - 1)
                    child[mutation_point] = random.randint(1, route_count)
                new_population.append(child)

        population = new_population

    # 返回最优解
    best_solution = max(population, key=lambda x: fitness_function(x, buses, stations, G))

    # 应用最优解
    for i, bus in enumerate(buses):
        bus.route_id = best_solution[i]

    # 计算运行时间
    time = simulate_transport(buses, stations, G)

    return time, best_solution


def optimize_free_allocation(buses, stations, G):
    """实现车辆自由调配的总体优化"""
    # 1. 早高峰优化
    morning_time, morning_solution = optimize_bus_allocation(buses, stations, G)
    print("\n早高峰优化结果:")
    print(f"优化后运行时间: {morning_time:.2f} min")
    print("车辆分配方案:", morning_solution)

    # 2. 下课高峰优化
    afternoon_time, afternoon_solution = optimize_bus_allocation(buses, stations, G)
    print("\n下课高峰优化结果:")
    print(f"优化后运行时间: {afternoon_time:.2f} min")
    print("车辆分配方案:", afternoon_solution)

    return min(morning_time, afternoon_time)


def evaluate_route_balance(buses, G):
    """评估路线分配的平衡性"""
    route_buses = {}
    for bus in buses:
        route_buses[bus.route_id] = route_buses.get(bus.route_id, 0) + 1
    
    total_routes = len(set(route_buses.keys()))
    avg_buses = sum(route_buses.values()) / total_routes
    variance = sum((count - avg_buses) ** 2 
                   for count in route_buses.values()) / total_routes
    
    return 1 / (1 + variance)  # 归一化，越平衡越接近1


def evaluate_peak_bus_efficiency(buses):
    """评估高峰车的使用效率"""
    peak_buses = [bus for bus in buses if "高峰" in bus.bus_type]
    if not peak_buses:
        return 1.0  # 如果没有高峰车，返回最高分
        
    # 计算高峰车的路线分布
    route_count = {}
    for bus in peak_buses:
        route_count[bus.route_id] = route_count.get(bus.route_id, 0) + 1
    
    # 计算路线分布的均匀度
    total_routes = len(set(route_count.keys()))
    if total_routes == 0:
        return 0.0
        
    avg_buses = len(peak_buses) / total_routes
    variance = sum((count - avg_buses) ** 2 
                  for count in route_count.values()) / total_routes
    
    # 计算高峰车的分散程度（越分散越好）
    dispersion = 1 / (1 + variance)
    
    # 计算高峰车的路线覆盖率
    coverage = total_routes / 9  # 9条路线
    
    # 综合评分
    return (dispersion * 0.6 + coverage * 0.4)


# ===================== 不确定性分析 =====================

def uncertainty_analysis(buses, stations, G):
    """不确定性分析改进版"""
    results = []
    # 设置变化范围
    variations = [
        (1.15, 1.05),  # 人数+15%，速度+5%
        (1.15, 0.95),  # 人数+15%，速度-5%
        (0.85, 1.05),  # 人数-15%，速度+5%
        (0.85, 0.95)   # 人数-15%，速度-5%
    ]

    # 计算基准时间
    base_time = simulate_transport(buses, stations, G)
    print(f"基准运行时间: {base_time:.2f} 分钟")
    
    for demand_factor, speed_factor in variations:
        # 保存原始需求
        original_demands = {
            station.name: station.demand_dict.copy() 
            for station in stations
        }
        
        # 修改需求
        for station in stations:
            for dest in station.demand_dict:
                station.demand_dict[dest] = int(station.demand_dict[dest] * demand_factor)
        
        # 仿真运行
        time = simulate_transport(buses, stations, G, speed=15*speed_factor)
        results.append({
            'scenario': f"需求{'增加' if demand_factor > 1 else '减少'}{abs(demand_factor-1)*100:.0f}%\n速度{'增加' if speed_factor > 1 else '减少'}{abs(speed_factor-1)*100:.0f}%",
            'time': time,
            'change': (time - base_time) / base_time * 100
        })
        
        # 恢复原始需求
        for station in stations:
            station.demand_dict = original_demands[station.name].copy()
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    scenarios = [r['scenario'] for r in results]
    times = [r['time'] for r in results]
    changes = [r['change'] for r in results]
    
    # 创建主柱状图
    bars = plt.bar(scenarios, times, color=['#2ecc71', '#e74c3c', '#3498db', '#f1c40f'])
    
    # 添加基准线
    plt.axhline(y=base_time, color='red', linestyle='--', alpha=0.5, label='基准时间')
    
    # 添加数值标签
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}分钟\n({change:+.1f}%)',
                ha='center', va='bottom',
                fontsize=10)
    
    # 设置图表样式
    plt.title('不同情景下的运输时间变化分析', fontsize=14, pad=20)
    plt.ylabel('运输时间（分钟）', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    plt.legend(['基准时间'])
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()
    
    return results


# ===================== 可视化与结果输出 =====================

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
        1: {'color': '#E41A1C', 'name': '1号线', 'linestyle': '-'},  # 深红色
        2: {'color': '#377EB8', 'name': '2号线', 'linestyle': '--'},  # 深蓝色
        3: {'color': '#4DAF4A', 'name': '3号线', 'linestyle': '-'},  # 深绿色
        4: {'color': '#984EA3', 'name': '4号线', 'linestyle': '-.'},  # 紫色
        5: {'color': '#FF7F00', 'name': '5号线', 'linestyle': ':'},  # 橙色
        6: {'color': '#FFFF33', 'name': '6号线', 'linestyle': '--'},  # 黄色
        7: {'color': '#A65628', 'name': '7号线', 'linestyle': '-'},  # 棕色
        8: {'color': '#F781BF', 'name': '8号线', 'linestyle': '-.'},  # 粉色
        9: {'color': '#999999', 'name': '支线', 'linestyle': ':'}  # 灰色
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

    # 添加节点标签，使用阴影效果33
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

    plt.show()


def visualize_simulation_results(morning_results, afternoon_results, free_results):
    """可视化优化结果对比"""
    plt.figure(figsize=(15, 8))
    
    times = [morning_results['time'], afternoon_results['time'], free_results['time']]
    labels = ['早高峰', '下课高峰', '自由调配']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = plt.bar(labels, times, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}分钟',
                ha='center', va='bottom')
    
    plt.title('不同调度方案运行时间对比')
    plt.ylabel('运行时间（分钟）')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_results(morning_time, afternoon_time, free_time, uncertainty_results):
    """格式化输出结果"""
    print("\n=== 调度优化结果 ===")
    print(f"早高峰最短时间: {morning_time:.2f} 分钟")
    print(f"下课高峰最短时间: {afternoon_time:.2f} 分钟")
    print(f"自由调配最短时间: {free_time:.2f} 分钟")
    
    print("\n=== 不确定性分析 ===")
    print("情景分析结果：")
    for result in uncertainty_results:
        print(f"需求{result['scenario']}：")
        print(f"  运输时间: {result['time']:.2f} 分钟")
        print(f"  变化率: {result['change']:+.2f}%")


def main():
    stations, morning_demands, afternoon_demands = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    # 1. 早高峰调度
    assign_buses_to_routes(buses, stations, G, demand_type="morning")
    time1 = simulate_transport(buses, stations, G)

    # 2. 下课高峰调度
    assign_buses_to_routes(buses, stations, G, demand_type="afternoon")
    time2 = simulate_transport(buses, stations, G)

    # 3. 车辆自由调配优化
    time3 = optimize_free_allocation(buses, stations, G)

    # 4. 不确定性分析
    uncertainty_results = uncertainty_analysis(buses, stations, G)

    # 输出结果
    print_results(time1, time2, time3, uncertainty_results)

    # 可视化路线图
    visualize_routes(G)


if __name__ == "__main__":
    main()