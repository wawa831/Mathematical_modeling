# common.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

class Bus:
    
    def __init__(self, bus_id, route_id, capacity, bus_type):
        self.bus_id = bus_id
        self.route_id = route_id
        self.capacity = capacity
        self.bus_type = bus_type  # 普通/高峰/支线
        self.location = None
        self.passengers = 0

    def copy(self):
        """创建当前 Bus 实例的深拷贝"""
        new_bus = Bus(
        self.bus_id,
        self.route_id,
        self.capacity,
        self.bus_type
        )
        new_bus.location = self.location
        new_bus.passengers = self.passengers
        return new_bus


class Station:
    def __init__(self, name, demand_dict):
        self.name = name
        self.demand_dict = demand_dict  # {目标站点: 人数}


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

    all_stations = set()
    for source, demands in morning_demands.items():
        all_stations.add(source)
        all_stations.update(demands.keys())
    for source, demands in afternoon_demands.items():
        all_stations.add(source)
        all_stations.update(demands.keys())

    stations = []
    for station_name in all_stations:
        demands = {}
        if station_name in morning_demands:
            demands = morning_demands[station_name]
        stations.append(Station(station_name, demands))

    return stations, morning_demands, afternoon_demands


def load_bus_data():
    buses = []
    bus_configs = [
        (1, 3, 13, "普通"),
        (2, 2, 13, "普通"),
        (3, 5, 13, "普通"),
        (4, 8, 26, "普通"),
        (5, 6, 13, "普通"),
        (6, 5, 26, "普通"),
        (7, 6, 26, "普通"),
        (8, 4, 13, "普通"),
        (9, 2, 13, "支线"),
        (10, 4, 13, "高峰车"),   # 高峰车1
        (11, 6, 26, "高峰车"),   # 高峰车2
    ]

    bus_id = 1
    for route_id, count, capacity, bus_type in bus_configs:
        for _ in range(count):
            # 普通车只能跑自己编号路线
            if bus_type == "普通":
                buses.append(Bus(bus_id, route_id, capacity, bus_type))
            # 支线车只能跑支线
            elif bus_type == "支线":
                buses.append(Bus(bus_id, 9, capacity, bus_type))
            # 高峰车初始不分配路线，后续优化分配
            elif bus_type == "高峰车":
                buses.append(Bus(bus_id, None, capacity, bus_type))
            bus_id += 1
    return buses


def load_routes():
    G = nx.DiGraph()
    # 所有涉及的站点
    stations = [
        "经管院", "资环院", "32教", "共青团花园", "楠园", "校史馆", "大礼堂", "研究生院",
        "中心图书馆(地科院)", "8教", "行署楼1号门", "田家炳", "文学院", "圆顶", "外办", "5号门",
        "校医院", "西师街", "后山竹园", "地科院", "心理学部", "外国语学院", "药学院", "梅园",
        "梅园食堂", "橘园", "桃园", "荟文楼", "橘园12舍", "橘园10舍", "26教", "梅园1舍", "音乐学院", "2号门", "共青团花园2号门"
    ]
    G.add_nodes_from(stations)

    routes = [
        ("经管院", "资环院", 0.370, 1),
        ("资环院", "32教", 0.383, 1),
        ("32教", "共青团花园", 0.272, 1),
        ("共青团花园", "楠园", 0.315, 1),
        ("楠园", "校史馆", 0.239, 1),
        ("校史馆", "大礼堂", 0.163, 1),
        ("大礼堂", "研究生院", 0.123, 1),
        ("研究生院", "中心图书馆(地科院)", 0.222, 1),
        ("中心图书馆(地科院)", "8教", 0.310, 1),
        ("8教", "行署楼1号门", 0.179, 1),
        ("行署楼1号门", "田家炳", 0.201, 1),
        ("田家炳", "文学院", 0.111, 1),
        ("文学院", "圆顶", 0.224, 1),
        ("圆顶", "外办", 0.204, 1),
        ("外办", "5号门", 0.230, 1),

        ("经管院", "资环院", 0.370, 2),
        ("资环院", "32教", 0.383, 2),
        ("32教", "共青团花园", 0.272, 2),
        ("共青团花园", "楠园", 0.315, 2),
        ("楠园", "校史馆", 0.239, 2),
        ("校史馆", "大礼堂", 0.163, 2),
        ("大礼堂", "研究生院", 0.123, 2),
        ("研究生院", "中心图书馆(地科院)", 0.222, 2),
        ("中心图书馆(地科院)", "8教", 0.310, 2),
        ("8教", "行署楼1号门", 0.179, 2),
        ("行署楼1号门", "田家炳", 0.201, 2),
        ("田家炳", "校医院", 0.249, 2),
        ("校医院", "西师街", 0.118, 2),
        ("西师街", "5号门", 0.455, 2),

        ("后山竹园", "32教", 0.466, 3),
        ("32教", "共青团花园", 0.272, 3),
        ("共青团花园", "楠园", 0.315, 3),
        ("楠园", "校史馆", 0.239, 3),
        ("校史馆", "大礼堂", 0.163, 3),
        ("大礼堂", "研究生院", 0.123, 3),
        ("研究生院", "中心图书馆(地科院)", 0.222, 3),
        ("中心图书馆(地科院)", "地科院", 0.074, 3),
        ("地科院", "心理学部", 0.260, 3),
        ("心理学部", "外国语学院", 0.099, 3),
        ("外国语学院", "药学院", 0.144, 3),
        ("药学院", "梅园", 0.157, 3),
        ("梅园", "梅园食堂", 0.155, 3),
        ("梅园食堂", "橘园", 0.358, 3),
        ("橘园", "桃园", 0.256, 3),
        ("桃园", "荟文楼", 0.092, 3),
        ("荟文楼", "圆顶", 0.309, 3),
        ("圆顶", "外办", 0.147, 3),
        ("外办", "5号门", 0.228, 3),
        ("橘园", "田家炳文学院", 0.390, 3),
        ("田家炳文学院", "圆顶", 0.320, 3),
        ("2号门", "共青团花园2号门", 0.121, 3),

        # 4号线
        ("2号门", "共青团花园2号门", 0.121, 4),
        ("共青团花园", "楠园", 0.315, 4),
        ("楠园", "校史馆", 0.239, 4),
        ("校史馆", "大礼堂", 0.163, 4),
        ("大礼堂", "研究生院", 0.123, 4),
        ("研究生院", "中心图书馆(地科院)", 0.222, 4),
        ("中心图书馆(地科院)", "地科院", 0.074, 4),
        ("地科院", "心理学部", 0.260, 4),
        ("心理学部", "外国语学院", 0.099, 4),
        ("外国语学院", "药学院", 0.144, 4),
        ("药学院", "梅园", 0.157, 4),
        ("梅园", "梅园食堂", 0.155, 4),
        ("梅园食堂", "橘园12舍", 0.172, 4),
        ("橘园12舍", "橘园", 0.186, 4),

        # 5号线
        ("2号门", "共青团花园2号门", 0.121, 5),
        ("共青团花园", "楠园", 0.315, 5),
        ("楠园", "校史馆", 0.239, 5),
        ("校史馆", "大礼堂", 0.163, 5),
        ("大礼堂", "研究生院", 0.123, 5),
        ("研究生院", "中心图书馆(地科院)", 0.222, 5),
        ("中心图书馆(地科院)", "8教", 0.310, 5),
        ("8教", "行署楼1号门", 0.179, 5),
        ("行署楼1号门", "田家炳", 0.201, 5),
        ("田家炳", "圆顶", 0.335, 5),
        ("圆顶", "外办", 0.204, 5),
        ("外办", "5号门", 0.230, 5),

        # 6号线
        ("2号门", "共青团花园2号门", 0.121, 6),
        ("后山竹园", "32教", 0.466, 6),
        ("32教", "共青团花园", 0.272, 6),
        ("共青团花园", "楠园", 0.315, 6),
        ("楠园", "校史馆", 0.239, 6),
        ("校史馆", "大礼堂", 0.163, 6),
        ("大礼堂", "研究生院", 0.123, 6),
        ("研究生院", "中心图书馆(地科院)", 0.222, 6),
        ("中心图书馆(地科院)", "地科院", 0.074, 6),
        ("地科院", "心理学部", 0.260, 6),
        ("心理学部", "外国语学院", 0.099, 6),
        ("外国语学院", "26教", 0.214, 6),
        ("26教", "梅园1舍", 0.133, 6),
        ("梅园1舍", "橘园12舍", 0.213, 6),
        ("橘园12舍", "橘园10舍", 0.086, 6),
        ("橘园10舍", "橘园", 0.137, 6),
        ("橘园", "桃园", 0.256, 6),
        ("桃园", "文学院", 0.282, 6),
        ("文学院", "田家炳", 0.108, 6),
        ("田家炳", "行署楼1号门", 0.201, 6),
        ("行署楼1号门", "8教", 0.179, 6),
        ("8教", "地科院", 0.236, 6),

        # 7号线
        ("2号门", "楠园", 0.436, 7),
        ("楠园", "校史馆", 0.239, 7),
        ("校史馆", "大礼堂", 0.163, 7),
        ("大礼堂", "研究生院", 0.123, 7),
        ("研究生院", "中心图书馆(地科院)", 0.222, 7),
        ("中心图书馆(地科院)", "地科院", 0.074, 7),
        ("地科院", "心理学部", 0.260, 7),
        ("心理学部", "外国语学院", 0.099, 7),
        ("外国语学院", "26教", 0.214, 7),
        ("26教", "梅园1舍", 0.133, 7),
        ("梅园1舍", "橘园12舍", 0.213, 7),
        ("橘园12舍", "橘园10舍", 0.086, 7),
        ("橘园10舍", "橘园", 0.137, 7),
        ("橘园", "桃园", 0.256, 7),
        ("桃园", "荟文楼", 0.092, 7),
        ("荟文楼", "文学院", 0.190, 7),
        ("文学院", "田家炳", 0.108, 7),
        ("田家炳", "行署楼1号门", 0.201, 7),
        ("行署楼1号门", "8教", 0.179, 7),
        ("8教", "地科院", 0.236, 7),

        # 8号线
        ("地科院", "心理学部", 0.260, 8),
        ("心理学部", "外国语学院", 0.099, 8),
        ("外国语学院", "26教", 0.214, 8),
        ("26教", "梅园1舍", 0.133, 8),
        ("梅园1舍", "橘园12舍", 0.213, 8),
        ("橘园12舍", "橘园10舍", 0.086, 8),
        ("橘园10舍", "橘园", 0.137, 8),
        ("橘园", "桃园", 0.256, 8),
        ("桃园", "荟文楼", 0.092, 8),
        ("荟文楼", "文学院", 0.190, 8),
        ("文学院", "田家炳", 0.108, 8),
        ("田家炳", "行署楼1号门", 0.201, 8),
        ("行署楼1号门", "8教", 0.179, 8),
        ("8教", "地科院", 0.236, 8),
        ("音乐学院", "8教", 0.181, 8),

        # 经管专线（支线）
        ("经管院", "资环院", 0.370, 9),
        ("资环院", "32教", 0.383, 9),
        ("32教", "共青团花园", 0.272, 9),
        ("共青团花园", "楠园", 0.315, 9),
    ]

    for start, end, dist, route in routes:
        G.add_edge(start, end, route=route, distance=dist)
        G.add_edge(end, start, route=route, distance=dist)  # 添加反向边

    return G

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
                continue
    return route_demands


def allocate_buses(buses, route_demands):
    """
    - 普通车只能跑自己编号的路线
    - 支线车只能跑支线（9号线）
    - 高峰车可以任意分配到1-9号线（可用优化算法分配）
    """
    normal_buses = [bus for bus in buses if bus.bus_type == "普通"]
    branch_buses = [bus for bus in buses if bus.bus_type == "支线"]
    peak_buses = [bus for bus in buses if bus.bus_type == "高峰车"]

    # 普通车分配到自己编号的路线
    for bus in normal_buses:
        # 只能跑自己编号路线
        pass  # bus.route_id 已在生成时指定，不可更改

    # 支线车分配到支线（9号线）
    for bus in branch_buses:
        bus.route_id = 9

    # 高峰车分配到需求最大的路线（可优化，这里简单分配）
    sorted_routes = sorted(route_demands.items(), key=lambda x: x[1], reverse=True)
    route_ids = [r[0] for r in sorted_routes] if sorted_routes else [1,2,3,4,5,6,7,8,9]
    for i, bus in enumerate(peak_buses):
        bus.route_id = route_ids[i % len(route_ids)]

    return buses

def assign_buses_to_routes(buses, stations, G, demand_type="morning"):
    route_demands = calculate_route_demands(stations, G)
    return allocate_buses(buses, route_demands)



# ...existing code...
def simulate_transport(buses, stations, G, speed=15, load_time=3):
    """
    校车运输过程仿真（离散事件模拟）
    
    参数:
      buses: 校车列表，每个 bus 包含 route_id、capacity、passengers、location 等属性
      stations: 站点列表，每个 station 包含 name 和 demand_dict，其结构为 {目的站: 人数}
      G: 校车网络图，各边属性中必须包含 'distance' 与 'route'
      speed: 校车行驶速度（km/h），默认15
      load_time: 每人上/下车时间（秒），默认3
    
    返回:
      (total_time, remaining) 总运输时间（分钟）和剩余未运输人数
    """
    total_time = 0.0
    # 初始化各站点的剩余需求
    remaining_passengers = {station.name: station.demand_dict.copy() for station in stations}

    # 当任一站点仍有需求时继续循环
    while any(sum(d.values()) for d in remaining_passengers.values()):
        # 每轮中，让所有车辆按其对应线路依次运行一轮
        for bus in buses:
            # 获取当前车辆的路线边（假设G中边顺序即为线路经过顺序）
            route_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('route') == bus.route_id]
            if not route_edges:
                continue

            # 设置车辆初始位置为路线第一个站点
            bus.location = route_edges[0][0]
            bus_round_time = 0.0

            # 按顺序遍历每一段边
            for start, end in route_edges:
                # 计算行驶时间（单位：分钟）
                distance = G[start][end]['distance']
                t_run = (distance / speed) * 60

                # 模拟下车（此处简化处理，下车时间与上车类似，不计入车辆内人数变化）
                # 可根据需要增加下车逻辑

                # 模拟上车：
                # 获取起点 start 的总等待人数（所有目的站需求合计）
                waiting = sum(remaining_passengers.get(start, {}).values())
                if waiting > 0:
                    # 车辆可上车人数 = 剩余容量（capacity - 当前车上人数）
                    can_board = max(bus.capacity - bus.passengers, 0)
                    boarding = min(waiting, can_board)
                    t_board = (boarding * load_time) / 60   # 转为分钟

                    # 更新车辆上乘客数量
                    bus.passengers += boarding

                    # 简单扣减起点需求：遍历所有目的站，直到 boarding 数量分配完毕
                    remaining = boarding
                    for dest in list(remaining_passengers[start].keys()):
                        req = remaining_passengers[start][dest]
                        if req <= remaining:
                            remaining -= req
                            del remaining_passengers[start][dest]
                        else:
                            remaining_passengers[start][dest] -= remaining
                            remaining = 0
                        if remaining == 0:
                            break
                else:
                    t_board = 0

                # 累加当前边的总时间
                seg_time = t_run + t_board
                bus_round_time += seg_time

                # 将车辆移动到下一个站点
                bus.location = end

            # 每辆车完成一轮后的运行时间累加到总时间
            total_time += bus_round_time

    # 计算剩余未满足的需求
    remaining = sum(sum(d.values()) for d in remaining_passengers.values())
    return total_time, remaining
# ...existing code...

def init_genetic_algorithm(buses, route_count, population_size):
    population = []
    for _ in range(population_size):
        solution = []
        peak_buses = [bus for bus in buses if bus.bus_type == "高峰车"]
        normal_buses = [bus for bus in buses if bus.bus_type != "高峰车"]

        peak_solution = [random.randint(1, route_count) for _ in peak_buses]
        normal_solution = [random.randint(1, route_count) for _ in normal_buses]
        solution.extend(peak_solution)
        solution.extend(normal_solution)

        while not is_solution_valid(solution, route_count):
            solution = [random.randint(1, route_count) for _ in buses]

        population.append(solution)

    return population


def is_solution_valid(solution, route_count):
    covered_routes = set(solution)
    return len(covered_routes) == route_count


def fitness_function(solution, buses, stations, G):
    temp_buses = [b.copy() for b in buses]  # 使用新增的 .copy()
    for i, bus in enumerate(temp_buses):
        bus.route_id = solution[i]
    time = simulate_transport(temp_buses, stations, G)
    coverage = calculate_route_coverage(temp_buses, G)
    time_weight = 0.8
    coverage_weight = 0.2
    return -time * time_weight + coverage * coverage_weight


def calculate_route_coverage(buses, G):
    covered_routes = set(b.route_id for b in buses)
    return len(covered_routes) / 9


def optimize_bus_allocation(buses, stations, G, generations=50, population_size=100):
    route_count = 9
    bus_count = len(buses)

    # 初始化种群：所有车辆都可分配到1~9号线
    population = []
    for _ in range(population_size):
        solution = [random.randint(1, route_count) for _ in range(bus_count)]
        # 保证每条路线至少有一辆车
        for rid in range(1, route_count+1):
            if rid not in solution:
                solution[random.randint(0, bus_count-1)] = rid
        population.append(solution)

    def fitness(solution):
        temp_buses = [b.copy() for b in buses]
        for i, bus in enumerate(temp_buses):
            bus.route_id = solution[i]  # 所有车辆都可以自由分配路线
        time, remaining = simulate_transport(temp_buses, stations, G)
        if remaining > 0:
            return -1e6  # 极大惩罚未完成运输的方案
        return -time

    best_solution = None
    best_fitness = float('-inf')
    
    for generation in range(generations):
        fitness_scores = [fitness(sol) for sol in population]
        elite_size = population_size // 4
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i] for i in elite_indices]

        # 更新全局最优解
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_solution = population[current_best_idx]

        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            crossover_point = random.randint(0, bus_count - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            # 变异
            if random.random() < 0.1:
                mutation_point = random.randint(0, bus_count - 1)
                child[mutation_point] = random.randint(1, route_count)
            
            # 保证每条路线至少有一辆车
            for rid in range(1, route_count+1):
                if rid not in child:
                    child[random.randint(0, bus_count-1)] = rid
            new_population.append(child)
        
        population = new_population

    # 使用最优解更新车辆分配
    optimized_buses = [b.copy() for b in buses]
    for i, bus in enumerate(optimized_buses):
        bus.route_id = best_solution[i]
    time, _ = simulate_transport(optimized_buses, stations, G)
    
    return time, best_solution, optimized_buses
def evaluate_peak_bus_efficiency(buses):
    peak_buses = [b for b in buses if b.bus_type == "高峰车"]
    if not peak_buses:
        return 1.0

    route_count = {}
    for bus in peak_buses:
        route_count[bus.route_id] = route_count.get(bus.route_id, 0) + 1

    total_routes = len(set(route_count.keys()))
    avg_buses = len(peak_buses) / total_routes if total_routes else 0
    variance = sum((v - avg_buses)**2 for v in route_count.values()) / total_routes if total_routes else 0
    dispersion = 1 / (1 + variance)
    coverage = total_routes / 9
    return (dispersion * 0.6 + coverage * 0.4)


def uncertainty_analysis(buses, stations, G):
    variations = [
        (1.15, 1.05), (1.15, 0.95),
        (0.85, 1.05), (0.85, 0.95)
    ]
    base_time = simulate_transport(buses, stations, G)
    results = []

    for df, sf in variations:
        original = {s.name: s.demand_dict.copy() for s in stations}
        for s in stations:
            for k in s.demand_dict:
                s.demand_dict[k] = int(original[s.name][k] * df)

        time = simulate_transport(buses, stations, G, speed=15 * sf)
        results.append({
            'demand_change': f"{'增加' if df > 1 else '减少'}{abs(df-1)*100:.0f}%",
            'speed_change': f"{'增加' if sf > 1 else '减少'}{abs(sf-1)*100:.0f}%",
            'time': time,
            'change_rate': (time - base_time) / base_time * 100
        })

        for s in stations:
            s.demand_dict = original[s.name].copy()

    print("\n=== 不确定性分析 ===")
    for r in results:
        print(f"需求{r['demand_change']}，速度{r['speed_change']}")
        print(f"运输时间: {r['time']:.2f} 分钟, 变化率: {r['change_rate']:+.2f}%")

    return results


def visualize_routes(G):
    pos = {
        "竹园": (8, 9), "国际学院(外办)": (6, 9),
        "楠园": (7, 7), "外国语学院": (5, 7),
        "二十七教(梅园)": (6, 5), "八教(李园)": (4, 5),
        "橘园十一舍": (2, 4), "橘园食堂": (3, 4), "九教(芸文楼)": (5, 4),
        "桃园": (2, 2), "经管院": (7, 2)
    }

    route_styles = {
        1: {'color': '#E41A1C', 'name': '1号线'},
        2: {'color': '#377EB8', 'name': '2号线'},
        3: {'color': '#4DAF4A', 'name': '3号线'},
        4: {'color': "#914E9B", 'name': '4号线'},
        5: {'color': '#FF7F00', 'name': '5号线'},
        6: {'color': '#FFFF33', 'name': '6号线'},
        7: {'color': '#A65628', 'name': '7号线'},
        8: {'color': '#F781BF', 'name': '8号线'},
        9: {'color': '#999999', 'name': '支线'}
    }

    plt.figure(figsize=(24, 16))
    for route_id, style in route_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['route'] == route_id]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=style['color'], width=3, arrows=True)

    node_colors = ["#B2E1F7" if '教' in n else '#C8E6C9' if '园' in n else '#FFECB3' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_family='SimHei', font_size=12, font_weight='bold')

    plt.title("西南大学校车路线图", fontsize=20)
    plt.axis('off')
    plt.show()