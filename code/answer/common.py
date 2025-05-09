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
        (10, 2, 13, "高峰车")
    ]

    bus_id = 1
    for route_id, count, capacity, bus_type in bus_configs:
        for _ in range(count):
            buses.append(Bus(bus_id, route_id, capacity, bus_type))
            bus_id += 1

    return buses


def load_routes():
    G = nx.DiGraph()
    stations = [
        "竹园", "楠园", "经管院", "八教(李园)", "九教(芸文楼)",
        "国际学院(外办)", "外国语学院", "二十七教(梅园)",
        "橘园十一舍", "橘园食堂", "桃园"
    ]
    G.add_nodes_from(stations)

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
    peak_buses = [bus for bus in buses if bus.bus_type == "高峰车"]
    normal_buses = [bus for bus in buses if bus.bus_type != "高峰车"]

    sorted_routes = sorted(route_demands.items(), key=lambda x: x[1], reverse=True)

    for bus in peak_buses:
        if sorted_routes:
            bus.route_id = sorted_routes[0][0]
            sorted_routes.pop(0)

    for bus in normal_buses:
        if bus.bus_type != "支线":
            continue
        if sorted_routes:
            bus.route_id = sorted_routes[0][0]
            sorted_routes.pop(0)

    return buses


def assign_buses_to_routes(buses, stations, G, demand_type="morning"):
    route_demands = calculate_route_demands(stations, G)
    return allocate_buses(buses, route_demands)


def simulate_transport(buses, stations, G, speed=15, load_time=3):
    total_time = 0
    remaining_passengers = {s.name: s.demand_dict.copy() for s in stations}

    while any(bool(d) for d in remaining_passengers.values()):
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

                # 计算行驶时间
                if (start, end) in travel_times:
                    travel_time = travel_times[(start, end)]
                elif (start, end, "顺") in travel_times:
                    travel_time = travel_times[(start, end, "顺")]
                elif (start, end, "逆") in travel_times:
                    travel_time = travel_times[(start, end, "逆")]
                else:
                    distance = G[start][end]['distance']
                    congestion_factor = 1.2 if total_time < 60 else 1.0
                    travel_time = (distance / (speed / congestion_factor)) * 60

                # 上下车时间
                boarding_time = 0
                if remaining_passengers[start]:
                    boarding_time = min(
                        bus.capacity - current_passengers,
                        sum(remaining_passengers[start].values())
                    ) * random.uniform(2, 4) / 60
                    travel_time += boarding_time

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

    return total_time


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


def optimize_bus_allocation(buses, stations, G, generations=100, population_size=200):
    route_count = 9
    population = init_genetic_algorithm(buses, route_count, population_size)

    for generation in range(generations):
        fitness_scores = [fitness_function(sol, buses, stations, G) for sol in population]
        elite_size = population_size // 4
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            crossover_point = random.randint(0, len(buses) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            if random.random() < 0.1:
                mutation_point = random.randint(0, len(buses) - 1)
                child[mutation_point] = random.randint(1, route_count)

            if is_solution_valid(child, route_count):
                new_population.append(child)
            else:
                while not is_solution_valid(child, route_count):
                    mutation_point = random.randint(0, len(buses) - 1)
                    child[mutation_point] = random.randint(1, route_count)
                new_population.append(child)

        population = new_population

    best_solution = max(population, key=lambda x: fitness_function(x, buses, stations, G))
    for i, bus in enumerate(buses):
        bus.route_id = best_solution[i]
    time = simulate_transport(buses, stations, G)
    return time, best_solution


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
        4: {'color': '#984EA3', 'name': '4号线'},
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

    node_colors = ['#B3E5FC' if '教' in n else '#C8E6C9' if '园' in n else '#FFECB3' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_family='SimHei', font_size=12, font_weight='bold')

    plt.title("西南大学校车路线图", fontsize=20)
    plt.axis('off')
    plt.show()