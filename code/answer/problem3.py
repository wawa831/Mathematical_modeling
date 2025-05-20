from common import *


def get_station_mapping():
    """获取站点名称映射关系，将需求中的名称转换为校车网络图中的节点名称"""
    return {
        # 早高峰需求部分
        "竹园": "后山竹园",   # 修改映射，将“竹园”映射为网络中存在的“后山竹园”
        "八教(李园)": "8教",
        "八教": "8教",
        "九教(芸文楼)": "荟文楼",
        "九教": "荟文楼",
        "国际学院(外办)": "外办",
        "国际学院": "外办",
        "二十七教(梅园)": "梅园",
        "二十七教": "梅园",
        "橘园十一舍": "橘园",
        "橘园食堂": "梅园食堂",
        "经管院": "经管院",
        "经济管理学院": "经管院",  
        # 下课需求部分
        "三十教": "32教",
        "二十五教": "26教",
        "二号门": "2号门",
    }

def filter_stations(stations, demand_dict, valid_stations):
    """确保所有校车网络节点都被初始化，且需求字典正确映射"""
    filtered = []
    station_mapping = get_station_mapping()
    
    # 所有校车网络节点都初始化为一个 Station 对象
    for name in valid_stations:
        new_station = Station(name, {})
        filtered.append(new_station)
    
    # 处理需求：使用映射表对需求数据进行转换后再赋值
    for source, demands in demand_dict.items():
        source_clean = source.strip()  # 清理首尾空格
        source_name = station_mapping.get(source_clean, source_clean)
        matched_station = next((s for s in filtered if s.name == source_name), None)
        if matched_station:
            for dest, count in demands.items():
                dest_clean = dest.strip()
                dest_name = station_mapping.get(dest_clean, dest_clean)
                if dest_name in valid_stations:
                    matched_station.demand_dict[dest_name] = count
    return filtered

def check_demand_coverage(demands, valid_stations):
    """检查需求覆盖情况，使用映射表，同时清洗输入字符串"""
    station_mapping = get_station_mapping()
    for source, dest_dict in demands.items():
        source_clean = source.strip()
        source_name = station_mapping.get(source_clean, source_clean)
        if source_name not in valid_stations:
            print(f"警告: 需求起点 {source}({source_name}) 未匹配到校车网络节点")
        for dest in dest_dict:
            dest_clean = dest.strip()
            dest_name = station_mapping.get(dest_clean, dest_clean)
            if dest_name not in valid_stations:
                print(f"警告: 需求终点 {dest}({dest_name}) 未匹配到校车网络节点")

def optimize_bus_allocation(buses, stations, G, generations=50, population_size=100):
    route_count = 9
    bus_count = len(buses)
    
    # 初始化种群：生成每辆车可分配到1~9号线的随机解，并确保每个路线至少有一辆车
    population = []
    for _ in range(population_size):
        solution = [random.randint(1, route_count) for _ in range(bus_count)]
        for rid in range(1, route_count + 1):
            if rid not in solution:
                solution[random.randint(0, bus_count - 1)] = rid
        population.append(solution)
    
    def fitness(solution):
        # 复制车辆并分配方案
        temp_buses = [b.copy() for b in buses]
        for i, bus in enumerate(temp_buses):
            bus.route_id = solution[i]
        time, remaining = simulate_transport(temp_buses, stations, G)
        # 若还有未运输需求，则方案无效，适应度返回0；否则，目标是最小化时间，因此使用倒数转换
        if remaining > 0:
            return 0
        return 1 / (1 + time)
    
    best_solution = None
    best_fitness = -1
    for generation in range(generations):
        # 计算当前种群的适应度
        fitness_scores = [fitness(sol) for sol in population]
        # 记录当前最优解
        for i, score in enumerate(fitness_scores):
            if score > best_fitness:
                best_fitness = score
                best_solution = population[i]
        # 锦标赛选择构建新种群
        new_population = []
        while len(new_population) < population_size:
            s1 = random.choice(population)
            s2 = random.choice(population)
            parent = s1 if fitness(s1) > fitness(s2) else s2
            # 随机选取另一父本进行交叉
            other = random.choice(population)
            crossover_point = random.randint(1, bus_count - 1)
            child = parent[:crossover_point] + other[crossover_point:]
            # 增加较高的变异概率，尝试跳出局部最优
            if random.random() < 0.2:
                mutation_point = random.randint(0, bus_count - 1)
                child[mutation_point] = random.randint(1, route_count)
            # 确保每个路线至少出现一次
            for rid in range(1, route_count + 1):
                if rid not in child:
                    child[random.randint(0, bus_count - 1)] = rid
            new_population.append(child)
        population = new_population
        # 可选：打印每一代最差/最佳适应度，便于调试
        # print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    # 利用最终最优解更新车辆分配
    optimized_buses = [b.copy() for b in buses]
    for i, bus in enumerate(optimized_buses):
        bus.route_id = best_solution[i]
    total_time, _ = simulate_transport(optimized_buses, stations, G)
    return total_time, best_solution, optimized_buses

def main():
    stations, morning_demands, afternoon_demands = load_station_data()
    G = load_routes()
    valid_stations = set(G.nodes())

    print("Valid stations:", valid_stations)
    
    # 使用映射表检查需求覆盖情况
    print("\n=== 检查早高峰需求覆盖 ===")
    check_demand_coverage(morning_demands, valid_stations)
    print("\n=== 检查下课高峰需求覆盖 ===")
    check_demand_coverage(afternoon_demands, valid_stations)

    # 早高峰自由调配
    morning_stations = filter_stations(stations, morning_demands, valid_stations)
    buses_morning = load_bus_data()
    time_morning, solution_morning, optimized_buses_morning = optimize_bus_allocation(
        buses_morning, 
        morning_stations, 
        G,
        generations=20,  # 先用小参数调试
        population_size=50
    )
    print(f"\n第三题早高峰自由调配最优运输时间: {time_morning:.2f} 分钟")

    # 下课高峰自由调配
    afternoon_stations = filter_stations(stations, afternoon_demands, valid_stations)
    buses_afternoon = load_bus_data()
    time_afternoon, solution_afternoon, optimized_buses_afternoon = optimize_bus_allocation(
        buses_afternoon, 
        afternoon_stations, 
        G,
        generations=20,  # 先用小参数调试
        population_size=50
    )
    print(f"\n第三题下课高峰自由调配最优运输时间: {time_afternoon:.2f} 分钟")

    # 输出调度方案
    print("\n=== 第三题车辆最优调度方案（早高峰） ===")
    for bus in optimized_buses_morning:
        print(f"车辆 {bus.bus_id}: 路线 {bus.route_id}, 类型 {bus.bus_type}, 容量 {bus.capacity}")

    print("\n=== 第三题车辆最优调度方案（下课高峰） ===")
    for bus in optimized_buses_afternoon:
        print(f"车辆 {bus.bus_id}: 路线 {bus.route_id}, 类型 {bus.bus_type}, 容量 {bus.capacity}")

if __name__ == "__main__":
    main()
