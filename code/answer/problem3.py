from common import *

def get_station_mapping():
    """获取站点名称映射关系"""
    return {
        "八教(李园)": "8教",
        "八教": "8教",
        "九教(芸文楼)": "荟文楼",
        "九教": "荟文楼",
        "国际学院": "外办",
        "二十七教": "27教",
        "二号门": "2号门",
        "三十教": "32教",
        "二十五教": "26教",  # 假设25教就是26教
        "经济管理学院": "经管院"
    }

def filter_stations(stations, demand_dict, valid_stations):
    """确保所有校车网络节点都被初始化，且需求字典正确映射"""
    filtered = []
    station_mapping = get_station_mapping()
    
    # 所有校车网络节点都要初始化一个Station对象
    for name in valid_stations:
        new_station = Station(name, {})
        filtered.append(new_station)
    
    # 处理需求
    for name, demands in demand_dict.items():
        # 找到对应的校车网络节点（使用映射表）
        source_name = station_mapping.get(name, name)
        matched_station = next((s for s in filtered if s.name == source_name), None)
        
        if matched_station:
            # 处理目的地的需求映射
            for dest, count in demands.items():
                # 使用映射表处理目的地
                dest_name = station_mapping.get(dest, dest)
                if dest_name in valid_stations:
                    matched_station.demand_dict[dest_name] = count

    return filtered

def check_demand_coverage(demands, valid_stations):
    """检查需求覆盖情况，使用映射表"""
    station_mapping = get_station_mapping()
    for source, dest_dict in demands.items():
        source_name = station_mapping.get(source, source)
        if source_name not in valid_stations:
            print(f"警告: 需求起点 {source}({source_name}) 未匹配到校车网络节点")
        for dest in dest_dict:
            dest_name = station_mapping.get(dest, dest)
            if dest_name not in valid_stations:
                print(f"警告: 需求终点 {dest}({dest_name}) 未匹配到校车网络节点")

def main():
    stations, morning_demands, afternoon_demands = load_station_data()
    G = load_routes()
    valid_stations = set(G.nodes())

    print("Valid stations:", valid_stations)
    
    # 检查站点覆盖
    def check_demand_coverage(demands, valid_stations):
        for source, dest_dict in demands.items():
            matched_source = next((v for v in valid_stations if source in v or v in source), None)
            if not matched_source:
                print(f"警告: 需求起点 {source} 未匹配到校车网络节点")
            for dest in dest_dict:
                matched_dest = next((v for v in valid_stations if dest in v or v in dest), None)
                if not matched_dest:
                    print(f"警告: 需求终点 {dest} 未匹配到校车网络节点")

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