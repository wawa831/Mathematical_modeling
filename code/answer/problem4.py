import copy
import random
from common import *

def uncertainty_analysis(buses, stations, G, num_scenarios=10, speed_variation=0.05, demand_variation=0.15):
    """
    对运行速率和需求量进行不确定性分析：
      - speed_variation: 运行速率允许±5%的变化
      - demand_variation: 候车需求允许±15%的变化
      - num_scenarios: 模拟情景数
    对比基准方案（无扰动）的运输时间，输出各场景下的结果及平均运输时间。
    """
    # 基准方案，采用默认速率 15km/h
    baseline_time, _ = simulate_transport(buses, stations, G)
    print(f"基准运输时间: {baseline_time:.2f} 分钟")
    
    scenario_times = []
    for i in range(num_scenarios):
        # 深拷贝数据以不破坏原始数据
        buses_copy = copy.deepcopy(buses)
        stations_copy = copy.deepcopy(stations)
        
        # 运行速率在15km/h基础上随机扰动±5%
        speed_factor = 1 + random.uniform(-speed_variation, speed_variation)
        current_speed = 15 * speed_factor
        
        # 对每个站点将需求乘以随机扰动因子±15%
        for st in stations_copy:
            new_demand = {}
            for dest, value in st.demand_dict.items():
                factor = 1 + random.uniform(-demand_variation, demand_variation)
                new_demand[dest] = value * factor
            st.demand_dict = new_demand
        
        time_i, remaining = simulate_transport(buses_copy, stations_copy, G, speed=current_speed)
        scenario_times.append(time_i)
        print(f"情景 {i+1}: 速率 = {current_speed:.2f} km/h, 运输时间 = {time_i:.2f} 分钟, 剩余需求 = {remaining}")
    
    avg_time = sum(scenario_times) / num_scenarios if scenario_times else 0
    print(f"所有情景平均运输时间: {avg_time:.2f} 分钟")
    return baseline_time, scenario_times, avg_time

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()
    
    # 此处建议使用 filter_stations 对需求做映射清洗（如果需要），确保各站点 demand_dict 正确
    # 例如：stations = filter_stations(stations, morning_demands, set(G.nodes()))
    
    # 调用不确定性分析
    uncertainty_analysis(buses, stations, G)

if __name__ == "__main__":
    main()