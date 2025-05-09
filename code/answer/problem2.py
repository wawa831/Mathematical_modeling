from common import *

def main():
    stations, _, afternoon_demands = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    # 替换为下课需求
    for station in stations:
        if station.name in afternoon_demands:
            station.demand_dict = afternoon_demands[station.name]

    assign_buses_to_routes(buses, stations, G, demand_type="afternoon")
    time = simulate_transport(buses, stations, G)
    print(f"\n第二题最优运输时间: {time:.2f} 分钟")

    print("\n=== 第二题车辆调度安排 ===")
    for bus in buses:
        print(f"车辆 {bus.bus_id}: 路线 {bus.route_id}, 类型 {bus.bus_type}, 容量 {bus.capacity}")