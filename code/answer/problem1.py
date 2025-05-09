from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    assign_buses_to_routes(buses, stations, G, demand_type="morning")
    time = simulate_transport(buses, stations, G)
    print(f"\n第一题最优运输时间: {time:.2f} 分钟")

    print("\n=== 第一题车辆调度安排 ===")
    for bus in buses:
        print(f"车辆 {bus.bus_id}: 路线 {bus.route_id}, 类型 {bus.bus_type}, 容量 {bus.capacity}")