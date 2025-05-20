from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    valid_stations = set(G.nodes())
    station_dict = {s.name: s for s in stations}
    all_stations = []
    for name in valid_stations:
        if name in station_dict:
            station = station_dict[name]
            station.demand_dict = {k: v for k, v in station.demand_dict.items() if k in valid_stations}
            all_stations.append(station)
        else:
            all_stations.append(Station(name, {}))
    stations = all_stations

    assign_buses_to_routes(buses, stations, G, demand_type="morning")
    time, remaining = simulate_transport(buses, stations, G)  # 解构返回值
    if remaining > 0:
        print(f"\n警告：还有 {remaining} 人未被运送！")
    print(f"\n第一题最优运输时间: {time:.2f} 分钟")

    print("\n=== 第一题车辆调度安排 ===")
    for bus in buses:
        print(f"车辆 {bus.bus_id}: 路线 {bus.route_id}, 类型 {bus.bus_type}, 容量 {bus.capacity}")

if __name__ == "__main__":
    main()
