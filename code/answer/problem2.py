from common import *
# 改为从 common 导入 get_station_mapping
from common import get_station_mapping

def main():
    stations, _, afternoon_demands = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    valid_stations = set(G.nodes())
    station_dict = {s.name: s for s in stations}
    mapping = get_station_mapping()  # 获取站点名称映射关系
    all_stations = []
    for name in valid_stations:
        if name in station_dict:
            station = station_dict[name]
            if name in afternoon_demands:
                # 转换需求中的起点和目的站点名称到校车网络中的名称
                new_demand = {}
                for k, v in afternoon_demands[name].items():
                    mapped_k = mapping.get(k.strip(), k.strip())
                    if mapped_k in valid_stations:
                        new_demand[mapped_k] = v
                station.demand_dict = new_demand
            else:
                station.demand_dict = {}
            all_stations.append(station)
        else:
            all_stations.append(Station(name, {}))
    stations = all_stations

    assign_buses_to_routes(buses, stations, G, demand_type="afternoon")
    time, remaining = simulate_transport(buses, stations, G)  # 解构返回值
    if remaining > 0:
        print(f"\n警告：还有 {remaining} 人未被运送！")
    print(f"\n第二题最优运输时间: {time:.2f} 分钟")

if __name__ == "__main__":
    main()
