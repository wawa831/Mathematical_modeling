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

if __name__ == "__main__":
    main()