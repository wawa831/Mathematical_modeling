from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    assign_buses_to_routes(buses, stations, G, demand_type="morning")
    time = simulate_transport(buses, stations, G)
    print(f"\n第一题最优运输时间: {time:.2f} 分钟")

if __name__ == "__main__":
    main()