from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    time, _ = optimize_bus_allocation(buses, stations, G)
    print(f"\n第三题自由调配最优运输时间: {time:.2f} 分钟")

if __name__ == "__main__":
    main()