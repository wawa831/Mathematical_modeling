from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    time, best_solution = optimize_bus_allocation(buses, stations, G)
    print(f"\n第三题自由调配最优运输时间: {time:.2f} 分钟")

    print("\n=== 第三题车辆最优调度方案 ===")
    for i, bus in enumerate(buses):
        print(f"车辆 {bus.bus_id}: 路线 {best_solution[i]}, 类型 {bus.bus_type}, 容量 {bus.capacity}")