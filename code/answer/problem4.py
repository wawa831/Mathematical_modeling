from common import *

def main():
    stations, _, _ = load_station_data()
    buses = load_bus_data()
    G = load_routes()

    result = uncertainty_analysis(buses, stations, G)

if __name__ == "__main__":
    main()