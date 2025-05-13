# visualize_routes.py

from common import *

def main():
    # 加载路线图
    G = load_routes()

    # 可视化路线图
    visualize_routes(G)

if __name__ == "__main__":
    main()