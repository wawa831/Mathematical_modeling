西南大学第十九届数学建模校赛 B 题
校车的优化方案
西南大学的校车问题一直受到广大学生的关注，现如今，西南大学开通了 8
条编号观光车，1 条支线，以及部分高峰旅游观光车，但仍在早晚高峰，下课高
峰期存在诸多站点等不到车的情况，据此，西南大学数学建模协会拟思考一个方
案，分析如何让校车的运转尽可能地满足学生的需求。
1. 西南大学的观光车数目拟定如下：
编号 数量 单车最大人数
1 3 13
2 2 13
3 5 13、
4 8 26
5 6 13
6 5 26
7 6 26
8 4 13
支线 2 13
高峰车1 4 13
高峰车2 6 26
各校车的行驶路线详见校园地图，高峰车可在任意线路行驶，此时视为该车
型，高峰车在转换线路时将会有 30s 的等待时间。必须完整经过该车型的所有站
点才可重新选择所需线路，重新选择的线路后，高峰车到达最近的该线路的站点
后沿该线路运行。
在早高峰期，各站点需求量如下：
站点
7：30 候车人数（后续不
增加）
拟到站点（每个站点人数
比例视为相同）
竹园 600 国际学院，八教，九教，
二十七教，外国语学院
楠园 600 国际学院，八教，九教，
二十七教，经济管理学院
梅园 400 八教，九教
八教（李园） 500 九教，二十七教
橘园十一舍 200 八教，外国语学院
橘园食堂 400 八教，二十七教，外国语
学院
桃园 300 八教，二十七教，外国语
学院
默认校车行驶速度为 15km/h，每上车/下车一个人需要 3s，请给出一个最优
方案，使得校车在最短时间内将所有人运输到相应站点。


2. 在下课时期，各站点需求量如下：
站点
9：45 候车人数（后续不
增加）
拟到站点（每个站点人数
比例视为相同）
国际学院 100 竹园
二十七教 350
竹园，橘园，楠园，桃园，
二号门
九教 170 八教，梅园，楠园
八教（李园） 350 楠园，竹园
外国语学院 150 楠园，竹园，桃园，橘园
经管院 80 楠园
30 教 100 楠园，二十五教
请给出一个最优方案，使得校车在最短时间内将所有人运输到相应站点。


3. 现在车辆数目和每辆车的荷载人数不变，你可以自由调配每辆车所属的路线，
请给出一个最优方案，在早晨七点时调配校车，使得校车在 1.2.问的条件下可
以在最短时间内将所有人运输到相应站点，并给出最短时间。


4. 受天气影响，在阴雨等天气下，候车人数可能会有±15%的变化，校车的运行
速率可能会发生±5%的变化，请综合考虑运行速率，候车人数等影响，给出校车
最优的调配方案，并与第三问的结果做比较分析，说明各影响因子对决策的影响。