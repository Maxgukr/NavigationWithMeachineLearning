import numpy as np
import matplotlib.pyplot as plt
import math

def loaddata(path):

    east_pos  = []   # 经度
    north_pos = []  # 维度
    height    = []  # 高程
    east_v    = []   # 东向速度
    north_v   = []  # 北向速度
    groud_v   = []  # 地向速度
    roll      = []  # 横滚角
    pitch     = []  # 俯仰角
    yaw       = []  # 航向角
    with open(path,'r') as file:
        for line in file:
            # 读取每一行的数据
            lines = line.split()
            # 存储位置信息
            north_pos.append(float(lines[2]))
            east_pos.append(float(lines[3]))
            height.append(float(lines[4]))
            # 存储速度信息
            east_v.append(float(lines[5]))
            north_v.append(float(lines[6]))
            groud_v.append(float(lines[7]))
            # 存储角度信息
            roll.append(float(lines[8]))
            pitch.append(float(lines[9]))
            yaw.append(float(lines[10]))
    
    file.close()

    pos = np.array([north_pos, east_pos, height], dtype=float)
    v   = np.array([north_v, east_v, groud_v], dtype=float)
    ang = np.array([roll,pitch,yaw], dtype=float)

    return pos,v,ang


def loadGPS(path):

    gps_time = []
    gps_latitude = []
    gps_longitude = []
    gps_height = []
    gps_v_north = []
    gps_v_east = []
    gps_v_groud = []

    with open(path,'r') as file:
        for line in file:
            lines = line.split()
            gps_time.append(lines[0])
            gps_latitude.append(lines[1])
            gps_longitude.append(lines[2])
            gps_height.append(lines[3])
            gps_v_east.append(lines[4])
            gps_v_north.append(lines[5])
            gps_v_groud.append(lines[6])

    file.close()
    gps_pos = np.array([gps_latitude, gps_longitude, gps_height], dtype=float)
    gps_v   = np.array([gps_latitude, gps_longitude, gps_height],dtype=float)
    gps_time= np.array(gps_time, dtype=float)

    return gps_pos, gps_v, gps_time


def deg(X):
    PI = math.pi
    X = (X / 180) * PI
    return X


def Translate(B, L, H, B0, L0, H0):
    # B为纬度，L为经度，H为高度(补偿高度)
    '''

    :param B:
    :param L:
    :param H: 其他坐标
    :param B0:
    :param L0:
    :param H0: 参考点坐标
    :return:
    '''
    a = 6378137
    # =6356755.00
    e = 0.016710219
    #print("GPS下经度、纬度、高度为", L, B, H)
    B = deg(B)
    L = deg(L)
    B0 = deg(B0)
    L0 = deg(L0)

    N = a / (math.sqrt(1 - e * e * math.sin(B) * math.sin(B)))  # 曲率半径
    X = (N + H) * math.cos(B) * math.cos(L)  # 空间直角坐标系X轴
    Y = (N + H) * math.cos(B) * math.sin(L)  # 空间直角坐标系Y轴
    Z = N * (1 - e * e) * math.sin(B)  # 空间直角坐标系Z轴

    N0 = a / (math.sqrt(1 - e * e * math.sin(B0) * math.sin(B0)))
    X0 = (N0 + H0) * math.cos(B0) * math.cos(L0)
    Y0 = (N0 + H0) * math.cos(B0) * math.sin(L0)
    Z0 = N0 * (1 - e * e) * math.sin(B0)
    #print("空间直角坐标系下X轴、Y轴、高度为", '%.3f' % X, '%.3f' % Y, '%.3f' % Z)

    mat = np.array([[-math.sin(L), math.cos(L), 0],
                    [-math.sin(B) * math.cos(L), -math.sin(B) * math.sin(L), math.cos(B)],
                    [math.cos(B) * math.cos(L), math.cos(B) * math.sin(L), math.sin(B)]])
    arr = np.array([[X - X0], [Y - Y0], [Z - Z0]])  # 计算和参考点的相对位置
    res = np.dot(mat, arr)

    #print("站心坐标系下东偏向、北偏向", '%.3f' % X2, '%.3f' % Y2)

    return res


def convertCoordinate(pos):
    '''
    坐标系转换
    :param pos:
    :return:
    '''
    ned = np.zeros((len(pos[0]),3),dtype=float)
    for i in range(len(pos[0])):
        ned[i] = Translate(pos[0][i], pos[1][i], pos[2][i], pos[0][0], pos[1][0], pos[2][0]).squeeze()

    return ned


def getNEUdata():
    # 低精度的惯导得到的数据
    path_init = "./data/HL20180824054627_0RTNAV_TXT.nav"  # 低精度惯导数据
    pos_init, v_init, ang_init = loaddata(path_init)
    ned_init = convertCoordinate(pos_init)

    # 高精度的惯导得到的数据
    path_heigh = "./data/posD_imu_heigh_quality.nav"
    pos_heigh_quality, v_heigh_quality, ang_heigh_quality = loaddata(path_heigh)
    ned_imu = convertCoordinate(pos_heigh_quality)

    ned_heigh = ned_imu[442:-1]  # 启示时刻对齐
    ned_heigh = ned_heigh - ned_heigh[0]

    ned_init = ned_init[0:len(ned_heigh)]

    ned_heigh = ned_heigh.transpose()
    ned_init = ned_init.transpose()

    #  加载GPS数据
    path_gps = "./data/HL20180824054627_0GNSS.txt"
    gps_pos, gps_v, gps_time = loadGPS(path_gps)
    ned_gps = convertCoordinate(gps_pos)
    ned_gps = ned_gps[493:-1] - ned_gps[0]
    gps_time = gps_time[493:-1] - gps_time[493]
    time_x = [0.0001 for i in range(len(gps_time))]

    #  绘制位置误差曲线 东向位置
    err_pos_x = np.abs(ned_init[0] - ned_heigh[0])
    plt.subplot(211)
    plt.title("east position error")
    plt.ylabel("east position error(m)")
    plt.xlabel("time(s)")
    plt.plot(err_pos_x, 'r-', label='position error')
    plt.plot(gps_time, time_x, 'k.', markersize=2, label='GPS signal')
    plt.legend()
    #plt.show()

    #  绘制位置误差曲线 北向位置
    err_pos_y = np.abs(ned_init[1] - ned_heigh[1])
    plt.subplot(212)
    plt.title("north position error")
    plt.ylabel("east position error(m)")
    plt.xlabel("time(s)")
    plt.plot(err_pos_y, 'r-', label='position error')
    plt.plot(gps_time, time_x, 'k.', markersize=2, label='GPS signal')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ned_init, ned_gps
