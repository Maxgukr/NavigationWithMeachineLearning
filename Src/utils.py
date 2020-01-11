import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from Earth import Earth
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
    ned = np.zeros((len(pos[0]),3), dtype=float)
    for i in range(len(pos[0])):
        ned[i] = Translate(pos[0][i], pos[1][i], pos[2][i], pos[0][0], pos[1][0], pos[2][0]).squeeze()
    ned = ned.T
    return ned

def generate_data_from_csv(path):
    """
    从数据中合成文件
    :param path_raw_imu:
    :param path_inspvaxa:
    :return:
    """
    # 加载GPS的数据
    df_gps = pd.read_csv(path[0], header=None,
                         delimiter=' ', dtype=float)
    df_gps = df_gps[[0, 1, 2, 3]]
    df_gps.columns = ['time', 'lat', 'lon', 'height']
    df_gps = df_gps.loc[df_gps.time >= 199230.0]
    df_gps = df_gps.loc[df_gps.time < 202831.0]
    df_gps.drop(['time'], axis=1, inplace=True)
    df_gps.reset_index(drop=True, inplace=True)
    # 加载imu的数据
    df_imu = pd.read_csv(path[1], header=None, names=['time', 'a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z'],
                         delimiter=' ', dtype=float)
    df_imu = df_imu.loc[df_imu.time >= 199230.0]
    df_imu = df_imu.loc[df_imu.time <= 202830.0]
    # df_imu.drop(['time'], axis=1, inplace=True)
    df_imu.reset_index(drop=True, inplace=True)
    # 加载组合导航输出的数据
    df_ins = pd.read_csv(path[2], header=None, names=['time', 'lat', 'lon', 'height', 'v_x', 'v_y', 'v_z', 'roll', 'pitch','yaw'],
                         delimiter=' ', dtype=float)
    df_ins = df_ins.loc[df_ins.time >=199230.0]
    df_ins = df_ins.loc[df_ins.time <=202830.0]
    df_ins.drop(['time'], axis=1, inplace=True)
    df_ins.reset_index(drop=True, inplace=True)
    df_ins = df_ins[['v_x', 'v_y', 'v_z', 'roll', 'pitch','yaw', 'lat', 'lon', 'height']]

    # 训练用数据
    df_data = pd.concat([df_imu, df_ins], axis=1)
    # 训练用标签
    df_label = df_gps.diff(-1)
    # 将经纬高对应的数据做差得到增量
    df_label.drop(index=df_gps.index[len(df_gps)-1], inplace=True)

    return df_data, df_label

def get_gps_test_true():
    path_gps = "../data/M39/M39_20190710.gps"
    df_gps = pd.read_csv(path_gps, header=None, delimiter=' ', dtype=float)
    df_gps = df_gps[[0, 1, 2, 3]]
    # df_gps[[2,1,3]].to_csv("../data/M39/gps_traj.csv", index=False, header=False)
    df_gps.columns = ['time', 'lat', 'lon', 'height']
    df_gps = df_gps.loc[df_gps.time >= 199230.0]
    df_gps = df_gps.loc[df_gps.time < 202831.0]
    # df_gps_test = df_gps.loc[df_gps.time >=201750.0]
    # df_gps_test[['lon','lat', 'height']].to_csv("../data/M39/gps_traj_test.csv", index=False, header=False)
    df_gps.drop(['time'], axis=1, inplace=True)
    df_gps.reset_index(drop=True, inplace=True)
    i_split = math.ceil(len(df_gps) * 0.7)
    df_gps_test_true = df_gps[i_split:]
    df_gps_test_true.reset_index(drop=True, inplace=True)
    return df_gps_test_true.get(['lat', 'lon', 'height']).values

def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * np.pi) + 20.0 *
            math.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * np.pi) + 40.0 *
            math.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * np.pi) + 320 *
            math.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
    return ret

def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * np.pi) + 20.0 *
            math.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * np.pi) + 40.0 *
            math.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * np.pi) + 300.0 *
            math.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
    return ret

def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    earth = Earth()
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = math.sin(radlat)
    magic = 1 - earth.e2 * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((earth.a * (1 - earth.e2)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (earth.a / sqrtmagic * math.cos(radlat) * np.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]

def get_train_test_traj():

    path_gps = "../data/M39-09-20/M39_20190920.gps"
    data_gps = np.loadtxt(path_gps, dtype=float, delimiter=' ', usecols=[0,1,2,3])
    i = np.argsort([0, 2, 1, 3])
    data_gps = data_gps[:, i]
    df_gps = pd.DataFrame(data_gps, columns=['time', 'lon', 'lat', 'he'])
    df_gps_train = df_gps.loc[df_gps.time>=469600.0]
    df_gps_train = df_gps_train.loc[df_gps_train.time < 473600].get_values()

    df_gps_test = df_gps.loc[df_gps.time>=474000.0]
    df_gps_test = df_gps_test.loc[df_gps_test.time<475400.0].get_values()

    df_gps_test_60 = df_gps.loc[df_gps.time>=475330.0]
    df_gps_test_60 = df_gps_test_60.loc[df_gps.time < 475390.0].get_values()

    df_gps_test_120 = df_gps.loc[df_gps.time >= 474510.0]
    df_gps_test_120 = df_gps_test_120.loc[df_gps.time < 474630.0].get_values()

    df_gps_test_180 = df_gps.loc[df_gps.time >= 474320.0]
    df_gps_test_180 = df_gps_test_180.loc[df_gps.time < 474500.0].get_values()

    '''
    # 坐标系转换
    for i in range(len(df_gps_test)):
        df_gps_test[i, 1:3] = wgs84_to_gcj02(df_gps_test[i, 1], df_gps_test[i, 2])
    for i in range(len(df_gps_train)):
        df_gps_train[i, 1:3] = wgs84_to_gcj02(df_gps_train[i, 1], df_gps_train[i, 2])
    for i in range(len(df_gps_test_60)):
        df_gps_test_60[i, 1:3] = wgs84_to_gcj02(df_gps_test_60[i, 1], df_gps_test_60[i, 2])
    for i in range(len(df_gps_test_120)):
        df_gps_test_120[i, 1:3] = wgs84_to_gcj02(df_gps_test_120[i, 1], df_gps_test_120[i, 2])
    for i in range(len(df_gps_test_180)):
        df_gps_test_180[i, 1:3] = wgs84_to_gcj02(df_gps_test_180[i, 1], df_gps_test_180[i, 2])
    '''
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=9)  # 设精度为3
    np.savetxt('../data/M39-09-20/m39_9_20-test.csv', df_gps_test[:, 1:], delimiter=',', fmt='%.09f')
    np.savetxt('../data/M39-09-20/m39_9_20-train.csv', df_gps_train[:, 1:], delimiter=',', fmt='%.09f')
    np.savetxt('../data/M39-09-20/m39_9_20-test_60.csv', df_gps_test_60[:, 1:], delimiter=',', fmt='%.09f')
    np.savetxt('../data/M39-09-20/m39_9_20-test_120.csv', df_gps_test_120[:, 1:], delimiter=',', fmt='%.09f')
    np.savetxt('../data/M39-09-20/m39_9_20-test_180.csv', df_gps_test_180[:, 1:], delimiter=',', fmt='%.09f')
    print("")

def plot_history(history):
    """
    plot result of val_loss, loss, learn rate
    :param history:
    :return:
    """
    plt.plot(history.epoch, history.history['loss'], label="loss")
    plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.plot(history.epoch, history.history['val_mean_squared_error'], label="val_mse")
    #plt.plot(history.epoch, history.history['mean_squared_error'], label="mse")
    plt.legend()
    plt.savefig("../result/loss.png", dpi=300)


if __name__ == "__main__":

    get_train_test_traj()
