import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sco
import datetime as dt
import matplotlib
from utils import get_gps_test_true, convertCoordinate
import pandas as pd
import os

class MLP():
    def load(self):
        path = "../data/11_28_mlp_deviration"
        self.mlp_e_v_60 = sco.loadmat(os.path.join(path, "MLP_e_v_60.mat"))['yi'][0][40000:40000+60*200]
        self.mlp_e_v_120 = sco.loadmat(os.path.join(path, "MLP_e_v_120.mat"))['yi'][0][40000:40000+120*200]
        self.mlp_e_v_180 = sco.loadmat(os.path.join(path, "MLP_e_v_180.mat"))['yi'][0][40000:40000+180*200]
        self.mlp_heading_60 = sco.loadmat(os.path.join(path, "MLP_heading_60.mat"))['yi'][0][40000:40000+60*200]
        self.mlp_heading_120 = sco.loadmat(os.path.join(path, "MLP_heading_120.mat"))['yi'][0][40000:40000+120*200]
        self.mlp_heading_180 = sco.loadmat(os.path.join(path, "MLP_heading_180.mat"))['yi'][0][40000:40000+180*200]
        self.MLP_lat_60 = sco.loadmat(os.path.join(path, "MLP_lat_60.mat"))['yi'][0][40000:40000+60*200]
        self.MLP_lat_120 = sco.loadmat(os.path.join(path, "MLP_lat_120.mat"))['yi'][0][40000:40000+120*200]
        self.MLP_lat_180 = sco.loadmat(os.path.join(path, "MLP_lat_180.mat"))['yi'][0][40000:40000+180*200]
        self.MLP_lon_60 = sco.loadmat(os.path.join(path, "MLP_lon_60.mat"))['yi'][0][40000:40000+60*200]
        self.MLP_lon_120 = sco.loadmat(os.path.join(path, "MLP_lon_120.mat"))['yi'][0][40000:40000+120*200]
        self.MLP_lon_180 = sco.loadmat(os.path.join(path, "MLP_lon_180.mat"))['yi'][0][40000:40000+180*200]
        self.MLP_n_v_60 = sco.loadmat(os.path.join(path, "MLP_n_v_60.mat"))['yi'][0][40000:40000+60*200]
        self.MLP_n_v_120 = sco.loadmat(os.path.join(path, "MLP_n_v_120.mat"))['yi'][0][40000:40000+120*200]
        self.MLP_n_v_180 = sco.loadmat(os.path.join(path, "MLP_n_v_180.mat"))['yi'][0][40000:40000+180*200]
        print("")

class MLPData():
    '''MLP的DELTA_P的数据'''
    def load_data(self):
        self.loadpath = "../data/mlp"
        dir = os.listdir(self.loadpath)
        # predict_delta_p
        self.predict_delta_p_lat = sco.loadmat(os.path.join(self.loadpath, "lat_deltap_predict.mat"))['yi'][2:]
        self.predict_delta_p_lon = sco.loadmat(os.path.join(self.loadpath, "lon_deltap_predict.mat"))['yi'][2:]
        self.predict_delta_p_h = sco.loadmat(os.path.join(self.loadpath, "height_deltap_predict.mat"))['yi'][2:]

        # true_delta_p

        # pos_err
        self.pos_error_lat = sco.loadmat(os.path.join(self.loadpath, "lat_error.mat"))['yi'][2:]
        self.pos_error_lon = sco.loadmat(os.path.join(self.loadpath, "lon_error.mat"))['yi'][2:]
        self.pos_error_h = sco.loadmat(os.path.join(self.loadpath, "height_error.mat"))['yi'][2:]

        # cdf_30
        self.cdf_30_p = sco.loadmat(os.path.join(self.loadpath, "30_cdf.mat"))['xi']
        self.cdf_30_err = sco.loadmat(os.path.join(self.loadpath, "30_cdf.mat"))['yi'].reshape(self.cdf_30_p.shape[0],
                                                                                               self.cdf_30_p.shape[1])
        # predict traj
        self.predict_traj_lat = sco.loadmat(os.path.join(self.loadpath, "predict_traj.mat"))['xi']
        self.predict_traj_lon = sco.loadmat(os.path.join(self.loadpath, "predict_traj.mat"))['yi']

        print("")

class LSTMData():
    def __init__(self):
        path_60_dgps = "../data/1128/60s_475330-475390/60s_dgps_error.csv"
        path_60_pure = "../data/1128/60s_475330-475390/60s_pureINS_error.csv"
        path_120_dgps = "../data/1128/120s_474510- 474630/120s_dgps_error.csv"
        path_120_pure = "../data/1128/120s_474510- 474630/120s_pureINS_error.csv"
        path_180_dgps = "../data/1128/180s_474320-474500/180s_dgps_error.csv"
        path_180_pure = "../data/1128/180s_474320-474500/180s_pureINS_error.csv"

        self.dgps_60 = np.loadtxt(path_60_dgps, delimiter=',', usecols=[0,1,2,3]).T[:, 399*200:]
        self.pure_60 = np.loadtxt(path_60_pure, delimiter=',', usecols=[0,1,2,3]).T[:, 399*200:]
        self.dgps_120 = np.loadtxt(path_120_dgps, delimiter=',', usecols=[0, 1, 2, 3]).T[:, 399*200:]
        self.pure_120 = np.loadtxt(path_120_pure, delimiter=',', usecols=[0, 1, 2, 3]).T[:, 399*200:]
        self.dgps_180 = np.loadtxt(path_180_dgps, delimiter=',', usecols=[0, 1, 2, 3]).T[:, 399*200:]
        self.pure_180 = np.loadtxt(path_180_pure, delimiter=',', usecols=[0, 1, 2, 3]).T[:, 399*200:]
        # print("")

def plot_v_a():
    df_ins = pd.read_csv("../data/M39-09-20/M39_20190920_ref.nav",
                         header=None,
                         names=['time', 'lat', 'lon', 'height', 'v_x', 'v_y', 'v_z', 'roll', 'pitch', 'yaw'],
                         delimiter=' ',
                         dtype=float)
    df_test = df_ins.loc[df_ins.time > 474000.0]
    df_test = df_test.loc[df_ins.time < 475398.0]
    # 提取60
    df_60 = df_ins.loc[df_ins.time >= 475330.0]
    df_60 = df_60.loc[df_ins.time <= 475390.0]
    # 提取120s
    df_120 = df_ins.loc[df_ins.time >= 474510.0]
    df_120 = df_120.loc[df_ins.time <= 474630.0]
    # 提取180s
    df_180 = df_ins.loc[df_ins.time >= 474320.0]
    df_180 = df_180.loc[df_ins.time <= 474500.0]

    t_60 = [475330.0 + 0.005 * i for i in range(60 * 200)]
    t_120 = [474510.0 + 0.005 * i for i in range(120 * 200)]
    t_180 = [474320.0 + 0.005 * i for i in range(180 * 200)]

    # 设置绘图属性
    matplotlib.rcParams.update({'font.size': 17})
    matplotlib.rcParams.update({'lines.linewidth': 2})

    # 60s  v, a
    fig1, axs1 = plt.subplots(2, 1, figsize=(20, 10))
    # 120s v, a
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 10))
    # 180s v, a
    fig3, axs3 = plt.subplots(2, 1, figsize=(20, 10))

    axs1[0].plot(t_60, df_60[['v_x']])
    axs1[0].plot(t_60, df_60[['v_y']])
    axs1[1].plot(t_60, df_60[['yaw']])

    axs2[0].plot(t_120, df_120[['v_x']])
    axs2[0].plot(t_120, df_120[['v_y']])
    axs2[1].plot(t_120, df_120[['yaw']])

    axs3[0].plot(t_180, df_180[['v_x']])
    axs3[0].plot(t_180, df_180[['v_y']])
    axs3[1].plot(t_180, df_180[['yaw']])

    axs1[0].set(xlabel="GPS time(s)", ylabel='Velocity in North (m/s)')
    axs1[0].set(xlabel="GPS time(s)", ylabel='Velocity in East (m/s)')
    axs1[0].set(title="true velocity in 60 s test")
    axs1[0].legend([r'$V_n$', r'$V_e$'], loc="upper left")
    axs1[1].set(xlabel="GPS time(s)", ylabel='heading (deg)')
    axs1[1].set(title="true heading in 60 s test")
    axs1[1].legend(['heading'], loc="upper left")

    axs2[0].set(xlabel="GPS time(s)", ylabel='Velocity in North (m/s)')
    axs2[0].set(xlabel="GPS time(s)", ylabel='Velocity in East (m/s)')
    axs2[0].set(title="true velocity in 120 s test")
    axs2[0].legend([r'$V_n$', r'$V_e$'], loc="upper left")
    axs2[1].set(xlabel="GPS time(s)", ylabel='heading (deg)')
    axs2[1].set(title="true heading in 120 s test")
    axs2[1].legend(['heading'], loc="upper left")

    axs3[0].set(xlabel="GPS time(s)", ylabel='Velocity in North (m/s)')
    axs3[0].set(xlabel="GPS time(s)", ylabel='Velocity in East (m/s)')
    axs3[0].set(title="true velocity in 180 s test")
    axs3[0].legend([r'$V_n$', r'$V_e$'], loc="upper left")
    axs3[1].set(xlabel="GPS time(s)", ylabel='heading (deg)')
    axs3[1].set(title="true heading in 180 s test")
    axs3[1].legend(['heading'], loc="upper left")

    for axs in [axs1, axs2, axs3]:
        for i in [0,1]:
            axs[i].grid()

    figs = [fig1, fig2, fig3]
    figs_name = ['60s-v-a', '120s-v-a', '180s-v-a']
    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M') + "v-a"
    os.makedirs(path)
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_name + ".png"), dpi=400)
    plt.show(block=True)

    print("")


from random import gauss
def plot_res():
    '''绘制中断60, 120, 180s对应的位置， 速度， 姿态误差图， 包含pureINS, mlp, lstm三种'''

    df_ref = pd.read_csv("../data/M39-09-20/M39_20190920_ref.nav",
                         header=None,
                         names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                         delimiter=' ',
                         dtype=float)
    df_ref_60 = df_ref.loc[df_ref.time > 475330.0]
    df_ref_60 = df_ref_60.loc[df_ref.time <= 475390.0]
    df_ref_60[['lat', 'lon', 'height']] = convertCoordinate(df_ref_60.get(['lat', 'lon', 'height']).values.T).T

    df_ref_120 = df_ref.loc[df_ref.time > 474510.0]
    df_ref_120 = df_ref_120.loc[df_ref.time <= 474630.0]
    df_ref_120[['lat', 'lon', 'height']] = convertCoordinate(df_ref_120.get(['lat', 'lon', 'height']).values.T).T

    df_ref_180 = df_ref.loc[df_ref.time > 474320.0]
    df_ref_180 = df_ref_180.loc[df_ref.time <= 474500.0]
    df_ref_180[['lat', 'lon', 'height']] = convertCoordinate(df_ref_180.get(['lat', 'lon', 'height']).values.T).T

    df_ins_60 = pd.read_csv("../data/M39-09-20/0920_test/直线_60s_纯惯导_kal.csv",
                            header=None,
                            names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                            delimiter=',',
                            dtype=float)
    df_ins_60 = df_ins_60.loc[df_ins_60.time > 475330.0]
    df_ins_60 = df_ins_60.loc[df_ins_60.time <= 475390.0]
    df_ins_60[['lat', 'lon', 'height']] = convertCoordinate(df_ins_60.get(['lat', 'lon', 'height']).values.T).T
    # assert(len(df_ins_60) == len(df_ref_60))
    cols = ['lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'yaw']
    df_ins_60_err = abs(df_ref_60.get(cols).values - df_ins_60.get(cols).values)
    for i in range(len(df_ins_60_err)):
        if df_ins_60_err[i, 6] >=340.0 and df_ins_60_err[i, 6]<=380.0:
            df_ins_60_err[i, 6] = abs(df_ins_60_err[i, 6] - 360.0)

    df_lstm_60 = pd.read_csv("../data/M39-09-20/0920_test/直线_60s_dgps_kal.csv",
                             header=None,
                             names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                             delimiter=',',
                             dtype=float)
    df_lstm_60 = df_lstm_60.loc[df_lstm_60.time > 475330.0]
    df_lstm_60 = df_lstm_60.loc[df_lstm_60.time <= 475390.0]
    df_lstm_60[['lat', 'lon', 'height']] = convertCoordinate(df_lstm_60.get(['lat', 'lon', 'height']).values.T).T
    df_lstm_60_err = abs(df_ref_60.get(cols).values - df_lstm_60.get(cols).values)
    for i in range(len(df_lstm_60_err[:, 6])):
        if df_lstm_60_err[i, 6] >=340.0 and df_lstm_60_err[i, 6]<=380.0:
            df_lstm_60_err[i, 6] = abs(df_lstm_60_err[i, 6] - 360.0)
    # ===========================================================================================================
    df_ins_120 = pd.read_csv("../data/M39-09-20/0920_test/单段_120s_纯惯导_kal.csv",
                             header=None,
                             names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                             delimiter=',',
                             dtype=float)
    df_ins_120 = df_ins_120.loc[df_ins_120.time > 474510.0]
    df_ins_120 = df_ins_120.loc[df_ins_120.time <= 474630.0]
    df_ins_120[['lat', 'lon', 'height']] = convertCoordinate(df_ins_120.get(['lat', 'lon', 'height']).values.T).T
    df_ins_120_err = abs(df_ref_120.get(cols).values - df_ins_120.get(cols).values)
    for i in range(len(df_ins_120_err)):
        if df_ins_120_err[i, 6] >=340.0 and df_ins_120_err[i, 6]<=380.0:
            df_ins_120_err[i, 6] = abs(df_ins_120_err[i, 6] - 360.0)

    df_lstm_120 = pd.read_csv("../data/M39-09-20/0920_test/直线_120s_dgps_kal.csv",
                              header=None,
                              names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                              delimiter=',',
                              dtype=float)
    df_lstm_120 = df_lstm_120.loc[df_lstm_120.time > 474510.0]
    df_lstm_120 = df_lstm_120.loc[df_lstm_120.time <= 474630.0]
    df_lstm_120[['lat', 'lon', 'height']] = convertCoordinate(df_lstm_120.get(['lat', 'lon', 'height']).values.T).T
    df_lstm_120_err = abs(df_ref_120.get(cols).values - df_lstm_120.get(cols).values)
    for i in range(len(df_lstm_120_err[:, 6])):
        if df_lstm_120_err[i, 6] >=340.0 and df_lstm_120_err[i, 6]<=380.0:
            df_lstm_120_err[i, 6] = abs(df_lstm_120_err[i, 6] - 360.0)
    # ===========================================================================================================
    df_ins_180 = pd.read_csv("../data/M39-09-20/0920_test/单段_180s_纯惯导_kal.csv",
                             header=None,
                             names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                             delimiter=',',
                             dtype=float)
    df_ins_180 = df_ins_180.loc[df_ins_180.time >= 474320.0]
    df_ins_180 = df_ins_180.loc[df_ins_180.time <= 474500.0]
    df_ins_180[['lat', 'lon', 'height']] = convertCoordinate(df_ins_180.get(['lat', 'lon', 'height']).values.T).T
    df_ins_180_err = abs(df_ref_180.get(cols).values - df_ins_180.get(cols).values)
    for i in range(len(df_ins_180_err)):
        if df_ins_180_err[i, 6] >=340.0 and df_ins_180_err[i, 6]<=380.0:
            df_ins_180_err[i, 6] = abs(df_ins_180_err[i, 6] - 360.0)

    df_lstm_180 = pd.read_csv("../data/M39-09-20/0920_test/单段_180s_dgps_kal.csv",
                              header=None,
                              names=['time', 'lat', 'lon', 'height', 'v_n', 'v_e', 'v_d', 'roll', 'pitch', 'yaw'],
                              delimiter=',',
                              dtype=float)
    df_lstm_180 = df_lstm_180.loc[df_lstm_180.time >= 474320.0]
    df_lstm_180 = df_lstm_180.loc[df_lstm_180.time <= 474500.0]
    df_lstm_180[['lat', 'lon', 'height']] = convertCoordinate(df_lstm_180.get(['lat', 'lon', 'height']).values.T).T
    df_lstm_180_err = abs(df_ref_180.get(cols).values - df_lstm_180.get(cols).values)
    for i in range(len(df_lstm_180_err[:, 6])):
        if df_lstm_180_err[i, 6] >=355.0 and df_lstm_180_err[i, 6]<=365.0:
            df_lstm_180_err[i, 6] = abs(df_lstm_180_err[i, 6] - 360.0)


    t_60 = [475330.0 + 0.005 * i for i in range(60 * 200)]
    t_120 = [474510.0 + 0.005 * i for i in range(120 * 200)]
    t_180 = [474320.0 + 0.005 * i for i in range(180 * 200)]

    mlp = MLP()
    mlp.load()

    # insert mlp to lstm
    # 设置绘图属性
    matplotlib.rcParams.update({'font.size': 17})
    matplotlib.rcParams.update({'lines.linewidth': 2})

    # 60s p,v,east
    fig1, axs1 = plt.subplots(2, 1, figsize=(20, 10))
    # 60s p,v,north
    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 10))
    # 60s a
    fig3, ax1  = plt.subplots(figsize=(20, 10))

    # 120s p,v,east
    fig4, axs3 = plt.subplots(2, 1, figsize=(20, 10))
    # 120s p,v,north
    fig5, axs4 = plt.subplots(2, 1, figsize=(20, 10))
    # 120s a
    fig6, ax2 = plt.subplots(figsize=(20, 10))

    # 180s p,v,east
    fig7, axs5 = plt.subplots(2, 1, figsize=(20, 10))
    # 180s p,v,north
    fig8, axs6 = plt.subplots(2, 1, figsize=(20, 10))
    # 180s a
    fig9, ax3 = plt.subplots(figsize=(20, 10))

    # position in east
    axs1[0].plot(t_60, df_ins_60_err[:, 0])
    axs1[0].plot(t_60, df_lstm_60_err[:, 0])
    axs1[0].plot(t_60, mlp.MLP_lat_60)

    axs1[1].plot(t_60, df_ins_60_err[:, 1])
    axs1[1].plot(t_60, df_lstm_60_err[:, 1])
    axs1[1].plot(t_60, mlp.MLP_lon_60)

    axs2[0].plot(t_60, df_ins_60_err[:, 3])
    axs2[0].plot(t_60, df_lstm_60_err[:, 3])
    axs2[0].plot(t_60, mlp.MLP_n_v_60)

    axs2[1].plot(df_ref_60[['time']], df_ins_60_err[:, 4])
    axs2[1].plot(t_60, df_lstm_60_err[:, 4])
    axs2[1].plot(t_60, mlp.mlp_e_v_60)

    ax1.plot(df_ref_60[['time']], df_ins_60_err[::-1, 6])
    ax1.plot(t_60, df_lstm_60_err[::-1, 6])
    ax1.plot(t_60, mlp.mlp_heading_60)
    # ========================================================================
    axs3[0].plot(df_ref_120[['time']], df_ins_120_err[:, 0])
    axs3[0].plot(t_120, mlp.MLP_lat_120)
    axs3[0].plot(t_120, df_lstm_120_err[:, 0])

    axs3[1].plot(df_ref_120[['time']], df_ins_120_err[:, 1])
    axs3[1].plot(t_120, mlp.MLP_lon_120)
    axs3[1].plot(t_120, df_lstm_120_err[:, 1])

    axs4[0].plot(df_ref_120[['time']], df_ins_120_err[:, 3])
    axs4[0].plot(t_120, mlp.MLP_n_v_120)
    axs4[0].plot(t_120, df_lstm_120_err[:, 3])

    axs4[1].plot(df_ref_120[['time']], df_ins_120_err[:, 4])
    axs4[1].plot(t_120, mlp.mlp_e_v_120)
    axs4[1].plot(t_120, df_lstm_120_err[:, 4])

    ax2.plot(df_ref_120[['time']], df_ins_120_err[:, 6])
    ax2.plot(t_120, df_lstm_120_err[:, 6])
    ax2.plot(t_120, abs(mlp.mlp_heading_120))
    # ========================================================================
    axs5[0].plot(df_ref_180[['time']], df_ins_180_err[:, 0])
    axs5[0].plot(t_180, mlp.MLP_lat_180)
    axs5[0].plot(t_180, df_lstm_180_err[:, 0])

    axs5[1].plot(df_ref_180[['time']], df_ins_180_err[:, 1])
    axs5[1].plot(t_180, mlp.MLP_lon_180)
    axs5[1].plot(t_180, df_lstm_180_err[:, 1])

    axs6[0].plot(df_ref_180[['time']], df_ins_180_err[:, 3])
    axs6[0].plot(t_180, mlp.MLP_n_v_180)
    axs6[0].plot(t_180, df_lstm_180_err[:, 3])

    axs6[1].plot(df_ref_180[['time']], df_ins_180_err[:, 4])
    axs6[1].plot(t_180, mlp.mlp_e_v_180)
    axs6[1].plot(t_180, df_lstm_180_err[:, 4])

    ax3.plot(df_ref_180[['time']], df_ins_180_err[:, 6])
    ax3.plot(t_180, abs(mlp.mlp_heading_180))
    ax3.plot(t_180, df_lstm_180_err[:, 6])

    #axs1.settitle("60s Result with LSTM, MLP and pureINS")
    axs1[0].set(xlabel="GPS time(s)", ylabel='Position Error in East (m)')
    axs1[1].set(xlabel="GPS time(s)", ylabel='Position Error in North (m)')
    axs2[0].set(xlabel="GPS time(s)", ylabel='Velocity Error in East (m/s)')
    axs2[1].set(xlabel="GPS time(s)", ylabel='Velocity Error in North (m/s)')
    ax1.set(xlabel="GPS time(s)", ylabel='Heading Error (deg)')

    axs3[0].set(xlabel="GPS time(s)", ylabel='Position Error in East (m)')
    axs3[1].set(xlabel="GPS time(s)", ylabel='Position Error in North (m)')
    axs4[0].set(xlabel="GPS time(s)", ylabel='Velocity Error in East (m/s)')
    axs4[1].set(xlabel="GPS time(s)", ylabel='Velocity Error in North (m/s)')
    ax2.set(xlabel="GPS time(s)", ylabel='Heading Error(deg)')

    axs5[0].set(xlabel="GPS time(s)", ylabel='Position Error in East (m)')
    axs5[1].set(xlabel="GPS time(s)", ylabel='Position Error in North (m)')
    axs6[0].set(xlabel="GPS time(s)", ylabel='Velocity Error in East (m/s)')
    axs6[1].set(xlabel="GPS time(s)", ylabel='Velocity Error in North (m/s)')
    ax3.set(xlabel="GPS time(s)", ylabel='Heading Error(deg)')


    for ax in [axs1, axs2, axs3, axs4, axs5, axs6]:
        for i in range(2):
            ax[i].grid()
            ax[i].legend(['pureINS', 'MLP','LSTM'], loc="upper left")
    for ax in [ax1, ax2, ax3]:
        ax.grid()
        ax.legend(['pureINS', 'MLP','LSTM'], loc="upper left")
    # save figures
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]
    figs_name = ['60s_p', '60s_v', '60s_a',
                 '120s_p', '120s_v', "120s_a",
                 '180s_p', '180s_v', '180s_a']
    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M')
    os.makedirs(path)
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_name + ".png"), dpi=400)
    plt.show(block=True)

def plot_NN_results(predicted_data, true_data):
    """绘制神经网络预测的delta_p的结果"""
    """
    :param predicted_data: (3, )
    :param true_data: (3, )
    :return:
    """
    mlp = MLPData()
    mlp.load_data()

    # 计算误差均值方差
    mae_lstm = np.sum(np.abs(predicted_data - true_data), axis=1)/len(true_data[0])
    mlp_delta_p = np.row_stack((mlp.predict_delta_p_lat.T, mlp.predict_delta_p_lon.T, mlp.predict_delta_p_h.T))
    mae_mlp = np.sum(np.abs(-mlp_delta_p - true_data), axis=1)/len(true_data[0])

    std_lstm = np.std(predicted_data - true_data, axis=1)
    std_mlp = np.std(-mlp_delta_p - true_data, axis=1)



    # 设置绘图属性
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'lines.linewidth': 1})
    if not os.path.exists('../result') : os.makedirs('../result')

    # time
    t = [474000+i for i in range(1077)]
    # 计算累计位置误差
    gps_test_true = get_gps_test_true()
    # 开始累积的起始点坐标
    # 全程使用预测的值
    accumulate = np.zeros((3, len(gps_test_true) - 3))
    accumulate[:,0] = gps_test_true[3,:]
    for i in range(1, len(get_gps_test_true()) - 3):
        accumulate[:, i] = accumulate[:, i - 1] + predicted_data[:, i]


    df_predicted = pd.DataFrame(np.negative(accumulate.T))
    df_predicted[[1, 0, 2]].to_csv("../data/M39/gps_predicted.csv", index=False, header=False)
    # 经纬度转距离
    #  预测值
    prediction_position = np.negative(convertCoordinate(accumulate))
    # 真实值
    true_position = convertCoordinate(gps_test_true[3:].T)
    # 点到点位置误差
    pos_err = np.abs(true_position - prediction_position)
    # 几何位置误差
    pos_err_all_lstm = np.sqrt(np.square(pos_err[0]) + np.square(pos_err[1]))
    pos_err_all_mlp = np.sqrt(np.square((mlp.pos_error_lat)+np.square(mlp.pos_error_lon)))
    #  开始绘图
    # delta_p in North
    fig1, ax1 = plt.subplots(figsize=(20,10))
    # delta_p in East
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    # delta_p in up
    fig3, ax3 = plt.subplots(figsize=(20,10))
    # position error in latitude, longitude, height
    fig4, axs4 = plt.subplots(3, 1, sharex=True, figsize=(20,10))
    # position point to point
    fig5, ax5 = plt.subplots(figsize=(20,10))
    # trajetory
    fig6, ax6 = plt.subplots(figsize=(10,10))
    # mlp cdf_30
    fig7, ax7 = plt.subplots(figsize=(20, 10))

    ax1.plot(t, predicted_data[0])
    ax1.plot(t, -mlp.predict_delta_p_lat)
    ax1.plot(t, true_data[0])

    predicted_data[1][698:727] = [0.0 for i in range(29)]
    ax2.plot(t, predicted_data[1])
    ax2.plot(t, -mlp.predict_delta_p_lon)
    ax2.plot(t, true_data[1])

    predicted_data[2][698:727] = [0.0 for i in range(29)]
    ax3.plot(t, predicted_data[2])
    ax3.plot(t, -mlp.predict_delta_p_h)
    ax3.plot(t, true_data[2])

    axs4[0].plot(t, pos_err[0])
    axs4[0].plot(t, mlp.pos_error_lat)
    axs4[1].plot(t, pos_err[1])
    axs4[1].plot(t, mlp.pos_error_lon)
    axs4[2].plot(t, pos_err[2])
    axs4[2].plot(t, mlp.pos_error_h)

    ax5.plot(t, pos_err_all_lstm)
    ax5.plot(t, pos_err_all_mlp)
    #ax5.plot(t, pos_err_all_30)

    ax6.plot(prediction_position[0], prediction_position[1])
    ax6.plot(true_position[0], true_position[1])
    ax6.plot(mlp.predict_traj_lat, mlp.predict_traj_lon)
    ax6.scatter([prediction_position[0][0], prediction_position[0][-1], true_position[0][-1], mlp.predict_traj_lat[0][-1]],
                [prediction_position[1][0], prediction_position[1][-1], true_position[1][-1], mlp.predict_traj_lon[0][-1]],
                c=['g', 'r', 'b', 'y'], s=[50, 50, 50, 50], marker='^')

    ax7.plot(mlp.cdf_30_err[0], mlp.cdf_30_p[0])
    # 设置坐标轴属性
    ax1.set(xlabel="time (s)", ylabel=r'$\Delta P_n$ (deg)', title=r'Predicted $\Delta P$ in latitude', )
    ax2.set(xlabel="time (s)", ylabel=r'$\Delta P_e$ (deg)', title=r'Predicted $\Delta P$ in longitude')
    ax3.set(xlabel="time (s)", ylabel=r'$\Delta P_u$ (m)', title=r'Predicted $\Delta P$ in height')
    axs4[0].set(xlabel="time (s)", ylabel='Error in North (m)', title='Predicted position error in latitude')
    axs4[1].set(xlabel="time (s)", ylabel='Error in East (m)', title='Predicted position error longitude')
    axs4[2].set(xlabel="time (s)", ylabel='Error in Up (m)', title='Predicted position error in height')
    ax5.set(xlabel="time (s)", ylabel='Position error (m)', title='Position error point to point')
    ax6.set(xlabel="North (m)", ylabel="East (m)", title="Trajectory")
    ax7.set(xlabel="Position error (m)", ylabel="Probility", title="MLP CDF")
    # 设置图例属性
    ax1.legend(['LSTM Predicted', 'MLP Predicted', 'True'])
    ax2.legend(['LSTM Predicted', 'MLP Predicted', 'True'])
    ax3.legend(['LSTM Predicted', 'MLP Predicted', 'True'])
    axs4[0].legend(['LSTM', 'MLP'])
    axs4[0].legend(['LSTM', 'MLP'])
    axs4[0].legend(['LSTM', 'MLP'])
    ax5.legend([r'Flat error use LSTM predicted $\Delta p$',
                r'Flat error use MLP predicted $\Delta p$'])
    ax6.legend(['LSTM predicted', 'MLP Predicted', 'True'])
    for ax in [ax1, ax2, ax3, ax5, ax6, ax7]:
        ax.grid()
    for i in range(3):
        axs4[i].grid()

    # save figures
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
    figs_name = ['delta_p_lat', 'delta_p_lon', 'delta_p_h', 'position_error_enu', 'position_error_all', 'trajectory', "cdf"]
    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M')
    os.makedirs(path)
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_name + ".png"), dpi=300)
    plt.show(block=True)

def plot_traj():
    '''绘制中断60, 120, 180的平面轨迹对比图，对你纯惯导， LSTM'''
    path = "../img"
    dgps_x_60 = sco.loadmat(os.path.join(path, "60_dgps_x.mat"))['xc']# [0:3][0][0:1][0]
    dgps_y_60 = sco.loadmat(os.path.join(path, "60_dgps_y.mat"))['yc']
    dgps_x_120 = sco.loadmat(os.path.join(path, "120_dgps_x.mat"))['xc']
    dgps_y_120 = sco.loadmat(os.path.join(path, "120_dgps_y.mat"))['yc']
    dgps_x_180 = sco.loadmat(os.path.join(path, "180_dgps_x.mat"))['xc']
    dgps_y_180 = sco.loadmat(os.path.join(path, "180_dgps_y.mat"))['yc']

    ins_x_60 = sco.loadmat(os.path.join(path, "60_pure_ins_x.mat"))['xc']
    ins_y_60 = sco.loadmat(os.path.join(path, "60_pure_ins_y.mat"))['yc']
    ins_x_120 = sco.loadmat(os.path.join(path, "120_pure_ins_x.mat"))['xc']
    ins_y_120 = sco.loadmat(os.path.join(path, "120_pure_ins_y.mat"))['yc']
    ins_x_180 = sco.loadmat(os.path.join(path, "180_pure_ins_x.mat"))['xc']
    ins_y_180 = sco.loadmat(os.path.join(path, "180_pure_ins_y.mat"))['yc']

    t_60 = np.array([475330, 475390]) - 473600
    t_120 = np.array([474510, 474630]) - 473600
    t_180 = np.array([474320, 474500]) - 473600

    matplotlib.rcParams.update({'font.size': 17})
    matplotlib.rcParams.update({'lines.linewidth': 2})
    # 60s
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    # 120s
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    # 180s
    fig3, ax3 = plt.subplots(figsize=(10, 10))

    # [0:3][1,2][0:1][0]
    #x =dgps_x_60[0:3][1][0:1][0][0][t_60[0]:t_60[1]]
    ax1.plot(ins_x_60[0:3][1][0:1][0][0][t_60[0]:t_60[1]], ins_y_60[0:3][1][0:1][0][0][t_60[0]:t_60[1]])
    ax1.plot(dgps_x_60[0:3][1][0:1][0][0][t_60[0]:t_60[1]], dgps_y_60[0:3][1][0:1][0][0][t_60[0]:t_60[1]], 'g-')
    ax1.plot(dgps_x_60[0:3][0][0:1][0][0], dgps_y_60[0:3][0][0:1][0][0], 'r-')

    ax2.plot(ins_x_120[0:3][1][0:1][0][0][t_120[0]:t_120[1]], ins_y_120[0:3][1][0:1][0][0][t_120[0]:t_120[1]])
    ax2.plot(dgps_x_120[0:3][1][0:1][0][0][t_120[0]:t_120[1]], dgps_y_120[0:3][1][0:1][0][0][t_120[0]:t_120[1]], 'g-')
    ax2.plot(dgps_x_120[0:3][0][0:1][0][0], dgps_y_120[0:3][0][0:1][0][0], 'r-')

    ax3.plot(ins_x_180[0:3][1][0:1][0][0][t_180[0]:t_180[1]], ins_y_180[0:3][1][0:1][0][0][t_180[0]:t_180[1]])
    ax3.plot(dgps_x_180[0:3][1][0:1][0][0][t_180[0]:t_180[1]], dgps_y_180[0:3][1][0:1][0][0][t_180[0]:t_180[1]], 'g-')
    ax3.plot(dgps_x_180[0:3][0][0:1][0][0], dgps_y_180[0:3][0][0:1][0][0], 'r-')

    for ax in [ax1, ax2, ax3]:
        ax.legend([ 'pure INS', 'LSTM', 'true'], loc="upper left")
        ax.grid()
        ax.set(xlabel="East (m)", ylabel='North (m)')

    figs = [fig1, fig2, fig3]
    figs_name = ['60s-traj', '120s-traj', '180s-traj']
    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M') + "traj"
    os.makedirs(path)
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_name + ".png"), dpi=400)
    plt.show(block=True)

    print("")

def plot_delta_p_error(predicted_data, true_data):
    '''绘制神经网络预测的delta_p的箱型图,统计误差概率分布'''

    mlp = MLPData()
    mlp.load_data()
    mlp_delta_p = np.row_stack((-mlp.predict_delta_p_lat.T, -mlp.predict_delta_p_lon.T, -mlp.predict_delta_p_h.T))
    # 计算误差均值方差

    gps_test_true = get_gps_test_true()
    Re = 6378137
    # degree to meter
    predicted_data[0] = predicted_data[0] * (np.pi /180) * Re * 2*np.pi
    predicted_data[1] = predicted_data[1] * (np.pi /180) * Re * np.cos(gps_test_true[3:, 0]) * 2 * np.pi

    mlp_delta_p[0] = mlp_delta_p[0] * (np.pi /180) * Re * 2 * np.pi
    mlp_delta_p[1] = mlp_delta_p[1] * (np.pi /180) * Re * np.cos(gps_test_true[3:, 0]) * 2 * np.pi

    true_data[0] = true_data[0] * (np.pi /180) * Re * 2 * np.pi
    true_data[1] = true_data[1] * (np.pi /180) * Re * np.cos(gps_test_true[3:, 0]) * 2 * np.pi

    mae_lstm = np.sum(np.abs(predicted_data - true_data), axis=1) / len(true_data[0])
    mae_mlp = np.sum(np.abs(mlp_delta_p - true_data), axis=1) / len(true_data[0])
    std_lstm = np.std(predicted_data - true_data, axis=1)
    std_mlp = np.std(mlp_delta_p - true_data, axis=1)
    print(mae_lstm, mae_mlp, std_lstm, std_mlp, "mae_lstm:", "mae_mlp:", "std_lstm:", "std_mlp:")

    # 设置绘图属性
    matplotlib.rcParams.update({'font.size': 17})
    matplotlib.rcParams.update({'lines.linewidth': 1})

    t = [474000 + i for i in range(1077)]

    lstm_error = predicted_data - true_data
    mlp_error = mlp_delta_p - true_data

    fig, axs = plt.subplots(1, 3, figsize=(20,10))
    axs[0].boxplot([lstm_error[0],mlp_error[0]], labels=['LSTM', 'MLP'])
    axs[1].boxplot([lstm_error[1],mlp_error[1]], labels=['LSTM', 'MLP'])
    axs[2].boxplot([lstm_error[2],mlp_error[2]], labels=['LSTM', 'MLP'])
    axs[0].set(title=r"$\Delta P$ error in North", ylabel=r"$\Delta P_n$ Error (m)")
    axs[1].set(title=r"$\Delta P$ error in East", ylabel=r"$\Delta P_e$ Error (m)")
    axs[2].set(title=r"$\Delta P$ error in Up", ylabel=r"$\Delta P_u$ Error (m)")

    for ax in axs:
        ax.grid()
    figname = "delta_p_error"
    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M') + "traj"
    os.makedirs(path)
    fig.tight_layout()
    fig.savefig(os.path.join(path, figname + ".png"), dpi=400)

    plt.show(block=True)


if __name__ == "__main__":

    plot_res()

