from Earth import Earth
from INSLib import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import convertCoordinate
import datetime as dt
import os
import scipy.io as sio
from enum import Enum

class Color(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37

def print_color(text: str, fg: Color = Color.WHITE.value):
    print(f'\033[{fg}m{text}\033[0m')

class Nav_class():
    def __int__(self):
        self.r = None #np.zeros(3, 1)
        self.v = None #np.zeros(3, 1)
        self.C_bn = None #np.zeros(3, 3)
        self.q_bn = None #np.zeros(4, 1)
        self.q_ne = None #np.zeros(4, 1)
        self.d_vn = None #np.zeros(3, 1)
        self.roll = None
        self.pitch = None
        self.heading = None


class Par():
    def __int__(self):
        self.Rm = None
        self.Rn = None
        self.w_ie = None
        self.w_en = None
        self.g = None
        self.f_n = None


class GCF():
    #def __int__(self):
    f_imu = "/home/fangwei/PycharmProjects/data/M39/M39_20190710.imu"#
    f_gps = "/home/fangwei/PycharmProjects/data/M39/M39_20190710.gps"
    f_odom = "/home/fangwei/PycharmProjects/data/M39/M39_20190710.odo"

    t_start = 199500.0
    t_end = 203000.0

    d_rate = 200  # IMU rate

    arm = np.array([0.449, -0.08, -1.21]).reshape(3, 1)  #  IMU arm

    # gyro and acce bias and scale factor
    bg = np.array([-0.1055522569, 0.00127413194, 0.02129097]).reshape(3, 1) * np.pi / 180
    ba = np.array([0.00947509, -0.00459435, -0.02599986]).reshape(3, 1)
    sg = np.array([0.0129800347, -0.001539965, 0.000487986]).reshape(3, 1)
    sa = np.array([-0.00055396, -0.0011768, 0.00006767]).reshape(3, 1)

    # f_ref = "../data/M39/M39_20190710_ref.nav"
    # ref_nav = np.loadtxt(f_ref, delimiter=' ', dtype=float)
    # start_nav = ref_nav[abs(ref_nav[:, 0]-t_start)<1/d_rate/2, :]
    # initialnization pos, vel, att
    # [[ 1.99500001e+05  3.04556977e+01  1.14459645e+02  2.98077100e+01
    #    1.27238900e+01 -3.44370000e+00  1.27100000e-02 -7.82820000e-01
    #    9.00130000e-01  3.44299160e+02]]
    # for saving debuging time, use value at t_start straight
    init_pos = np.array([30.4556977, 114.459645, 29.80771]).reshape(3, 1) # deg
    init_vel = np.array([12.72389, -3.4437, -0.01271]).reshape(3, 1)
    init_att = np.array([0.78282, 0.90013, 344.29916]) # deg

    # init variance
    init_pos_var = 100
    init_vel_var = 0.01
    init_att_var = (np.array([2.0, 2.0, 5.0])*np.pi/180) ** 2

    opt_ins_align = 0  # Initial alignment mode: *% 0 = Use given; 1 = In-motion.

    f_gps_col = 14  # GPS file column
    gps_pos_std_scale = 1  # gps 位置方差尺度因子
    gps_vel_std_scale = 1  # gps 速度方差尺度因子

    # ZUPT
    var_zupt = 0.01**2  # 零速修正方差
    t_zupts = [] # 零速修正时间区间
    opt_zuptA = 1 # 零速修正的的时候是否约束航向角
    var_zuptA = (0.1*np.pi/180) ** 2 # 零速修正的航向角方差

    # Non-holonomic
    opt_holo = 1 # 是否启用非完整性约束
    var_holo = np.array([0.1, 0.1])**2  # 非完整性约束方差 (m/s)^2
    # Rotation matrix from IMU frame to vehicle frame.
    C_bv = euler2dcm(np.array([0*np.pi/180, 1.113*np.pi/180, -0.224*np.pi/180]))

    # Odometer update
    f_odom_col = 2
    var_odom = np.array([0.05])**2
    sf_odom = 1 # 里程计速度的比例因子
    # lever-arm of odometer update (from IMU to rear wheel) [m]
    la_odom = np.array([0.543, 0.629, 0.586]).reshape(3, 1)

    # IMU PERFERMANCE
    bg_var = (5/3600*np.pi/180) ** 2  # 陀螺零偏方差
    arw = 0.00667/60*np.pi/180  # 陀螺角度随机游走
    ba_var = (180*0.00001)**2  # 加速度计 零偏方差
    vrw = 0.05/60 # 速度随机游走
    bg_Tcor = 300 # 陀螺零偏相关时间
    ba_Tcor = 1000 # 加速度零偏相关时间
    kd_var = 0.00001 # 里程计方差
    kd_Tcor = 300 # 里程计相关时间

    # calculate necessary parameters for 1st order GM process, based on given parameters
    bg_model = np.exp(-1/bg_Tcor/d_rate)*np.array([1,1,1])
    ba_model = np.exp(-1/ba_Tcor/d_rate)*np.array([1,1,1])
    kd_model = np.exp(-1/kd_Tcor/d_rate)*np.array([1])
    q_bg = 2*bg_var/bg_Tcor
    q_ba = 2 * ba_var / ba_Tcor
    q_kd = 2 * kd_var / kd_Tcor

    # 初始化Q阵
    Q = np.diag([0,0,0, vrw**2, vrw**2, vrw**2, arw**2, arw**2, arw**2, q_bg, q_bg, q_bg, q_ba, q_ba, q_ba, q_kd])

    # output设置
    opt_feedback_bias = 1
    opt_feedback_kd = 1
    T_output = 0.999999

    path = '../result/' + dt.datetime.now().strftime('%Y%m%d-%H%M-') + "nav_res"
    if not os.path.exists(path) : os.makedirs(path)
    fname_output = 'ublox_Test_Rlarge'
    f_sol  = path + "/"+ fname_output + ".sol"
    #if not os.path.exists(f_sol) : os.makedirs(f_sol)

    f_fil = path + "/"+ fname_output + ".kf"
    #if not os.path.exists(f_fil) : os.makedirs(f_fil)


def insupdate_(meas1, meas, earth, nav, par):
    nav1 = Nav_class()
    par.Rm, par.Rn = RC(earth.a, earth.e2, nav.r[0][0])

    dt = meas[0][0] - meas1[0][0]

    # 速度的划桨补偿项
    scull = 0.5 * CrossProduct(meas[1:4], meas[4:7]) + \
                 (CrossProduct(meas1[1:4], meas[4:7]) +
                  CrossProduct(meas1[4:7], meas[1:4]))/12.0
    """更新速度"""
    # 外推中间时刻的位置
    mid_r = np.zeros((3, 1))
    mid_r[0][0] = nav.r[0][0]
    mid_r[2][0] = nav.r[2][0] - 0.5 * nav.v[2][0] * dt
    d_lat = 0.5 * nav.v[0][0] * dt / (par.Rm + mid_r[2][0])
    d_lon = 0.5 * nav.v[1][0] * dt / (par.Rn + mid_r[2][0]) / np.cos(nav.r[0][0])
    # 由位置的变化量得到旋转矢量
    d_theta = dpos2rvec(nav.r[0][0], d_lat, d_lon)
    # 得到中间时刻四元数
    mid_q = qmulq(nav.q_ne, rvec2quat(d_theta))
    mid_r[0][0], mid_r[1][0] = quat2pos(mid_q)
    # 更新重力项
    par.g = np.array([0, 0, NormalGravity(mid_r[0][0], mid_r[2][0])]).reshape(3, 1)
    # 计算中间时刻速度
    mid_v = nav.v + 0.5 * nav.d_vn
    # 计算n系旋转矩阵
    par.w_ie = earth.we * np.array([np.cos(mid_r[0][0]), 0.0, -np.sin(mid_r[0][0])]).reshape(3, 1)
    par.w_en = TransRate(mid_r, mid_v, par.Rm, par.Rn)
    zeta = (par.w_en + par.w_ie) * dt
    Cn = np.eye(3) - askew(0.5*zeta)
    # 比例积分项
    v = meas[1:4].reshape(3,1)+scull
    dv_f_n = np.matmul(np.matmul(Cn, nav.C_bn), v)
    par.f_n = dv_f_n /dt
    # 哥氏项
    dv_g_cor = (par.g - CrossProduct(2*par.w_ie + par.w_en, mid_v)) * dt
    nav1.d_vn = dv_f_n + dv_g_cor
    nav1.v = nav.v + nav1.d_vn
    """更新位置"""
    # 外推中间时刻速度
    '''
    # 更新高程
    nav1.r[2][0] = nav.r[2][0] - 0.5*(nav1.v[2][0] + nav.v[2][0])*dt
    # 更新维度
    h_ = 0.5*(nav1.r[2][0] + nav.r[2][0])
    nav1.r[0][0] = nav.r[0][0] + 0.5*(nav1.v[0][0] + nav.v[0][0])*dt/(par.Rm+h_)
    # 更新经度
    phi_ = 0.5*(nav1.r[0][0] + nav.r[0][0])
    par.Rm, par.Rn = RC(earth.a, earth.e2, phi_)
    nav1.r[1][0] = nav.r[1][0] + 0.5*(nav1.v[1][0] + nav.v[1][0])*dt/(par.Rn+h_)/np.cos(phi_)

    '''
    mid_v = 0.5*(nav.v + nav1.v)
    # Recompute w_en using updated velocity
    par.w_en = TransRate(mid_r, mid_v, par.Rm, par.Rn)
    zeta = (par.w_en + par.w_ie) * dt
    qn = rvec2quat(zeta)
    xi = np.array([0, 0, earth.we*dt]).reshape(3, 1)
    qe = rvec2quat(np.negative(xi))
    nav1.q_ne = qmulq(qe, qmulq(nav.q_ne, qn))
    nav1.q_ne = normquater(nav1.q_ne)
    nav1.r = np.zeros((3, 1))
    nav1.r[0][0], nav1.r[1][0] = quat2pos(nav1.q_ne)
    nav1.r[2][0] = nav.r[2][0] - mid_v[2][0] * dt
    """姿态更新"""
    # b系旋转矢量的圆锥补偿项
    beta = CrossProduct(meas1[4:7].reshape(3, 1), meas[4:7].reshape(3, 1))/12.0
    # b系四元数
    qb = rvec2quat(meas[4:7].reshape(3, 1) + beta)
    #  外推中间位置
    mid_r = np.array([nav1.r[0][0] + 0.5 * dist_ang(nav1.r[0][0], nav.r[0][0]),
                      nav1.r[1][0] + 0.5 * dist_ang(nav1.r[1][0], nav.r[1][0]),
                      0.5 * (nav1.r[2][0] + nav.r[2][0])]).reshape(3, 1)
    # 计算n系四元数
    par.w_ie = earth.we * np.array([np.cos(mid_r[0][0]), 0.0, -np.sin(mid_r[0][0])]).reshape(3, 1)
    par.w_en = TransRate(mid_r, mid_v, par.Rm, par.Rn).reshape(3, 1)
    zeta = (par.w_en + par.w_ie) * dt
    qn = rvec2quat(np.negative(zeta))
    nav1.q_bn = qmulq(qn, qmulq(nav.q_bn, qb))
    nav1.q_bn = normquater(nav1.q_bn)
    nav1.C_bn = quater2dcm(nav1.q_bn)
    nav1.roll, nav1.pitch, nav1.heading = dcm2euler(nav1.C_bn)

    return nav1, par


def insupdate(imu_data, len, nav, par, earth):
    # global nav
    pos = []
    vn = []
    att = []
    pos.append(GCF.init_pos)
    vn.append(GCF.init_vel)
    att.append(GCF.init_att)
    #len = self.end - self.start
    start = 184400
    meas = imu_data[start]
    # meas = np.concatenate((np.array([ins[0]]), compensate(ins[1:4], ba*self.ts, sa), compensate(ins[4:7]*np.pi/180, bg*self.ts, sg)))
    for i in range(1, len):
        # 上一时刻的值
        meas1 = meas
        # 新的观测量
        meas = imu_data[start+i]
        # meas = np.concatenate((np.array([ins[0]]), compensate(ins[1:4], ba * dt, sa), compensate(ins[4:7]*np.pi/180, bg * dt, sg)))
        nav, par = insupdate_(meas1, meas, earth, nav, par)
        vn.append([nav.v[0][0], nav.v[1][0], nav.v[2][0]])
        pos.append([nav.r[0][0] * 180 / np.pi, nav.r[1][0] * 180 / np.pi, nav.r[2][0]])
        att.append([nav.roll* 180 / np.pi, nav.pitch* 180 / np.pi, nav.heading* 180 / np.pi])
        if i%200 == 0:
            print("velocity :", vn[i], i)
            print("position :", pos[i], i)
            print("attitude :", att[i], i)

    pos, vn, att = np.array(pos), np.array(vn), np.array(att)


def INS_EKF(CFG):

    nav = Nav_class()
    par = Par()
    earth = Earth() # 地球参数

    nStates = 16
    # open INS file===================================================
    # time(s) gyro_x gyro_y gyro_z (rad) accel_x accel_y accel_z (m/s)
    f_ins = open(CFG.f_imu, 'rt')
    if f_ins is None:
        print("'Cannot open INS file!'")
        return

    t_end = CFG.t_end

    # open GPS file===================================================
    # time(sec) lat(rad) long(rad) height(m) std_lat std_long std_height (m) VN VE VD (m/s) std_VN std_VE std_VD (m/s)
    N_COL_F_GPS = CFG.f_gps_col
    f_gps = open(CFG.f_gps, 'rt')
    gps = np.zeros((N_COL_F_GPS, 1))
    if f_gps is None:
        print("'Cannot open GPS file!'")
        return
    else:
        # move gps file pointer to CFG.t_start
        while gps[0][0] < CFG.t_start:
            gps = np.array(f_gps.readline().split(' '), dtype=float).reshape(14, 1)
            if len(gps) != N_COL_F_GPS:
                f_gps.close()
                break

            gps[1:3] = gps[1:3]*np.pi/180

    # ====== Initialize the state and covariance ======
    x = np.zeros((nStates, 1))
    p = np.zeros((nStates, nStates))

    bg = CFG.bg
    ba = CFG.ba
    sg = CFG.sg
    sa = CFG.sa

    # alien  =====use given=====
    nav.r = np.array([GCF.init_pos[0] * np.pi / 180,
                      GCF.init_pos[1] * np.pi / 180,
                      GCF.init_pos[2]]).reshape(3, 1)
    nav.v = GCF.init_vel.reshape(3, 1)
    nav.C_bn = euler2dcm(GCF.init_att * np.pi / 180)
    ins = [-1]
    # Pointing the IMU file to CFG.t_start
    while ins[0] < CFG.t_start:
        ins = np.array(f_ins.readline().split(' '), dtype=float).reshape(7, 1)
        if len(ins) != 7:
            f_ins.close()
            print("Reached the end of INS file.\r\n")
            break

    dt = 1 / CFG.d_rate
    # 补偿IMU
    MEAS = np.concatenate((np.array([ins[0][0]]), compensate(ins[1:4][0], ba*dt, sa), compensate(ins[4:7][0]*np.pi/180, bg*dt, sg))).reshape(7, 1)

    nav.q_bn = euler2quater(GCF.init_att * np.pi / 180).reshape(4, 1)
    nav.q_ne = pos2quat(nav.r[0][0], nav.r[1][0]).reshape(4, 1)
    nav.d_vn = np.zeros((3, 1))

    # init p
    p[0, 0] = CFG.init_pos_var
    p[1, 1] = CFG.init_pos_var
    p[2, 2] = CFG.init_pos_var
    p[3, 3] = CFG.init_vel_var
    p[4, 4] = CFG.init_vel_var
    p[5, 5] = CFG.init_vel_var
    p[6, 6] = CFG.init_att_var[0]
    p[7, 7] = CFG.init_att_var[1]
    p[8, 8] = CFG.init_att_var[2]
    p[9, 9] = CFG.bg_var
    p[10, 10] = CFG.bg_var
    p[11, 11] = CFG.bg_var
    p[12, 12] = CFG.ba_var
    p[13, 13] = CFG.ba_var
    p[14, 14] = CFG.ba_var
    p[15, 15] = CFG.kd_var

    t = ins[0] # imu 的起始时刻
    # Find next gps solution after the current IMU time
    if f_gps is not None:
        while gps[0][0] <= t:
            gps = np.array(f_gps.readline().split(' '), dtype=float).reshape(14, 1)
            if len(gps) != N_COL_F_GPS:
                f_gps.close()
                break
            gps[1:3] = gps[1:3]*np.pi/180

    # Find next odom after the current IMU time
    if not os.path.exists(CFG.f_odom):
        f_odom = None
    else:
        f_odom = open(CFG.f_odom, 'rt')
    odom = [-999, -999]
    if f_odom:
        while odom[0] <= t:
            odom = np.array(f_odom.readline().split(' '), dtype=float).reshape(2, 1)
            if len(odom) != CFG.f_odom_col:
                f_odom.close()
                break

    # open outputfile
    f_sol = open(CFG.f_sol, 'w')

    f_fil = open(CFG.f_fil, 'w')

    print_color('Navigation...', fg=Color.GREEN.value)
    print('Start: %.3f, End: %.3f\n', t, t_end)

    t_update = t
    t_output = -1
    t_zupt = -1
    t_holo = -1
    n_update = 0

    # ===================main loop===================
    while t < t_end:
        MEAS1 = MEAS
        ins = np.array(f_ins.readline().split(' '), dtype=float).reshape(7, 1)
        if len(ins) != 7:
            print_color("Reached the end of INS file", fg=Color.RED.value)
            break
        dt = 1/CFG.d_rate
        t = ins[0]

        MEAS = np.concatenate((np.array([ins[0][0]]),
                               compensate(ins[1:4][0], ba*dt, sa),
                               compensate(ins[4:7][0]*np.pi/180, bg*dt, sg))).reshape(7, 1)
        # mechanization
        nav, par = insupdate_(meas1=MEAS1, meas=MEAS, earth=earth, nav=nav, par=par)
        # kalman prediction
        PHI = INS_KF_TRN_PSI_STATE_15(dt, nav, par, CFG.bg_model, CFG.ba_model, CFG.kd_model)
        x, p = KF_Predict(x, p, PHI, CFG.Q, dt)

        # GPS measurement update
        # ======================
        if f_gps is not None:
            if abs(t - gps[0])<1/CFG.d_rate/2:
                if int(t) not in CFG.n_nogps: # 不在中断GNSS的区间内
                    # gps的位置
                    r_gps = [gps[1], gps[2], gps[3]]
                    par.Rm, par.Rn = RC(earth.a, earth.e2, r_gps[0])
                    # 将INS的位置和GPS的位置统一转换到e系
                    r_gps_e = geo2ecef(earth.e2, par.Rn, r_gps)
                    r_ins_e = geo2ecef(earth.e2, par.Rn, nav.r)
                    C_en = pos2dcm(r_gps[0], r_gps[1]).T
                    la_r = nav.C_bn*CFG.arm
                    # calculate the position difference in ECEF frame is better for high latitude area
                    z = C_en * (r_ins_e - r_gps_e)+la_r
                    if abs(gps[4]+111) < 0.01:
                        # only position measurement
                        std_gps_pos = CFG.gps_pos_std_scale*[gps[7], gps[8], gps[9]]
                        # 量测噪声方差
                        R = np.diag(std_gps_pos**2)
                        H = np.zeros((3, nStates))
                        H[0:3, 0:3] = np.eye(3,3)
                        inno = z - np.matmul(H[0:3,:], x)
                        x, p, is_not_pdf = KF_Update(x, p, inno, H[0:3, :], R)
                        if is_not_pdf:
                            print('%.3f: Not positive definete in GPS update!\n', t)
                            break
                        n_update = 3
                    else:
                        # have vel measurement
                        C_ec = pos2dcm(nav.r[0][0], nav.r[1][0]) # 从e系到里程计所在的车辆系
                        C_nc = np.matmul(C_ec, C_en)
                        w_in = par.w_ie + par.w_en
                        w_ib = MEAS[1:4] / dt
                        z_v = nav.v - CrossProduct(w_in,la_r)- np.matmul(nav.C_bn, CrossProduct(CFG.arm,w_ib)) - \
                              C_nc * [gps[4], gps[5], -gps[6]]
                        # input vector
                        z = np.vstack((z, z_v))
                        # measurement matrix
                        H = np.zeros((6, nStates))
                        H[0:6, 0:6] = np.eye(6,6)
                        std_gps_pos = CFG.gps_pos_std_scale*[gps[7], gps[8], gps[9]]
                        std_gps_vel = CFG.gps_vel_std_scale*[gps[7], gps[8], gps[9]]
                        R = np.diag([std_gps_pos**2, std_gps_vel**2])
                        inno = z[0:6, 0] - np.matmul(H[0:6, :], x)
                        x, p, ndf = KF_Update(x, p, inno, H[0:6,:], R)
                        if ndf:
                            print('%.3f: Not positive definete in GPS update!\n', t)
                            break
                        n_update = 6

                # read the next GPS measurement
                gps = np.array(f_gps.readline().split(' '), dtype=float)
                if len(gps) != N_COL_F_GPS:
                    f_gps.close()
                    break
                gps[1:2] = gps[1:3] * np.pi / 180

        # End of GPS measurement
        # =======================
        # Start odometer update
        if f_odom is not None:
            if abs(t-odom[0]) < 1/CFG.d_rate/2:

                C_nv = np.matmul(CFG.C_bv, nav.C_bn.T)
                w_ib = MEAS[1:4] / dt
                w_nb_b = w_ib - np.matmul(nav.C_bn.T, par.w_ie + par.w_en)
                # tansform velocity in b frame to vehicle frame
                v_v = np.matmul(C_nv, nav.v) + np.matmul(CFG.C_bv, CrossProduct(w_nb_b, CFG.la_odom))

                H_odom = np.zeros((3, nStates))
                H_odom[:, 3:6] = C_nv
                H_odom[:, 7:9] = np.matmul(-C_nv, askew(nav.v))
                H_odom[:, 9:12] = np.matmul(-CFG.C_bv, askew(CFG.la_odom))
                H_odom[:, 15] = odom[1]
                R = np.diag(CFG.var_odom[0])

                z_odom = v_v[0] - CFG.sf_odom*odom[1]
                inno = z_odom[0] - np.matmul(H_odom[0, :], x)
                x, p, ndf = KF_Update(x, p, inno, H_odom[0, :], R)
                if ndf:
                    print('%.3f: Not positive definete in odometer update!\n', t)
                    break
                n_update = n_update + 1
                odom = np.array(f_odom.readline().split(' '), dtype=float)
                if len(odom) != CFG.f_odom_col:
                    f_odom.close()
                    break

        # End of odom measurement
        # ======================
        # Start Zero velocity update
        id_zupt = 0  # belong_to(CFG.t_zupts, t) #TODO:
        if id_zupt:
            if (t-t_zupt >= 1) and (t - t_update > 0.5):
                t_update = t
                t_zupt = t
                H_zupt = np.zeros((3, nStates))
                H_zupt[:, 3:6] = np.eye(3, 3)
                #  观测
                z_zupt = nav.v - np.zeros((3, 1))
                #  新息
                inno = z_zupt - np.matmul(H_zupt, x)
                R = np.eye(3, 3) * CFG.var_zupt
                #  update
                x, p, _ = KF_Update(x, p, inno, H_zupt, R)
                n_update = n_update + 3

                if CFG.opt_zuptA == 1:  # Update heading gyro during ZUPT
                    H_zuptA = np.zeros((1, nStates))
                    H_zuptA[0, 11] = 1
                    z_zuptA = -(bg[2, 0] - MEAS[3, 1])
                    inno = z_zuptA - np.matmul(H_zuptA, x)
                    R_zuptA = CFG.var_zuptA
                    x, p, _ = KF_Update(x, p, inno, H_zuptA, R_zuptA)
                    n_update = n_update + 1

        # end of zupt
        # ========================
        # Start non-holonomic constraints
        if CFG.opt_holo ==1 and t - t_holo >= 0.01 and t - t_update > 0.01:
            t_update = t
            t_holo = t


            C_nv = CFG.C_bv * nav.C_bn.T
            w_ib = MEAS[1:4] / dt
            w_nb_b = w_ib - np.matmul(nav.C_bn.T, par.w_ie + par.w_en)
            v_v = np.matmul(C_nv, nav.v) + np.matmul(CFG.C_bv, CrossProduct(w_nb_b, CFG.la_odom))

            H_holo = np.zeros((3, nStates))
            H_holo[:, 3:6] = C_nv
            H_holo[:, 6:9] = np.matmul(-C_nv, askew(nav.v))
            H_holo[:, 9:12] = np.matmul(-CFG.C_bv, askew(CFG.la_odom))
            R = np.diag(CFG.var_holo)
            inno = v_v[1:3] - np.matmul(H_holo[1:3, :],x)
            x, p, is_not_pdf = KF_Update(x, p, inno, H_holo[1:3, :], R)
            if is_not_pdf:
                print('%.3f: Not positive definete in GPS update!\n', t)
                break
            n_update = n_update + 2
        #  write filtered solution
        if f_sol is not None:
            # position
            d_lat = x[0] / (par.Rm + nav.r[2])
            d_lon = x[1] / (par.Rm + nav.r[2]) / np.cos(nav.r[0])
            d_theta = dpos2rvec(nav.r[0], d_lat, d_lon)
            qn = rvec2quat(-d_theta)
            q_ne = qmulq(nav.q_ne, qn)
            r1, r2 = quat2pos(q_ne)
            r3 = nav.r[2][0] + x[2][0]

            # velocity
            C_cn = np.eye(3,3) + askew(d_theta)
            v_n = np.matmul(C_cn, nav.v - x[3:6, 0])

            # attitude correction
            phi_ang = x[6:9, 0] + d_theta
            C_pn = np.eye(3,3) + askew(phi_ang)
            C_bn = np.matmul(C_pn, nav.C_bn)
            att = dcm2euler(C_bn)

            sol = [t, r1, r2, r3, v_n[0][0], v_n[1][0], v_n[2][0], att[0], att[1], att[2],
                   np.sqrt(p[0][0]), np.sqrt(p[1][1]), np.sqrt(p[2][2]),
                   np.sqrt(p[3][3]), np.sqrt(p[4][4]), np.sqrt(p[5][5]),
                   np.sqrt(p[6][6]), np.sqrt(p[7][7]), np.sqrt(p[8][8])]

            f_sol.write(str(sol)+"\r\n")

            t_output = t

        if f_fil is not None:
            if n_update > 0:
                fil = [t, (bg+x[9:12, 0])[0, 0], (bg+x[9:12, 0])[1,0], (bg+x[9:12, 0])[2,0],
                       (ba + x[12:15, 0])[0, 0], (ba + x[12:15, 0])[1, 0], (ba + x[12:15, 0])[2, 0],
                       np.sqrt(p[9,9]), np.sqrt(p[10,10]), np.sqrt(p[11,11]),
                       np.sqrt(p[12, 12]), np.sqrt(p[13,13]), np.sqrt(p[14,14]),
                       CFG.sf_odom + x[15, 0]]
                f_fil.write(str(fil)+"\r\n")

        # position, velocity, attitude feedback
        if n_update > 0:
            # position feedback
            d_lat = x[0] / (par.Rm + nav.r[2])
            d_lon = x[1] / (par.Rm + nav.r[2]) / np.cos(nav.r[0])
            d_theta = dpos2rvec(nav.r[0], d_lat, d_lon)
            qn = rvec2quat(-d_theta)
            nav.q_ne = qmulq(nav.q_ne, qn)
            nav.r[0], nav.r[1] = quat2pos(nav.q_ne)
            nav.r[2] = nav.r[2][0] + x[2][0]
            # velocity feedback
            C_cn = np.eye(3, 3) + askew(d_theta)
            nav.v = np.matmul(C_cn, nav.v - x[3:6])
            # attitude correction
            phi_ang = x[6:9] + d_theta
            qe = rvec2quat(phi_ang)
            nav.q_bn = qmulq(qe, nav.q_bn)
            nav.C_bn = quater2dcm(nav.q_bn)

            x[0:9] = np.zeros((9, 1))

            if CFG.opt_feedback_bias == 1:
                bg = bg + x[9:12]
                ba = ba + x[12:15]
                x[9:15] = np.zeros((6, 1))

            if CFG.opt_feedback_kd == 1:
                CFG.sf_odom = CFG.sf_odom + x[15, 0]
                x[15, 0] = 0

        n_update = 0

    if f_sol is not None:
        f_sol.close()

    if f_fil is not None:
        f_fil.close()

    if f_ins is not None:
        f_ins.close()

    if f_gps is not None:
        f_gps.close()

    if f_odom is not None:
        f_odom.close()

    print('\nINS EKF processing finished. \n\n')


def plot_ins(test, len):

    true_position = test[:len, 1:4]
    true_xyz = convertCoordinate(true_position.T)
    ins_xyz  = convertCoordinate(pos.T)

    time = [i for i in range(len)]
    # 设置绘图属性
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'lines.linewidth': 3})

    # 北向，东向位置误差
    fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
    # 北向，东向速度误差
    fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    # 欧拉角误差
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

    axs1[0].plot(time, np.abs(true_xyz[0] - ins_xyz[0]))
    axs1[1].plot(time, np.abs(true_xyz[1] - ins_xyz[1]))
    axs1[2].plot(time, np.abs(true_xyz[2] - ins_xyz[2]))

    axs2[0].plot(time, np.abs(vn.T[0] - test[:len, 4:7].T[0]))
    axs2[1].plot(time, np.abs(vn.T[1] - test[:len, 4:7].T[1]))

    axs3[0].plot(time, np.abs(att.T[0] - test[:len, 7:10].T[0]))
    axs3[1].plot(time, np.abs(att.T[1] - test[:len, 7:10].T[1]))
    axs3[2].plot(time, np.abs(att.T[2] - test[:len, 7:10].T[2]))

    # save figures
    figs = [fig1, fig2, fig3]
    figs_name = ['pos_err', 'velocity_err', 'euler_err']
    path = '../result/' + "pure_ins"+dt.datetime.now().strftime('%Y%m%d-%H%M')
    os.makedirs(path)
    for i, fig in enumerate(figs):
        fig_name = figs_name[i]
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_name + ".png"), dpi=300)
    plt.show(block=True)


if __name__ == "__main__":

    GCF = GCF()
    INS_EKF(GCF)

