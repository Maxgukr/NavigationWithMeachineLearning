import numpy as np
import math

def euler2dcm(att):
    """
    :param att: 姿态角[phi, theta, psi] 横滚,俯仰,航向
    :return:
    """
    s = np.sin(att)
    c = np.cos(att)
    dcm = np.array([[c[1]*c[2], -c[0]*s[2]+s[0]*s[1]*c[2], s[0]*s[2]+c[0]*s[1]*c[2]],
           [c[1]*s[2], c[0]*c[2]+s[0]*s[1]*s[2], -s[0]*c[2]+c[0]*s[1]*s[2]],
           [-s[1], s[0]*c[1], c[0]*c[1]]])
    return dcm

def askew(v):
    """
    一维向量v的反对称矩阵
    :param v:[vx, vy, vz]
    :return: 三维反对称矩阵
    """
    m = np.array([[0, -v[2][0], v[1][0]],
         [v[2][0], 0, -v[0][0]],
         [-v[1][0], v[0][0], 0]])

    return m

def quater2dcm(q):
    """
    姿态四元数转方向预先矩阵
    :param q: 四元数
    :return:
    """
    q11 = q[0][0]*q[0][0]
    q12 = q[0][0]*q[1][0]
    q13 = q[0][0]*q[2][0]
    q14 = q[0][0]*q[3][0]

    q22 = q[1][0]*q[1][0]
    q23 = q[1][0]*q[2][0]
    q24 = q[1][0]*q[3][0]

    q33 = q[2][0]*q[2][0]
    q34 = q[2][0]*q[3][0]

    q44 = q[3][0]*q[3][0]

    dcm = np.array([[q11+q22-q33-q44, 2*(q23-q14), 2*(q24+q13)],
           [2*(q23+q14), q11-q22+q33-q44, 2*(q34-q12)],
           [2*(q24-q13), 2*(q34+q12), q11-q22-q33+q44]])

    return dcm

def euler2quater(att):
    """
    欧拉角转四元数
    :param att: 欧拉角 横滚，俯仰，横向
    :return:
    """
    c = np.cos(0.5*att)
    s = np.sin(0.5*att)

    q = np.array([c[0]*c[1]*c[2]+s[0]*s[1]*s[2],
         s[0]*c[1]*c[2]-c[0]*s[1]*s[2],
         c[0]*s[1]*c[2]+s[0]*c[1]*s[2],
         c[0]*c[1]*s[2]-s[0]*s[1]*c[2]])

    return q

def dcm2euler1(dcm):
    """
    方向矩阵得到欧拉角
    :param dcm: 方向矩阵
    :return: 欧拉角
    """
    att = np.array([math.atan(dcm[2][1]/dcm[2][2]),
           math.atan(-dcm[2][0]/(np.sqrt(np.square(dcm[2][1])+np.square(dcm[2][2])))),
           math.atan(dcm[1][0]/dcm[0][0])])

    return att

def dcm2euler(dc):
    pitch = math.atan(-dc[2,0]/np.sqrt(np.power(dc[2,1],2) + np.power(dc[2,2],2)))
    if dc[2,0] <= -0.999:
        roll = np.NaN
        heading = math.atan2((dc[1,2]-dc[0,1]),(dc[0,2]+dc[1,1]))
    elif dc[2,0] >= 0.999:
        roll = np.NaN
        heading = np.pi + math.atan2((dc[1,2]+dc[0,1]),(dc[0,2]-dc[1,1]))
    else:
        roll = math.atan2(dc[2,1], dc[2,2])
        heading = math.atan2(dc[1,0], dc[0,0])
    if heading<0:
        heading = heading + 2*np.pi
    return np.array([roll, pitch, heading])

def normquater1(q):
    """
    归一化四元数
    :param q: 四元数
    :return:
    """
    e = (np.matmul(q, q.T) - 1)/2.0
    q_n = (1-e) * q
    return q_n

def normquater(q):
    """
    归一化四元数
    :param q: 四元数
    :return:
    """
    mag = np.sqrt(np.matmul(q.T, q))
    # e = (np.matmul(q, q.T) - 1)/2.0
    q_n = q/mag
    return q_n

def quaterConj(q):
    """
    计算四元数的共轭
    :param q:
    :return:
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qmulq(q, p):
    """
    四元数乘法
    :param q1:
    :param q2:
    :return:
    """
    qs = q[0][0]
    qv = np.array([q[1][0], q[2][0], q[3][0]]).reshape(3, 1)
    ps = p[0][0]
    pv = np.array([p[1][0], p[2][0], p[3][0]]).reshape(3, 1)
    t = qs * pv + ps * qv + CrossProduct(qv, pv)
    t2 = np.array([qs * ps - np.matmul(qv.T,pv)]).reshape(1, 1)
    q1 = np.concatenate((t2, t), axis=0).reshape(4, 1)

    if q1[0][0] < 0:
          q1 = np.negative(q1)

    return q1

def qmulq1(q1, q2):

    q = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                  q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                  q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3],
                  q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]])
    return q

def qmulv(q, v):
    """
    四元数乘向量
    :param q: 四元数
    :param v: 向量
    :return:
    """
    qi = np.array([0, v[0], v[1], v[2]])
    qo = qmulq(qmulq(q, qi), quaterConj(q))
    vo = qo[1:4, 0]
    return vo

def NormalGravity(latitude, he):
    '''计算重力项'''
    a1 = 9.7803267715
    a2 = 0.0052790414
    a3 = 0.0000232718
    a4 = -0.000003087691089
    a5 = 0.000000004397731
    a6 = 0.000000000000721
    s2 = np.power(np.sin(latitude), 2)
    s4 = s2 * s2
    ng = a1 * (1 + a2 * s2 + a3 * s4) + (a4 + a5 * s2) * he + a6 * he * he

    return ng

def RC(a, e2, lat):
    '''计算曲率半径'''
    # M = a*(1-e2)/(1-e2 * np.sin(lat) ** 2) ** 3/2
    M = a*(1-e2)/(1-e2 * np.sin(lat) ** 2) ** (3/2)

    N = a/np.sqrt(1-e2 * np.sin(lat) ** 2)

    return M, N

def dist_ang(ang1, ang2):
    '''计算角度变化'''
    ang = ang2 - ang1
    if ang > np.pi:
        ang = ang - 2*np.pi
    elif ang < -np.pi:
        ang = ang + 2*np.pi

    return ang

def compensate(obs, bias, scale):
    '''补偿零偏和比例因子'''
    Corr = np.matmul(obs - bias, np.diag([1/(1+scale[0]), 1/(1+scale[1]), 1/(1+scale[2])]))
    return Corr


def pos2quat(lat, lon):
    s1 = np.sin(lon/2)
    c1 = np.cos(lon/2)
    s2 = np.sin(-np.pi/4-lat/2)
    c2 = np.cos(-np.pi/4-lat/2)

    q_ne = np.array([c1 * c2, -s1 * s2, c1 * s2, c2 * s1])

    return q_ne

def pos2dcm(lat, lon):
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    s_lon = np.sin(lon)
    c_lon = np.cos(lon)

    C_ne = np.array([[-s_lat*c_lon, -s_lon, -c_lat*c_lon],
         [-s_lat*s_lon, c_lon, -c_lat*s_lon],
         [c_lat, 0.0, -s_lat]])

    return C_ne

def CrossProduct(a, b):
    c = np.zeros((3, 1))
    c[0][0] = a[1][0]*b[2][0] - b[1][0]*a[2][0]
    c[1][0] = b[0][0]*a[2][0] - a[0][0]*b[2][0]
    c[2][0] = a[0][0]*b[1][0] - b[0][0]*a[1][0]
    return c

def rvec2quat(rot_vec):

    mag2 = np.power(rot_vec[0][0], 2) + np.power(rot_vec[1][0], 2) + np.power(rot_vec[2][0], 2)

    if mag2 < 1.0e-8:
        # == approximate solution ===
        mag2 = 0.25 * mag2

        c = 1.0 - mag2/2.0 * (1.0 - mag2/12.0 * (1.0 - mag2/30.0 ))
        s = 1.0 - mag2/6.0 * (1.0 - mag2/20.0 * (1.0 - mag2/42.0 ))

        q = np.array([c, s * 0.5 * rot_vec[0][0], s * 0.5 * rot_vec[1][0], s * 0.5 * rot_vec[2][0] ]).reshape(4, 1)
    else:
        # == Analytical solution ===
        mag = np.sqrt(mag2)
        s_mag = np.sin(mag/2)

        q = np.array([ np.cos(mag/2),
            rot_vec[0][0]*s_mag/mag,
            rot_vec[1][0]*s_mag/mag,
            rot_vec[2][0]*s_mag/mag ]).reshape(4, 1)

        if q[0][0] < 0:
            q = np.negative(q)

    return q

def dpos2rvec(lat, delta_lat, delta_lon):
    rv = np.array([ delta_lon * np.cos(lat),
                -delta_lat,
                -delta_lon * np.sin(lat)]).reshape(3, 1)

    return rv

def quat2pos(q_ne):
    lat = -2 * math.atan(q_ne[2][0]/q_ne[0][0]) - np.pi/2
    lon = 2 * math.atan2(q_ne[3][0], q_ne[0][0])

    return lat, lon

def TransRate(r_n, v_n, M, N):
    w_en = np.zeros((3, 1))
    w_en[0][0] =  v_n[1][0]/(N+r_n[2][0]) # Ve / (N+h)
    w_en[1][0] = -v_n[0][0]/(M+r_n[2][0]) # -Vn / (M+h)
    w_en[2][0] = -v_n[1][0] * np.tan(r_n[0][0]) / (N+r_n[2][0]) # -Ve*tan(lat) / (N+h)

    return w_en

def rvec2dcm(rot_vec):
     mag2 = np.power(rot_vec[0][0], 2) + np.power(rot_vec[1][0], 2) + np.power(rot_vec[2][0], 2)

     mag = np.sqrt(mag2)

     c_bn = np.eye(3,3) + (np.sin(mag)/mag)*askew(rot_vec) + ((1-np.cos(mag))/mag2)*np.matmul(askew(rot_vec), askew(rot_vec))

     return c_bn

def geo2ecef(e2, Rn, geo):
    c_lat = np.cos(geo[0][0])
    s_lat = np.sin(geo[0][0])
    c_lon = np.cos(geo[1][0])
    s_lon = np.sin(geo[1][0])

    Rn_h = Rn + geo[2][0]

    r_e = np.array([Rn_h*c_lat*c_lon, Rn_h*c_lat*s_lon, (Rn*(1-e2)+geo[2][0])*s_lat])

    return r_e


def INS_KF_TRN_PSI_STATE_15(dt, nav, par, gb_model, ab_model, kd_model):
    '''Transition matrix for the psi-angle error model, 16 state KF'''
    '''
    - INPUTS -
    nav.r
    nav.v
    nav.C_bn = DCM
    par.w_ie
    par.w_en
    par.Rm
    par.Rn
    par.g
    par.f_n % specific force in the n-frame 
    par.f_b % specific force in the b-frame
    par.w_b % angular rate in the b-frame
    - OUTPUT -
    PHI = 16x16 transition matrix
    '''
    phi = np.zeros((16, 16))

    # position error model
    # pos to pos
    phi[0:3, 0:3] = np.eye(3,3) + askew(-par.w_en)*dt
    # pos to vel
    phi[0:3, 3:6] = np.eye(3,3)*dt

    # velocity error model
    # vel to pos
    R = np.sqrt(par.Rm*par.Rn)
    phi[3, 0] = -par.g[2]/(R+nav.r[2])*dt
    phi[4, 1] = phi[3, 0]
    phi[5, 2] = -2*phi[3, 0]
    # vel to vel
    phi[3:6, 3:6] = np.eye(3) + askew(-2*par.w_ie-par.w_en)*dt
    # vel to att
    phi[3:6, 6:9] = askew(par.f_n)*dt
    # vel to acc bias
    phi[3:6, 12:15] = nav.C_bn*dt

    # att error model
    w_in = par.w_ie + par.w_en
    phi[6:9, 6:9] = np.eye(3,3) + askew(-w_in)*dt
    # att to gyro bias
    phi[6:9, 9:12] = -nav.C_bn*dt

    # gyro bias
    phi[9, 9] = gb_model[0]
    phi[10, 10] = gb_model[1]
    phi[11, 11] = gb_model[2]

    # acc bias
    phi[12, 12] = ab_model[0]
    phi[13, 13] = ab_model[1]
    phi[14, 14] = ab_model[2]

    # odom
    phi[15, 15] = kd_model[0]

    return phi


def KF_Predict(x, P, PHI, Q, dt):
    '''Kalman filter prediction'''
    # 一步预测值
    x1 = np.matmul(PHI, x)
    PHI_t = PHI.T
    Qd = 0.5 * (np.matmul(PHI, Q) + np.matmul(Q, PHI_t))*dt
    # 一步预测协方差
    P1 = np.matmul(np.matmul(PHI, P), PHI_t) + Qd

    return x1, P1

def KF_Update(x, P, inno, H, R):
    '''Kalman Filter Update'''
    PHt = np.matmul(P, H.T)
    HPHt = np.matmul(H, PHt)
    RHPHt = R + HPHt

    # RHPHt[0, 0] = RHPHt[0,0] + 1e-9
    # RHPHt[1, 1] = RHPHt[1,1] + 1e-9
    # RHPHt[2, 2] = RHPHt[2,2] + 1e-9
    is_not_pdf = False
    try:
        L= np.linalg.cholesky(RHPHt)
        # kalman 增益
        K = np.matmul(PHt, np.linalg.inv(RHPHt))
        dx = np.matmul(K, inno)
        x_up = x + dx
        IKH = np.eye(len(x)) - np.matmul(K, H)
        P_up = np.matmul(np.matmul(IKH, P), IKH.T) + np.matmul(np.matmul(K, R), K.T)
    except Exception: # NOT PDF
        is_not_pdf = True
        print("not positive definite")
        x_up = x
        P_up = P

    return x_up, P_up, is_not_pdf


if __name__ == "__main__":
    ini_att = np.array([-0.24743, 0.02929, 238.31190])*np.pi / 180
    dcm = euler2dcm(ini_att)
    quater1 = euler2quater(ini_att)
    dcm1 = quater2dcm(quater1.reshape(4, 1))
    att = dcm2euler(dcm1)*180/np.pi

    ini_pos = np.array([30.4569930274*np.pi/180, 114.4717650637*np.pi/180, 25.96469])

    q_pos = pos2quat(ini_pos[0], ini_pos[1])
    pos_q1, pos_q2 = quat2pos(q_pos.reshape(4, 1))
    pos1 = pos_q1*180/np.pi
    pos2 = pos_q2*180/np.pi

    q1=np.array([1,2,3,4]).reshape(4, 1)
    q2=np.array([1,2,3,4]).reshape(4, 1)
    qq1 = qmulq(q1, q2)
    qq2 = qmulq1(q1, q2)

    ini_rot_vec = np.array([0.234, 0.356, 1.234])
    dcm_rv = rvec2dcm(ini_rot_vec.reshape(3, 1))
    q_rv = rvec2quat(ini_rot_vec.reshape(3, 1))
    dcm_q_rv = quater2dcm(q_rv.reshape(4, 1))

    q_norm = normquater(np.array([1,2,3,4]))
    q_norm1 = normquater1(np.array([1,2,3,4]))

    print("")


    print(qmulq(np.array([1,2,3,4]).reshape(-1,1),np.array([5,6,7,8]).reshape(-1,1)))