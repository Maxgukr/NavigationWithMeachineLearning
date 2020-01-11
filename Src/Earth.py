class Earth():
    '''地球椭球模型相关参数计算 WGS84'''
    def __init__(self):
        """
        :param pos: 位置[纬，经，高]
        :param vn: 速度[N,E,D]
        """
        # 长半轴
        self.a = 6.378137e6
        # 短半轴
        self.b = 6356752.3141
        # 扁率
        self.f = 1.0/298.2572236
        # 自传角速度
        # self.we = 1.0/298.2572236
        self.we=7.2922115147e-5
        # 第一偏心率平方
        self.e2 = 0.00669437999013
        '''
        # 赤道正常重力
        self.g0 = 9.7803267714
        # 计算参数
        self.sinL = np.sin(pos[0]) # pos[0]代表纬度
        self.cosL = np.cos(pos[0])
        self.sqrt = np.sqrt(1-self.e2*self.sinL*self.sinL)
        self.sqrt2 = self.sqrt*self.sqrt
        # 子午圈曲率半径
        self.RM = self.a * (1-self.e2) / (self.sqrt * self.sqrt2)
        # 卯酉圈曲率半径
        self.RN = self.a / self.sqrt
        # 地球自传角速度在n系的投影
        self.wnie = np.array([self.we * np.cos(pos[0]), 0, -self.we * np.sin(pos[0])]) # [N,E,D]
        # 载体相对地球运动产生的角速度在n系的投影
        self.wnen = np.array([vn[1]/(self.RN + pos[2]), -vn[0]/(self.RM + pos[2]), -vn[1]*np.tan(pos[0])/(self.RN + pos[2])]) # [N,E,D]
        self.wnin = self.wnie + self.wnen
        # 地球表面重力变化模型
        self.gLH = self.g0 * (1+5.27094e-3 * np.power(self.sinL, 2) + 2.32718e-5 * np.power(self.sinL, 4)) - 3.086e-6 * pos[2]
        self.gn = np.array([0,0,self.gLH]) # ned, 重力是正号
        '''