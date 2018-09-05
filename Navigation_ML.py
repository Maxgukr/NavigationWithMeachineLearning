import xgboost as xgb
import numpy as np
from readfile import loaddata
import math


class data():
    def __init__(self,
                 east_pos,  # 经度
                 north_pos,  # 维度
                 height,  # 高程
                 east_v,  # 东向速度
                 north_v,  # 北向速度
                 groud_v,  # 地向速度
                 roll,  # 横滚角
                 pitch,  # 俯仰角
                 yaw  # 航向角
                 ):

        self.longitude = east_pos
        self.latitude  = north_pos
        self.height    = height
        self.east_v    = east_v
        self.north_v   = north_v
        self.groud_v   = groud_v
        self.roll      = roll
        self.pitch     = pitch
        self.yaw       = yaw
        self.e1        = 0.00669437999013  # 椭球模型的第一偏心率平方
        self.a         = 6378137  # 椭球长半轴
        self.ned       = np.zeros((3,1),dtype=float)


    def convertAxis(self):
        '''
        进行坐标系转换，地球坐标系转为NEU的导航坐标系
        :return:
        '''
        RM = self.a * (1-self.e1) / math.pow(1-self.e1*math.pow(math.sin(self.latitude),2),1.5)  # 子午圈曲率半径
        RN = self.a / math.sqrt(1-self.e1*math.pow(math.sin(self.latitude),2)) #  卯酉圈曲率半径
        c = np.zeros((3,3),dtype=float)
        c[0][0] = RM + self.height
        c[1][1] = (RN + self.height) * math.cos(self.latitude)
        c[2][2] = -1.0
        self.ned = np.dot(c,[self.latitude,self.longitude,self.height])

        #return NED


