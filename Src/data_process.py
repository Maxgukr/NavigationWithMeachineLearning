import numpy as np
import pandas as pd
import math
from utils import generate_data_from_csv
from sklearn.preprocessing import StandardScaler

class DataLoader():
    """为lstm模型生成所需要的数据，包括训练集和测试集"""

    def __init__(self, path, split, cols):

        df_data, df_label = generate_data_from_csv(path)
        self.scaler_train = StandardScaler()
        # 使用fit有利于后续反变换
        df_data = df_data.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        i_split = math.ceil(len(df_data) * split)

        self.test_pos = df_data[i_split:].get(['lat', 'lon', 'height']).values # 用与惯导更新的数据做比较
        #cols = cols.insert(0, 'time')
        self.test_data = df_data[i_split:].get(cols).values #  用于机械编排的数据

        data = self.scaler_train.fit_transform(df_data.get(cols[1:13]).values)

        self.scaler_label = StandardScaler()
        label_data = df_label.get(['lat', 'lon', 'height']).values
        self.label_train = self.scaler_label.fit_transform(label_data[:int(i_split/200)]) #  训练标签数据

        self.data_train = data[:i_split] #  用于训练数据的部分
        self.data_test  = data[i_split:] #  用于测试的数据部分
        self.len_data_train  = len(self.data_train) #  训练数据长度
        self.len_data_test   = len(self.data_test) #  测试数据长度

        self.label_test = label_data[int(i_split/200):] #  测试标签数据，未归一化
        self.len_label_train = len(self.label_train)
        self.len_label_test = len(self.label_test)

    def get_test_data(self, seq_len, normalise):
        '''
        得到测试数据集(x,y) 确保内存足够，不然减少数据集的尺寸
        '''
        data_x = []
        data_y = []
        for i in range(self.len_label_test - 3):
            data_x.append(self.data_test[i*200:i*200+seq_len])

        for i in range(self.len_label_test-3):
            data_y.append(self.label_test[i+3])

        data_x = np.array(data_x).astype(float)
        data_y = np.array(data_y).astype(float)
        #data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        return data_x, data_y

    def get_train_data(self, seq_len, normalise):
        '''
        创建训练数据(x,y)
        需要确保内存可以足够装下所有的训练数据，不然需要使用用generate_train_batch方法
        '''
        data_x = []
        data_y = []
        for i in range(self.len_label_train - 3):
            data_x.append(self.data_train[i*200:i*200 + seq_len])

        for i in range(self.len_label_train-3):
            data_y.append(self.label_train[i+3])

        data_x, data_y = np.array(data_x), np.array(data_y)

        assert data_x.shape[0] == data_y.shape[0]
        return data_x, data_y
