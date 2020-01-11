import os
import json
from plot_Result import plot_NN_results, plot_delta_p_error
from data_process import DataLoader
from Model import LSTMModel
import time
import pickle
import scipy.io as sco

def main():
    configs = json.load(open('config.json', 'r'))  # 加载配置文件
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    path_ref = "../data/M39/M39_20190710_ref.nav"
    path_imu = "../data/M39/M39_20190710.imu"
    path_gps = "../data/M39/M39_20190710.gps"
    path = [path_gps, path_imu, path_ref]

    data = DataLoader(
        path, #  文件路径数据放在src上一级的data文件夹下
        configs['data']['train_test_split'], # 训练集和测试集的分割比例
        configs['data']['columns'] #  columns是所有特征的名字，速度，姿态等, 在config.json中定义
    )

    # 训练LSTM模型
    model = LSTMModel()
    model_path = os.listdir(configs['model']['save_dir'])[0]
    if(configs['model']['mode']=="test"):
        model.load_model(os.path.join(configs['model']['save_dir'], model_path))
    else:
        # 加载数据
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        # 构建模型
        model.build_model(configs)
        start = time.time()
        history = model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
        end = time.time()
        print("training time is", end-start)

        with open('./history'+str(end-start)+str('.p'), 'wb') as file:
            pickle.dump(history, file)

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.predict_point_by_point(x_test)

    predict_inerse = data.scaler_label.inverse_transform(predictions)



    plot_delta_p_error(predict_inerse.T, y_test.T)

    print("end")

if __name__ == '__main__':
    main()