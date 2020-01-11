import os
import datetime as dt
from keras.layers import Dense, LeakyReLU, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class LSTMModel():
    """class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        """
        load an existing model
        :param filepath:
        :return:
        """
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        """
        build a new model from config.json file
        :param configs:
        :return:
        """
        print('[Model] Begin build Model')
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons,
                                     activation=activation))
                self.model.add(LeakyReLU(alpha=0.2))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons,
                                    input_shape=(input_timesteps, input_dim),
                                    return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
        #  设置学习率和学习率衰减因子
        adam = Adam(lr=0.001, decay=1e-8)

        self.model.compile(loss=configs['model']['loss'],
                           optimizer=adam,
                           metrics=[mean_squared_error])

        print('[Model] Model Compiled')

    def train(self, x, y, epochs, batch_size, save_dir):
        """
        :param x: train data
        :param y: label
        :param epochs: training epochs
        :param batch_size:
        :param save_dir: model saved director path
        :return:
        """

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        #  设置回调
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2000),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=1e-8, patience=1000, mode='auto', min_lr=1e-7)
            # 在patience个epoch中看不到模型性能提升，则减少学习率
        ]
        history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False,
            validation_split=0.2
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)

        return history

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        """
        :param data_gen: 数据生成器，用于train batch by batch
        :param epochs:
        :param batch_size:
        :param steps_per_epoch:
        :param save_dir:
        :return:
        """
        """生成式的训练方法，可以高效的并行运行模型，在GPU上进行分布式训练"""
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        # predicted = np.reshape(predicted, (predicted.size,))
        return predicted

