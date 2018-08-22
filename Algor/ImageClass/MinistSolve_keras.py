import numpy as np
import os
import sys


def load_data():
    # 从文件中读取数据并进行预处理

    # 读取数据
    parent_path = os.path.dirname(sys.argv[0])
    data_path = os.path.join(parent_path, 'mnist.npz')
    data = np.load(data_path)
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    # 数据预处理
    # X归一化，平摊处理
    from sklearn import preprocessing
    x_scale = preprocessing.MinMaxScaler()
    original_shape = x_train.shape[1:]
    x_train = x_train.reshape(len(x_train), -1).astype('float32')
    x_test = x_test.reshape(len(x_test), -1).astype('float32')
    x_train = x_scale.fit_transform(x_train)
    x_test = x_scale.transform(x_test)

    # Y进行One-hot编码
    import keras
    y_train = keras.utils.to_categorical(y_train.reshape(len(y_train), -1), 10)
    y_test = keras.utils.to_categorical(y_test.reshape(len(y_test), -1), 10)
    data = {}
    data['x_train'], data['x_test'], data['y_train'], data['y_test'] = x_train, x_test, y_train, y_test
    data['original_shape'] = original_shape
    return data

def main():
    data = load_data()
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    original_shape = data['original_shape']

    net_type = 1
    if net_type == 1:
        # keras搭建普通的网络
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model
        input_img = Input(shape=(x_train.shape[1],))
        lay1 = Dense(256, activation='relu')(input_img)
        dro1 = Dropout(0.2)(lay1)
        lay2 = Dense(256, activation='relu')(dro1)
        dro2 = Dropout(0.2)(lay2)
        lay3 = Dense(256, activation='relu')(dro2)
        out = Dense(10, activation='softmax')(lay3)
        model = Model(input_img, out)
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=512,
                epochs=10, validation_data=[x_test, y_test])

        y_test_pre = model.predict(x_test).argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        print(y_test, y_test_pre)
    elif net_type == 2:
        # 卷积神经网络
        from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten
        from keras.models import Model
        x_train = x_train.reshape(len(x_train), *original_shape, 1)
        x_test = x_test.reshape(len(x_test), *original_shape, 1)
        # keras的输入shape是除了样本数量之外的其余维度
        input_img = Input(shape=(*original_shape, 1))
        conv1 = Conv2D(16, (3, 3), padding='same')(input_img)
        max1 = MaxPool2D((2, 2))(conv1)
        conv2 = Conv2D(16, (3, 3), padding='same')(max1)
        max2 = MaxPool2D((2, 2))(conv2)
        input1 = Flatten()(max2)
        lay1 = Dense(256, activation='relu')(input1)
        dro1 = Dropout(0.2)(lay1)
        lay2 = Dense(256, activation='relu')(dro1)
        dro2 = Dropout(0.2)(lay2)
        lay3 = Dense(256, activation='relu')(dro2)
        out = Dense(10, activation='softmax')(lay3)

        model = Model(input_img, out)

        model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=512,
                epochs=10, validation_data=[x_test, y_test])
        y_test_pre = model.predict(x_test).argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        print(y_test, y_test_pre)


if __name__ == '__main__':
    main()
