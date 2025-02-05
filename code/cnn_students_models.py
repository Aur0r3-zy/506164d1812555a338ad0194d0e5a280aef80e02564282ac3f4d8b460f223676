import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def flexible_student_model(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 扩展的卷积模块
    h = layers.Conv1D(16, 5, padding='same', activation='relu')(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPool1D(2)(h)
    h = layers.Dropout(0.3)(h)

    h = layers.Conv1D(32, 5, padding='same', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPool1D(2)(h)
    h = layers.Dropout(0.3)(h)

    h = layers.Conv1D(64, 3, padding='same', activation='relu')(h)
    h = layers.BatchNormalization()(h)
    h = layers.GlobalAveragePooling1D()(h)

    # 扩展的全连接部分
    h = layers.Dense(128, activation='relu')(h)
    h = layers.Dropout(0.6)(h)
    h = layers.Dense(64, activation='relu')(h)
    outputs = layers.Dense(7, activation='softmax')(h)

    return keras.Model(inputs, outputs, name="enhanced_flexible_student")

def rigid_student_model(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 单一卷积层，固定结构
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    # Flatten 和 全连接层
    h1 = layers.Flatten()(h1)
    h1 = layers.Dense(7, activation='softmax')(h1)

    model = keras.Model(inputs, h1, name="rigid_student_model")
    return model

def resource_efficient_student_model(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 小卷积核和少量通道
    h1 = layers.Conv1D(filters=4, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    # 简化的全连接层
    h1 = layers.Flatten()(h1)
    h1 = layers.Dense(32, activation='relu')(h1)
    h1 = layers.Dense(7, activation='softmax')(h1)

    model = keras.Model(inputs, h1, name="resource_efficient_student_model")
    return model


def higher_resource_student_model(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 多层卷积层
    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h2 = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h2 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h2)

    # 全连接层
    h2 = layers.Flatten()(h2)
    h2 = layers.Dense(128, activation='relu')(h2)
    h2 = layers.Dropout(0.4)(h2)  # Dropout正则化
    h2 = layers.Dense(7, activation='softmax')(h2)

    model = keras.Model(inputs, h2, name="higher_resource_student_model")
    return model
