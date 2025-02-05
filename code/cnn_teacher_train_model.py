import joblib
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
#from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
from sklearn.metrics import classification_report
import warnings
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import LearningRateScheduler

import cnn_students_models
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


warnings.filterwarnings("ignore")
# 设置字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
tf.config.set_visible_devices([], 'GPU')


def model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs):
    model = mymodel(x_train)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define steps_per_epoch to avoid the ValueError
    steps_per_epoch = x_train.shape[0] // batch_size  # Calculate steps per epoch

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_valid, y_valid), steps_per_epoch=steps_per_epoch)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("=========模型训练结束==========")
    print("测试集结果： ", '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    y_predict = model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))
    return history, model


# 数据预处理代码
def prepro(file_path, spilt_rate):
    # 读取csv文件
    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 分割特征和标签
    X = df.iloc[:, 3:-1]  # 特征数据
    y = df.iloc[:, -1]   # 标签数据

    # 划分数据集
    # 先将数据分为训练集和剩余部分（测试集+验证集）
    x_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-spilt_rate, stratify=y, random_state=60)

    # 再将剩余部分划分为测试集和验证集
    x_valid, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=60)

    # 初始化归一化器，将范围设为 0-255
    scaler = MinMaxScaler(feature_range=(0, 255))

    # 只用训练集拟合归一化器，然后分别转换训练集和测试集
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_valid = scaler.transform(x_valid)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)

    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_valid, y_val, x_test, y_test, scaler

def lr_schedule(epoch):
    initial_lr = 0.001  # 初始学习率
    drop = 0.5  # 学习率下降的比例
    epochs_drop = 10  # 每10个epochs下降一次
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

# 归一化数据
def normalize_data(x_train, x_valid, x_test):
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_valid, x_test


def data_pre(file_path, spilt_rate, length):
    x_train, y_train, x_valid, y_valid, x_test, y_test,scaler = prepro(file_path, spilt_rate)
    x_train, x_valid, x_test = normalize_data(x_train, x_valid, x_test)
    print(x_train)

    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]

    # 打乱顺序
    index = [i for i in range(len(x_train))]
    random.seed(1)
    random.shuffle(index)
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]

    index1 = [i for i in range(len(x_valid))]
    random.shuffle(index1)
    x_valid = np.array(x_valid)[index1]
    y_valid = np.array(y_valid)[index1]

    index2 = [i for i in range(len(x_test))]
    random.shuffle(index2)
    x_test = np.array(x_test)[index2]
    y_test = np.array(y_test)[index2]

    x_train = tf.reshape(x_train, (len(x_train), length, 1))
    x_valid = tf.reshape(x_valid, (len(x_valid), length, 1))
    x_test = tf.reshape(x_test, (len(x_test), length, 1))


    print(x_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test,scaler


def acc_loss_line(teacher_history, student_history):
    print("绘制教师模型和学生模型的准确率和损失值曲线")

    # 检查历史对象是否包含训练信息
    if hasattr(teacher_history, 'history'):
        teacher_acc = teacher_history.history.get('accuracy', [])
        teacher_val_acc = teacher_history.history.get('val_accuracy', [])
        teacher_loss = teacher_history.history.get('loss', [])
        teacher_val_loss = teacher_history.history.get('val_loss', [])
    else:
        teacher_acc = teacher_val_acc = teacher_loss = teacher_val_loss = []

    if hasattr(student_history, 'history'):
        student_acc = student_history.history.get('accuracy', [])
        student_val_acc = student_history.history.get('val_accuracy', [])
        student_loss = student_history.history.get('loss', [])
        student_val_loss = student_history.history.get('val_loss', [])
    else:
        student_acc = student_val_acc = student_loss = student_val_loss = []

    # 绘制准确率曲线
    plt.figure(figsize=(12, 6))

    # 教师模型的准确率曲线
    if teacher_acc: plt.plot(teacher_acc, 'r', linestyle='-.', label="教师模型训练集准确率")
    if teacher_val_acc: plt.plot(teacher_val_acc, 'b', linestyle='dashdot', label="教师模型验证集准确率")

    # 学生模型的准确率曲线
    if student_acc: plt.plot(student_acc, 'g', linestyle='-.', label="学生模型训练集准确率")
    if student_val_acc: plt.plot(student_val_acc, 'orange', linestyle='dashdot', label="学生模型验证集准确率")

    plt.title('教师模型和学生模型的准确率曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend(loc='upper left')
    plt.show()

    # 绘制损失值曲线
    plt.figure(figsize=(12, 6))

    # 教师模型的损失值曲线
    if teacher_loss: plt.plot(teacher_loss, 'r', linestyle='-.', label="教师模型训练集损失值")
    if teacher_val_loss: plt.plot(teacher_val_loss, 'b', linestyle='dashdot', label="教师模型验证集损失值")

    # 学生模型的损失值曲线
    if student_loss: plt.plot(student_loss, 'g', linestyle='-.', label="学生模型训练集损失值")
    if student_val_loss: plt.plot(student_val_loss, 'orange', linestyle='dashdot', label="学生模型验证集损失值")

    plt.title('教师模型和学生模型的损失值曲线')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend(loc='upper right')
    plt.show()


def mymodel(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 初始卷积模块
    x = layers.Conv1D(16, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # 残差模块
    shortcut = x
    x = layers.Conv1D(16, 3, padding='same', activation='relu')(x)  # filters=16
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(16, 3, padding='same', activation='relu')(x)  # filters=16
    x = layers.BatchNormalization()(x)

    x = layers.add([shortcut, x])  # 现在可以相加了
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # 注意力机制
    attention = layers.GlobalAveragePooling1D()(x)
    attention = layers.Dense(16, activation='relu')(attention)  # 调整为 16
    attention = layers.Dense(16, activation='sigmoid')(attention)  # 调整为 16
    attention = layers.Reshape((-1, 16))(attention)  # 调整为 16
    x = layers.multiply([x, attention])

    # 分类头
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="optimized_cnn")

    # 添加L2正则化
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = keras.regularizers.l2(0.001)

    return model


# 学生模型选择函数
def select_student_model(model_type, x_train):
    if model_type == 'flexible':
        return students_models.flexible_student_model(x_train)
    elif model_type == 'rigid':
        return students_models.rigid_student_model(x_train)
    elif model_type == 'resource_efficient':
        return students_models.resource_efficient_student_model(x_train)
    elif model_type == 'higher_resource':
        return students_models.higher_resource_student_model(x_train)
    else:
        raise ValueError("Unknown model type")

# 动态调整蒸馏损失的权重
def distillation_loss_with_dynamic_weight(y_true, y_pred, teacher_logits, epoch, total_epochs, temperature=5.0):
    batch_size = tf.shape(y_true)[0]
    teacher_logits = tf.gather(teacher_logits, tf.range(batch_size))
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    student_probs = tf.nn.softmax(y_pred / temperature)

    distillation_loss = tf.reduce_mean(
        tf.keras.losses.kullback_leibler_divergence(teacher_probs, student_probs) * (temperature ** 2)
    )
    hard_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

    # 动态调整 alpha，随着训练的进行逐渐降低教师损失的权重
    alpha = 1 - (epoch / float(total_epochs))  # 逐步减小蒸馏损失的权重
    return alpha * distillation_loss + (1 - alpha) * hard_loss


def student_train_with_dynamic_weight(x_train, y_train, x_valid, y_valid, x_test, y_test, teacher_model, batch_size,
                                      epochs, temperature=5.0, model_type='flexible'):
    student_model = select_student_model(model_type, x_train)

    teacher_logits = teacher_model.predict(x_train)

    student_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                          loss=lambda y_true, y_pred: distillation_loss_with_dynamic_weight(y_true, y_pred,
                                                                                            teacher_logits, epochs,
                                                                                            epochs),
                          metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

    history = student_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(x_valid, y_valid),
                                callbacks=[LearningRateScheduler(lr_schedule), early_stopping])

    scores = student_model.evaluate(x_test, y_test, verbose=1)
    print("=========学生模型训练结束==========")
    print("测试集结果： ", '%s: %.2f%%' % (student_model.metrics_names[1], scores[1] * 100))

    y_predict = student_model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))

    return history, student_model


def convert_to_student_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('student_model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    print("学生模型已成功保存为 tflite 格式")

def convert_to_teacher_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('teacher_model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    print("教师模型已成功保存为 tflite 格式")

if __name__ == '__main__':
    file_path = '../data/cic.csv'
    length = 76
    spilt_rate = 0.7
    batch_size = 128
    epochs = 3
    x_train, y_train, x_valid, y_valid, x_test, y_test, scaler = data_pre(file_path, spilt_rate, length)

    teacher_history, teacher_model = model_train(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs)

    # 选择学生模型类型（灵活、固定、高效或高资源）

    model_type = 'flexible'  # 修改为你想要的模型类型：'flexible', 'rigid', 'resource_efficient', 'higher_resource'
    student_history, student_model = student_train_with_dynamic_weight(x_train, y_train, x_valid, y_valid, x_test,
                                                                       y_test, teacher_model, batch_size, epochs,
                                                                       model_type=model_type)
    convert_to_student_tflite(student_model)
    convert_to_teacher_tflite(teacher_model)

    # 保存归一化器
    joblib.dump(scaler, 'scaler.pkl')
    print("归一化器已保存为 'scaler.pkl'")

    acc_loss_line(teacher_history, student_history)
