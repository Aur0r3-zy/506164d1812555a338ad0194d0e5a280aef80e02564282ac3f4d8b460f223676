import tensorflow as tf
import numpy as np
import pandas as pd
import joblib


# 1. 加载 TensorFlow Lite 模型
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# 2. 预处理输入数据（特征在第四列到倒数第一列之间）
def prepro_for_inference(file_path,length):
    df = pd.read_csv(file_path)  # 读取CSV文件
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)  # 处理缺失值

    # 提取前三列数据（不包括表头）
    first_three_columns = df.iloc[:, :3]  # 提取前三列

    # 输出前三列内容（不包括表头）
    for index, row in first_three_columns.iterrows():
        print(row.values[0])
        print(row.values[1])
        print(row.values[2])# 输出每一行的前三列数据

    # 加载训练时保存的归一化器
    scaler = joblib.load('scaler.pkl')

    # 分割特征
    X = df.iloc[:, 3:]  # 特征数据

    # 使用与训练时相同的归一化器
    X_normalized = scaler.transform(X)  # 使用已加载的归一化器进行转换

    # 转换为 numpy 数组并重塑形状
    X = np.array(X_normalized)
    X = X.astype('float32') / 255

    print("预处理后的数据形状:", X.shape)
    print(X)
    return X


# 3. 使用 TensorFlow Lite 模型进行推理
def make_tflite_prediction(interpreter, X_input):
    # 获取输入输出张量的索引
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 输入数据的形状
    input_shape = input_details[0]['shape']
    print("模型期望的输入形状:", input_shape)

    # 确保输入数据的形状与模型期望的输入形状匹配
    if len(X_input.shape) == 2 and len(input_shape) == 3:  # 假设模型期望三维数据
        X_input = np.expand_dims(X_input, axis=-1)  # 为输入数据添加一个维度，转为三维数据
        print("调整后的数据形状:", X_input.shape)

    # 为 TensorFlow Lite 输入张量准备数据
    interpreter.set_tensor(input_details[0]['index'], X_input.astype(np.float32))  # 转换为 float32 类型

    # 执行推理
    interpreter.invoke()

    # 获取推理结果
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 假设是分类问题，使用 argmax 得到预测类别
    y_pred_classes = np.argmax(output_data, axis=1)

    return y_pred_classes


# 4. 后处理预测结果（例如转换为标签或其他格式）
def postprocess_predictions(predictions):
    # 在这里根据需要转换预测结果，例如将类别索引转换为标签
    # 攻击类别类别索引与标签一一对应
    label_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}  # 示例映射
    return [label_mapping[pred] for pred in predictions]


# 主函数：加载模型并进行预测
if __name__ == '__main__':
    # 设置模型路径和输入数据路径
    model_path = './student_model_quantized.tflite'  #  TensorFlow Lite 模型路径
    file_path = '../data/test.csv'  # 待预测数据输入数据路径
    length = 76  # 特征的长度

    # 1. 加载 TensorFlow Lite 模型
    interpreter = load_tflite_model(model_path)

    # 2. 预处理输入数据
    X_input = prepro_for_inference(file_path, length)

    # 3. 进行推理
    predictions = make_tflite_prediction(interpreter, X_input)

    # 4. 后处理预测结果
    processed_predictions = postprocess_predictions(predictions)

    # 打印或保存预测结果
    print("预测结果：", processed_predictions)
