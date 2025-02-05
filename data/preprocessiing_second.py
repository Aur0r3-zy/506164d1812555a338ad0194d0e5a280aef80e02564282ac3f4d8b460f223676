import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取 CSV 文件
df = pd.read_csv('./22-02-2018.csv', header=None)

# 获取最后一列
last_column = df.iloc[:, -1]

# 使用 LabelEncoder 转换字符串为数字
encoder = LabelEncoder()
df.iloc[:, -1] = encoder.fit_transform(last_column)

# 删除所有第一个列数据为 "DST" 的行
#df = df[df.iloc[:, 0] != 'Dst Port']

# 如果你需要将结果保存为新 CSV 文件
df.to_csv('22-02processed_file_total.csv', index=False, header=False)
