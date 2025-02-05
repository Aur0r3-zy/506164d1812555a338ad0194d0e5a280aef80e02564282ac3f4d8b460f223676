import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from scipy.spatial import cKDTree
import pandas as pd
def create_data(samples_num , n_features):
    y = np.array([i % 20 for i in range(samples_num)])  #生成5种标签
    print(y.shape)
    np.random.shuffle(y)  #打乱标签顺序

    X = np.random.randn(samples_num, n_features)  #生成数组

    patterns = {  #定义patterns字典，为每个类别设置特定模式
        0: (0.0, 0.2), #正常流量
        1: (3.0, 0.6),  # DoS
        2: (-2.0, 0.4),  # Probe
        3: (0.0, 1.5),  # R2L
        4: (1.5, 0.8)  # U2R
    }
    for cls, (shift,scale) in patterns.items():
        mask = (y == cls) #创建一个bool型数组，即y==cls的位置为True
        X[mask] += np.random.normal(loc = shift, scale = scale, size = (sum(mask), n_features))

    return X,y

class AIS:
    def __init__(self, n_detectors=30, self_radius=1.0, attack_radius=0.5):
        self.n_detectors = n_detectors
        self.self_radius = self_radius
        self.attack_radius = attack_radius
        self.detector_pools = {}  # 结构：{类别: {'detectors': [], 'kdtree': cKDTree}}

    def fit(self, X_self, X_attack, y_attack):
        self.classes = np.unique(y_attack)
        # 为每个类别独立生成检测器池
        for cls in self.classes:
            cls_samples = X_attack[y_attack == cls]
            detectors = []
            attempts = 0

            while len(detectors) < self.n_detectors and attempts < 1000:
                # 基于类别特征分布生成检测器
                if len(cls_samples) > 0:
                    center = cls_samples[np.random.choice(len(cls_samples))]
                else:
                    center = np.mean(X_attack, axis=0)

                candidate = center + np.random.normal(scale=0.1, size=X_self.shape[1])

                # 排除自体反应
                if np.min(np.linalg.norm(X_self - candidate, axis=1)) > self.self_radius:
                    detectors.append(candidate)
                attempts += 1

            # 存储检测器及加速结构
            self.detector_pools[cls] = {
                'detectors': np.array(detectors),
                'kdtree': cKDTree(detectors) if len(detectors) > 0 else None
            }

    def predict(self, X):
        y_pred = []
        for x in X:
            scores = {}
            for cls, pool in self.detector_pools.items():
                if pool['kdtree'] is None:
                    scores[cls] = np.inf
                    continue
                dist, _ = pool['kdtree'].query(x)
                scores[cls] = dist

            if len(scores) == 0:
                y_pred.append(0)  # 无检测器时判为正常
            else:
                best_cls = min(scores, key=lambda k: scores[k])
                y_pred.append(best_cls if scores[best_cls] <= self.attack_radius else 0)
        return np.array(y_pred)


'''
真实数据集准确率太低。。。
df = pd.read_csv('./KDDTrain+20_percent_3.csv')
X = df.iloc[:, :41].values
y = df.iloc[:, 41].values

X = StandardScaler().fit_transform(X)

X_self = X[y==0]  #正常网络行为
X_attack = X[y!=0] #攻击行为
y_attack = y[y!=0]

df2 = pd.read_csv('./KDDTest+3.csv')
x2 = df2.iloc[:, :41].values
y2 = df2.iloc[:, 41].values
x2 = StandardScaler().fit_transform(x2)
x2_self = x2[y2==0]
x2_attack= x2[y2!=0]
y2_attack= y2[y2!=0]
'''
if __name__ == '__main__':
    #随机生成的数据
    x,y = create_data(2000,20)
    x = StandardScaler().fit_transform(x)
    x_self = x[y==0]  #正常网络行为
    x_attack = x[y!=0] #攻击行为
    y_attack = y[y!=0]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    ais_model = AIS(
            n_detectors=500,
            self_radius=1,
            attack_radius=1
        )
    ais_model.fit(x_self,x_attack, y_attack)


    y_pred = ais_model.predict(x_attack)
    print('预测结果：',y_pred)
    match1 = np.sum(y_pred == y_attack)
    rate1 = match1/ len(y_pred)
    print(f"攻击行为匹配率：,{rate1:.2f}")

    print('-----------------------------------------------------------------------------------------------------------------------------------')
    false = ais_model.predict(x_self)
    print("自体集验证结果：",false)
    test = np.zeros(false.shape,dtype=int)
    test = test.flatten()
    match2 = np.sum(false == test)
    rate2 = match2/ len(false)
    print(f"放行合法流量概率：,{rate2:.2f}")

