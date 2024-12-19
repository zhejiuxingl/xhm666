# 基本库
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical
import os
import librosa
import glob
from tqdm import tqdm

# 数据预处理
feature = []
label = []
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

def extract_features(parent_dir, sub_dirs, max_file=100, file_ext="*.wav"):
    feature, label = [], []
    for sub_dir in sub_dirs:
        # 使用 tqdm 包装 glob.glob 的结果，并应用 max_file 限制
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            label_name = fn.split(os.sep)[-2]
            if label_name in label_dict:  # 确保 label_name 在 label_dict 中
                label.append(label_dict[label_name])
            else:
                print(f"Label '{label_name}' not found in label_dict. Skipping file: {fn}")
            # 读取音频并提取梅尔频谱特征
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            feature.append(mels)
    return np.array(feature), np.array(label)

# 数据路径
parent_dir = 'F:\\xz\\train'
sub_dirs = ['aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots', 'chips', 'chocolate', 'drinks',
            'fries', 'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles', 'pizza', 'ribs',
            'salmon', 'soup', 'wings']

# 获取特征feature以及类别的label
X, Y = extract_features(parent_dir, sub_dirs, max_file=10000)

# 打印特征和标签的维度
print('X的特征尺寸是：', X.shape)
print('Y的特征尺寸是：', Y.shape)

# 将类别标签转为one-hot编码
Y = to_categorical(Y, num_classes=20)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, stratify=Y)
print('训练集的大小', len(X_train))
print('测试集的大小', len(X_test))

# 重新调整数据形状，以适配卷积神经网络
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)

# 搭建卷积神经网络模型
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=(16, 8, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.49))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(20, activation="softmax"))  # 20个类别

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 训练模型
history = model.fit(X_train, Y_train, epochs=2000, batch_size=660, validation_data=(X_test, Y_test))

# 测试集特征提取
def extract_features_test(test_dir, file_ext="*.wav"):
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        feature.append(mels)
    return np.array(feature)

# 提取测试集特征
X_test_new = extract_features_test('F:\\xz\\test_a')
X_test_new = X_test_new.reshape(-1, 16, 8, 1)

# 进行预测
predictions = model.predict(X_test_new)
preds = np.argmax(predictions, axis=1)
preds = [label_dict_inv[x] for x in preds]

# 输出结果
path = glob.glob('F:\\xz\\test_a/*.wav')
result = pd.DataFrame({'name': path, 'label': preds})
result['name'] = result['name'].apply(lambda x: x.split(os.sep)[-1])
result.to_csv('s.csv', index=None)

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))

# 绘制训练和验证准确率
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

# 绘制训练和验证损失
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
