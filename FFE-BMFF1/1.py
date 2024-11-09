import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing.sequence import pad_sequences

# 设定参数
max_features = 10000  # 仅考虑最常见的10000个单词
maxlen = 100  # 限定每条评论的最大长度
batch_size = 32

# 加载IMDb数据集
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# 数据预处理：将整数列表转换为形状为 (samples, maxlen) 的二维整数张量
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# 构建模型
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# 训练模型
history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=batch_size,
                    validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
