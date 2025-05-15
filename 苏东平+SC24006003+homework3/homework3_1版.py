# -*- coding: utf-8 -*-
"""
2025.5.14   22:13,可用, 准确率为0.8498
@author: 苏东平
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 超参数配置
maxlen = 200  # 文本最大长度
vocab_size = 20000  # 词汇量
embedding_dim = 256  # 词向量维度
batch_size = 64  # 批大小
epochs = 10  # 训练轮次

# 加载IMDB本地数据集
def load_imdb_data(path):
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    # 加载训练集
    train_dir = os.path.join(path, 'train')
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                fpath = os.path.join(dir_name, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    train_texts.append(f.read())
                train_labels.append(1 if label_type == 'pos' else 0)
    # 加载测试集
    test_dir = os.path.join(path, 'test')
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                fpath = os.path.join(dir_name, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    test_texts.append(f.read())
                test_labels.append(1 if label_type == 'pos' else 0)
    return train_texts, np.array(train_labels), test_texts, np.array(test_labels)

# 加载数据
train_texts, train_labels, test_texts, test_labels = load_imdb_data('C:/Users/苏东平/Desktop/aclImdb')

# 文本预处理
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

# 构建优化后的模型
model = Sequential()
model.add(Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_dim, 
    input_length=maxlen
))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 添加早停法
es = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 训练模型
history = model.fit(
    train_data,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[es]
)

# 评估模型
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"最终测试准确率: {test_acc:.4f}")


# 在训练结束后添加可视化代码
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')        #训练准确率
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')  #验证准确率
    plt.title('Accuracy curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')                #训练损失
    plt.plot(history.history['val_loss'], label='Validation loss')          #验证损失
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 在模型评估后添加
plot_history(history)

# 添加混淆矩阵和分类报告
y_pred = (model.predict(test_data) > 0.5).astype("int32")

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(test_labels, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'positive'], 
            yticklabels=['negative', 'positive'])
plt.title('Confusion matrix')   # 混淆矩阵
plt.xlabel('Predicted label')   # 预测标签
plt.ylabel('True label')        # 真实标签
plt.show()

print("\n分类报告：")
print(classification_report(test_labels, y_pred, target_names=['negative', 'positive']))

# 添加示例预测展示
sample_texts = test_texts[:5]
sample_sequences = tokenizer.texts_to_sequences(sample_texts)
sample_data = pad_sequences(sample_sequences, maxlen=maxlen)
predictions = model.predict(sample_data)

print("\n示例预测：")
for text, pred in zip(sample_texts[:3], predictions[:3]):
    print(f"\n文本：{text[:50]}...")
    print(f"预测情感：{'正面' if pred > 0.5 else '负面'} (置信度：{float(pred[0]):.2f})")
    
    
    