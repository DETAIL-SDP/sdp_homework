# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:06:32 2025
@author: 苏东平
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置随机种子确保可重复性
np.random.seed(42)
torch.manual_seed(42)
# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据加载与预处理
# 从CMU数据库获取波士顿房价数据集
from urllib.request import urlopen
import re
import pandas as pd
import numpy as np
# 设置URL地址
url = "https://lib.stat.cmu.edu/datasets/boston"
# 从URL读取数据
with urlopen(url) as response:
    content = response.read().decode('latin1')  # 使用latin1编码读取
# 使用正则表达式更准确地找到数据部分
pattern = r'(\s+\d+\s+\d+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s*[\r\n])'
matches = re.findall(pattern, content)
# 如果找到了数据行
if matches:
    # 合并所有匹配行
    data_text = ''.join(matches)
    # 定义列名
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # 定义每列的宽度，这是固定宽度文件格式的关键
    # 根据数据格式调整这些宽度
    widths = [8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    # 使用pd.read_fwf并明确指定列宽
    df = pd.read_fwf(pd.io.common.StringIO(data_text), 
                     widths=widths, 
                     header=None, 
                     names=column_names)
    # 验证数据维度和列名
    print(f"数据集维度：{df.shape}")
    print("数据列名：", df.columns.tolist())
else:
    # 替代方案：如果网页“https://lib.stat.cmu.edu/datasets/boston”失效，使用sklearn的fetch_openml函数获取数据
    from sklearn.datasets import fetch_openml
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    df = boston.data
    df['MEDV'] = boston.target
    print(f"使用OpenML获取数据成功，维度：{df.shape}")
    print("数据列名：", df.columns.tolist())

# 目标列名为 "MEDV"（房价中位数）
target_col = 'MEDV'
features_cols = [col for col in df.columns if col != target_col]
# 随机分割数据集（不再使用顺序分割）
X = df[features_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 进一步分割训练集为训练和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------
# 2. 相关性分析
# ---------------------
print("\n== 相关性分析 ==")
corr_matrix = df.corr()
target_corr = corr_matrix[target_col].sort_values(ascending=False)
print("与房价相关的特征及系数：")
print(target_corr)
# 选择高相关性特征
threshold = 0.5
selected_features = target_corr[abs(target_corr) > threshold].index.tolist()
selected_features.remove(target_col)  # 移除目标变量
print(f"\n高相关性特征 (|相关系数| > {threshold})：{selected_features}")
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

# ---------------------
# 3. 主成分分析（PCA）
# ---------------------
print("\n== 主成分分析 ==")
pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance >= 0.9)[0][0] + 1
print(f"需要 {n_components} 个主成分解释 90% 的方差")
pca_final = PCA(n_components=n_components)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_val_pca = pca_final.transform(X_val_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

# ---------------------
# 4. 基准模型计算
# ---------------------
def calculate_baseline(y_true):
    baseline_pred = np.full_like(y_true, fill_value=np.mean(y_true))
    return np.sqrt(mean_squared_error(y_true, baseline_pred))

baseline_rmse = calculate_baseline(y_test)
print(f"\n基线模型（均值预测）RMSE: {baseline_rmse:.4f}")

# ---------------------
# 5. 线性回归模型训练与评估
# ---------------------
lr_model = LinearRegression()

# a) 相关性分析后的特征
print("\n== 相关性分析特征线性回归模型 ==")
lr_model.fit(X_train_selected, y_train)
y_pred_selected = lr_model.predict(X_test_selected)
selected_rmse = np.sqrt(mean_squared_error(y_test, y_pred_selected))
selected_r2 = r2_score(y_test, y_pred_selected)
print(f"RMSE: {selected_rmse:.4f} | R² Score: {selected_r2:.4f}")

# b) PCA 降维后的特征
print("\n== PCA 特征线性回归模型 ==")
lr_model.fit(X_train_pca, y_train)
y_pred_pca = lr_model.predict(X_test_pca)
pca_rmse = np.sqrt(mean_squared_error(y_test, y_pred_pca))
pca_r2 = r2_score(y_test, y_pred_pca)
print(f"RMSE (PCA): {pca_rmse:.4f} | R² Score (PCA): {pca_r2:.4f}")

# ---------------------
# 6. 集成模型（作为参考）
# ---------------------
print("\n== 随机森林模型 ==")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
print(f"随机森林 RMSE: {rf_rmse:.4f} | R² Score: {rf_r2:.4f}")
print("\n== 梯度提升树模型 ==")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_r2 = r2_score(y_test, y_pred_gb)
print(f"梯度提升树 RMSE: {gb_rmse:.4f} | R² Score: {gb_r2:.4f}")

# ---------------------
# 7. 简化的神经网络模型构建
# ---------------------
# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# 定义简化的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.layer3(x)
        return x

# 实例化模型
input_size = X_train_scaled.shape[1]  # 使用所有特征
model_nn = SimpleNN(input_size)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ---------------------
# 8. 训练神经网络模型（带早停）
# ---------------------
print("\n== 训练神经网络模型（带早停） ==")
# 设置训练参数
num_epochs = 1000
batch_size = 16
patience = 50  # 早停耐心值

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=len(X_val_tensor), shuffle=False)

# 训练循环
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model_nn.train()
    train_epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # 前向传播
        y_pred = model_nn(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
    
    train_avg_loss = train_epoch_loss / len(train_loader)
    train_losses.append(train_avg_loss)
    
    # 验证阶段
    model_nn.eval()
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = model_nn(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch).item()
            val_losses.append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model_nn.state_dict().copy()
            else:
                patience_counter += 1
    
    if (epoch+1) % 20 == 0:
        print(f'轮次 {epoch+1}/{num_epochs}, 训练损失: {train_avg_loss:.4f}, 验证损失: {val_loss:.4f}')
    
    # 早停条件
    if patience_counter >= patience:
        print(f'验证损失没有改善 {patience} 轮，提前停止训练')
        break

# 加载最佳模型
if best_model_state:
    model_nn.load_state_dict(best_model_state)
    print(f'已加载验证损失最低的模型')

# ---------------------
# 9. 交叉验证神经网络评估
# ---------------------
print("\n== 神经网络交叉验证 ==")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    # 分割数据
    X_tr = X_train_scaled[train_idx]
    y_tr = y_train.iloc[train_idx].values
    X_vl = X_train_scaled[val_idx]
    y_vl = y_train.iloc[val_idx].values
    
    # 转换为张量
    X_tr_tensor = torch.FloatTensor(X_tr)
    y_tr_tensor = torch.FloatTensor(y_tr).reshape(-1, 1)
    X_vl_tensor = torch.FloatTensor(X_vl)
    y_vl_tensor = torch.FloatTensor(y_vl).reshape(-1, 1)
    
    # 创建数据加载器
    tr_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    
    # 实例化模型
    fold_model = SimpleNN(input_size)
    fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # 训练模型
    for epoch in range(100):  # 使用较少的轮次
        fold_model.train()
        for X_batch, y_batch in tr_loader:
            y_pred = fold_model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            fold_optimizer.zero_grad()
            loss.backward()
            fold_optimizer.step()
    
    # 评估模型
    fold_model.eval()
    with torch.no_grad():
        y_vl_pred = fold_model(X_vl_tensor).numpy().flatten()
        fold_rmse = np.sqrt(mean_squared_error(y_vl, y_vl_pred))
        fold_r2 = r2_score(y_vl, y_vl_pred)
        cv_scores.append((fold_rmse, fold_r2))
    
    print(f"Fold {fold+1}: RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")

cv_rmse = np.mean([score[0] for score in cv_scores])
cv_r2 = np.mean([score[1] for score in cv_scores])
print(f"\n交叉验证平均结果: RMSE = {cv_rmse:.4f}, R² = {cv_r2:.4f}")

# ---------------------
# 10. 最终测试评估
# ---------------------
print("\n== 评估最佳神经网络模型 ==")
model_nn.eval()
with torch.no_grad():
    y_pred_tensor = model_nn(X_test_tensor)
    y_pred_nn = y_pred_tensor.numpy().flatten()
    test_loss = criterion(y_pred_tensor, y_test_tensor).item()
    rmse_nn = np.sqrt(test_loss)
    r2_nn = r2_score(y_test, y_pred_nn)

print(f'最佳神经网络测试集RMSE: {rmse_nn:.4f}')
print(f'最佳神经网络测试集R²: {r2_nn:.4f}')

# ---------------------
# 11. 模型对比与可视化
# ---------------------
print("\n== 模型对比 ==")
print(f"基线模型 RMSE: {baseline_rmse:.4f}")
print(f"线性回归 (相关性特征) RMSE: {selected_rmse:.4f} | R²: {selected_r2:.4f}")
print(f"线性回归 (PCA特征) RMSE: {pca_rmse:.4f} | R²: {pca_r2:.4f}")
print(f"随机森林 RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")
print(f"梯度提升树 RMSE: {gb_rmse:.4f} | R²: {gb_r2:.4f}")
print(f"神经网络 RMSE: {rmse_nn:.4f} | R²: {r2_nn:.4f}")
print(f"神经网络 (交叉验证) RMSE: {cv_rmse:.4f} | R²: {cv_r2:.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='训练损失')
plt.plot(range(len(val_losses)), val_losses, label='验证损失')
plt.title('训练过程中的损失变化')
plt.xlabel('轮次')
plt.ylabel('均方误差')
plt.legend()
plt.grid(True)
plt.show()

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("特征与房价的相关性热力图")
plt.show()

# 绘制 PCA 累积方差贡献率
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% 方差阈值')
plt.xlabel('主成分数量')
plt.ylabel('累积方差贡献率')
plt.title('PCA 累积方差贡献率')
plt.legend()
plt.grid()
plt.show()

# 绘制各模型预测结果对比图
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_selected)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('线性回归(相关性特征)')

plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_pca)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('线性回归(PCA特征)')

plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('随机森林')

plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred_gb)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('梯度提升树')

plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_nn)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('优化后的神经网络')

plt.tight_layout()
plt.show()

# 绘制神经网络残差图
plt.figure(figsize=(10, 6))
residuals_nn = y_test - y_pred_nn
plt.scatter(y_pred_nn, residuals_nn)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('优化后的神经网络模型：残差图')
plt.show()

# 绘制特征重要性（随机森林模型）
plt.figure(figsize=(12, 6))
feature_importance = pd.Series(rf_model.feature_importances_, index=features_cols)
feature_importance.sort_values(ascending=False).plot(kind='bar')
plt.title('随机森林模型特征重要性')
plt.tight_layout()
plt.show()