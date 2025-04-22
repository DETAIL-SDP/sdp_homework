# -*- coding: utf-8 -*-
# -homework2,4.21,no1-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
dataset_path = r"C:\Users\苏东平\Desktop\cifar-10-batches-py"
train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # 展平特征图
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 初始化模型
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    train_acc = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每100个batch打印一次
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 计算整个epoch的准确率和损失
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_acc

# 4. 测试函数
def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0.0
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%')
        
        # 打印分类报告
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds, target_names=classes))
    
    return test_loss, test_acc, all_labels, all_preds

# 5. 可视化函数
def plot_results(train_losses, train_acc, test_loss, test_acc):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 6. 主程序
if __name__ == '__main__':
    # 训练模型
    print("Starting training...")
    train_losses, train_acc = train_model(model, train_loader, criterion, optimizer, epochs=15)
    
    # 测试模型
    print("\nTesting model...")
    test_loss, test_acc, all_labels, all_preds = test_model(model, test_loader)
    
    # 可视化结果
    plot_results(train_losses, train_acc, test_loss, test_acc)