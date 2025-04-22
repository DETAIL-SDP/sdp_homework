# -*- coding: utf-8 -*-
# -homework2,4.21,no4-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import time
import csv

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 创建结果保存目录
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(os.getcwd(), 'results', timestamp)
os.makedirs(results_dir, exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
dataset_path = r"C:\Users\苏东平\Desktop\cifar-10-batches-py"
train_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 32  # 减小batch size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 减少通道数
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 减少通道数
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 减少通道数
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # 减少通道数
        self.bn4 = nn.BatchNorm2d(64)
        # 添加SE注意力模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64 // 8, 1),  # 调整SE模块通道数
            nn.ReLU(),
            nn.Conv2d(64 // 8, 64, 1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 调整全连接层大小
        self.fc2 = nn.Linear(128, 10)  # 修正输出维度
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # 应用SE注意力
        se_weight = self.se(x)
        x = x * se_weight
        x = self.pool(x)
        
        # 展平特征图
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=12):
    model.train()
    train_losses = []
    train_acc = []
    best_acc = 0.0
    patience = 5
    patience_counter = 0
    accumulation_steps = 2  # 梯度累积步数
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 缩放损失
            
            # 反向传播和优化
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:  # 每accumulation_steps步更新一次
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计信息
            running_loss += loss.item() * accumulation_steps  # 恢复原始损失值
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每100个batch打印一次
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 处理最后一个不完整的accumulation_steps
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 计算整个epoch的准确率和损失
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # 早停机制
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        # 调用学习率调度器
        scheduler.step(epoch_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%, LR: {lr:.6f}')
    
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
    fig = plt.figure(figsize=(12, 5))
    
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
    fig.savefig(os.path.join(results_dir, 'loss_acc.png'))
    plt.show()

# 5.1 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.show()

# 5.2 展示示例预测结果
def show_sample_predictions(model, test_loader, classes, device, num_images=8):
    model.eval()
    images, labels = next(iter(test_loader))
    inputs = images.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    images = images.numpy()
    plt.figure(figsize=(12, 6))
    for idx in range(num_images):
        plt.subplot(2, num_images//2, idx+1)
        img = np.transpose(images[idx], (1, 2, 0))
        img = img * 0.5 + 0.5  # 反归一化
        plt.imshow(img)
        plt.title(f"P:{classes[preds[idx]]}\nT:{classes[labels[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sample_predictions.png'))
    plt.show()

# 6. 主程序
if __name__ == '__main__':
    # 训练模型
    print("Starting training...")
    train_losses, train_acc = train_model(model, train_loader, criterion, optimizer, epochs=20)
    
    # 测试模型
    print("\nTesting model...")
    test_loss, test_acc, all_labels, all_preds = test_model(model, test_loader)
    
    # 保存训练日志
    with open(os.path.join(results_dir, 'train_log.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_acc'])
        for epoch, (l, a) in enumerate(zip(train_losses, train_acc), start=1):
            writer.writerow([epoch, l, a])
    # 保存模型
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    # 保存分类报告
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    # 可视化结果
    plot_results(train_losses, train_acc, test_loss, test_acc)
    # 混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, classes)
    # 示例预测
    show_sample_predictions(model, test_loader, classes, device)