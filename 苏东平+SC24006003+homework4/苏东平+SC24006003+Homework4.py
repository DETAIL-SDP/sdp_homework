# 白盒PGD攻击(ε=0.0392)精度: 0.0811
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 新增：导入imgclsmob的CIFAR-10预训练模型
from pytorchcv.model_provider import get_model as ptcv_get_model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 新增：导入进度条库

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CIFAR-10数据预处理（标准化参数根据CIFAR-10统计值）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
denorm = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)

# 加载CIFAR-10测试集
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------- 白盒攻击：PGD实现 ----------------------
# 加载目标模型（ResNet-20，CIFAR-10预训练）
target_model = ptcv_get_model("resnet20_cifar10", pretrained=True).to(device)
target_model.eval()

def pgd_attack(image, target, epsilon, alpha, steps):
    """
    PGD攻击实现(L∞约束)
    """
    perturbed_image = image.clone().detach()
    
    for _ in range(steps):
        # 每次迭代创建新的叶子节点
        current_image = perturbed_image.clone().detach().requires_grad_(True)
        output = target_model(current_image)
        loss = F.nll_loss(output, target)
        
        target_model.zero_grad()
        loss.backward()
        data_grad = current_image.grad.data
        
        # 更新扰动图像
        with torch.no_grad():
            perturbed_image = current_image + alpha * data_grad.sign()
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
    
    return perturbed_image

# 新增：CIFAR-10类别名称映射（提前到visualize_attack函数前）
cifar10_classes = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# 新增可视化函数
def visualize_attack(original, adversarial, epsilon, original_conf, perturbed_conf, original_label, perturbed_label):
    """可视化原始图像、对抗样本及扰动（显示类别名称）"""
    plt.figure(figsize=(14, 4)) 
    
    # 反归一化并转换为numpy
    original_img = denorm(original.squeeze(0)).cpu().detach().numpy().transpose(1,2,0)
    adversarial_img = denorm(adversarial.squeeze(0)).cpu().detach().numpy().transpose(1,2,0)
    noise = (adversarial_img - original_img) * 10  # 放大扰动
    
    # 原始图像（添加类别名称）
    plt.subplot(1,3,1)
    plt.title(f"Original: {cifar10_classes[original_label]}\n置信度: {original_conf:.2f}") 
    plt.imshow(np.clip(original_img, 0, 1))
    plt.axis('off')
    
    # 对抗样本（添加类别名称）
    plt.subplot(1,3,2)
    plt.title(f"Adversarial (ε={epsilon:.4f}): {cifar10_classes[perturbed_label]}\n置信度: {perturbed_conf:.2f}") 
    plt.imshow(np.clip(adversarial_img, 0, 1))
    plt.axis('off')
    
    # 扰动（添加数值范围标签）
    plt.subplot(1,3,3)
    plt.title(f"Perturbation (×10)\n范围: [{noise.min():.2f}, {noise.max():.2f}]") 
    plt.imshow(noise)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'attack_vis_{epsilon:.4f}_{cifar10_classes[original_label]}.png', dpi=300)
    plt.close()

global_epsilons = []
global_success_rates = []

def visualize_global_stats(confidence_diffs, attack_success_rates, epsilons):
    """全局统计可视化（置信度分布+多ε对比）"""
    plt.figure(figsize=(16, 6))
    
    # 置信度下降分布直方图
    plt.subplot(1,2,1)
    plt.hist(confidence_diffs, bins=20, color='skyblue')
    plt.title("Adversarial Sample Confidence Degradation Distribution")#对抗样本置信度下降分布
    plt.xlabel("Original Confidence - Adversarial Sample Confidence")#原始置信度 - 对抗样本置信度
    plt.ylabel("Sample Size")#样本数量
    
    # 多ε值攻击成功率对比
    plt.subplot(1,2,2)
    plt.plot(epsilons, attack_success_rates, 'o-', color='coral')
    plt.title("Attack success rates for different ε values")#不同ε值的攻击成功率
    plt.xlabel("The value of ε")#ε值
    plt.ylabel("Attack success rate (model correct classification rate)")#攻击成功率（模型正确分类率）
    plt.xticks(epsilons)
    
    plt.tight_layout()
    plt.savefig('global_attack_stats2.png', dpi=300)
    plt.close()

# 修改测试函数（以白盒测试为例）
def white_box_test(epsilon=10/255, alpha=4/255, steps=20, collect_stats=True):  # 修改参数：增大epsilon、alpha和steps
    correct = 0 
    total_attacked = 0
    confidence_diffs = []  
# 原代码定义了两个列表用于存储原始标签和扰动后的标签，若在后续代码中未使用这两个列表，它们可能是无用的。
# 若确实无用，可直接注释掉这两行代码。
# original_labels = [] 
# perturbed_labels = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="白盒攻击进度")
    for i, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        output = target_model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        
        total_attacked += 1
        perturbed_data = pgd_attack(data, target, epsilon, alpha, steps) 
        
        # 收集标签信息（新增）
        original_label = target.item()
        perturbed_output = target_model(perturbed_data)
        perturbed_label = perturbed_output.max(1, keepdim=True)[1].item()
        
        # 收集置信度差异（新增）
        original_conf = F.softmax(output, dim=1).max().item()
        perturbed_conf = F.softmax(perturbed_output, dim=1).max().item()
        confidence_diffs.append(original_conf - perturbed_conf)
        
        # 触发单样本可视化（修改参数）
        if i < 10:
            visualize_attack(
                data, perturbed_data, epsilon,
                original_conf, perturbed_conf,
                original_label, perturbed_label  
            )
        
        if perturbed_label == original_label: 
            correct += 1
        
        pbar.set_postfix({"当前精度": f"{correct/total_attacked:.4f}"})
    
    final_acc = correct / total_attacked if total_attacked !=0 else 0
    print(f"白盒PGD攻击(ε={epsilon:.4f})精度: {final_acc:.4f}")
    

    if collect_stats:
        global_epsilons.append(epsilon)
        global_success_rates.append(final_acc)
        if len(global_epsilons) >= 3:  
            visualize_global_stats(confidence_diffs, global_success_rates, global_epsilons)
    
    return final_acc


# ---------------------- 黑盒攻击：替代模型实现 ----------------------
# 加载替代模型（更改为pytorchcv支持的CIFAR-10预训练模型）
surrogate_model = ptcv_get_model("densenet40_k12_cifar10", pretrained=True).to(device)
surrogate_model.train()  # 需要训练模式计算梯度
optimizer = optim.Adam(surrogate_model.parameters(), lr=0.001)

def black_box_attack(image, target, epsilon, alpha, steps):
    """基于替代模型的黑盒攻击(FGSM变体)"""
    perturbed_image = image.clone().detach()  # 初始为叶子张量
    
    for _ in range(steps):
        # 每次迭代时，创建新的叶子张量并设置requires_grad=True
        perturbed_image = perturbed_image.detach().clone()  # 确保是叶子张量
        perturbed_image.requires_grad = True  # 安全修改叶子张量的属性
        
        # 使用替代模型计算梯度
        output = surrogate_model(perturbed_image)
        loss = F.nll_loss(output, target)
        
        surrogate_model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        
        # 生成扰动（使用无梯度上下文更新）
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * data_grad.sign()
            delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + delta, 0, 1)
    
    return perturbed_image

def black_box_test(epsilon=8/255, alpha=2/255, steps=10):
    correct = 0
    confidence_diffs = []  # 新增：置信度差异
    perturbation_norms = []  # 新增：扰动范数
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        output = target_model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        
        perturbed_data = black_box_attack(data, target, epsilon, alpha, steps)
        
        # 新增可视化样本（前5个）
        if len(confidence_diffs) < 5:
            original_conf = F.softmax(output, dim=1).max().item()
            perturbed_output = target_model(perturbed_data)
            perturbed_conf = F.softmax(perturbed_output, dim=1).max().item()
            visualize_attack(
                data, perturbed_data, epsilon,
                original_conf, perturbed_conf,
                target.item(), perturbed_output.argmax().item()
            )
        
        # 收集统计信息
        delta = (perturbed_data - data).norm(p=float('inf')).item()
        perturbation_norms.append(delta)
        confidence_diffs.append(original_conf - perturbed_conf)
        
        # 攻击目标模型
        output = target_model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
    
    # 新增黑盒攻击可视化
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(perturbation_norms, bins=20, color='lightgreen')
    plt.title("Black-box Attack Perturbation Distribution (L∞)")#黑盒攻击扰动分布(L∞)
    plt.xlabel("Perturbation magnitude")#扰动大小
    
    plt.subplot(1,2,2)
    plt.scatter(perturbation_norms, confidence_diffs, alpha=0.6)
    plt.title("Perturbation magnitude vs. Confidence degradation")#扰动大小 vs 置信度下降
    plt.xlabel("L∞ Norm")
    plt.ylabel("Confidence Drop")#置信度下降
    
    plt.tight_layout()
    plt.savefig(f'blackbox_stats_{epsilon:.4f}.png', dpi=300)
    plt.close()
    
    final_acc = correct / len(test_loader)
    print(f"黑盒攻击(替代模型DenseNet40)精度: {final_acc:.4f}")
    return final_acc

# 执行测试（示例参数）
if __name__ == "__main__":
    # 测试不同ε值（触发多ε对比图）
    for eps in [6/255, 8/255, 10/255]:
        white_box_test(epsilon=eps, alpha=eps/5, steps=20)  #steps=20
    
    # 黑盒攻击测试保持不变
    black_box_test(epsilon=8/255, alpha=4/255, steps=20) # alpha=2/255，原steps=5
