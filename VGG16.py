import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# ==========================================
# 1. 设置设备 (Device Configuration)
# ==========================================
# 检查是否有 GPU (NVIDIA) 或 MPS (Mac M1/M2)，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"正在使用的计算设备: {device}")

# ==========================================
# 2. 数据准备与预处理 (Data Preparation)
# ==========================================
# VGG16 标准输入通常是 224x224。
# 虽然 CIFAR-10 是 32x32，为了利用预训练权重，我们这里将其 Resize 到 224。
# 注意：这会增加显存占用和计算时间。
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),      # 调整图像大小以适配 VGG
    transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转
    transforms.ToTensor(),              # 转为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载数据集 (会自动下载到 ./data 文件夹)
print("正在加载数据集...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 定义 DataLoader
batch_size = 32 # 如果显存不够，可以调小这个数字，例如 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==========================================
# 3. 模型搭建 (Model Setup)
# ==========================================
print("正在加载 VGG16 模型...")
# 使用预训练模型 (pretrained=True)，这意味着模型已经在大规模数据集上学到了提取特征的能力
# 这种方法叫“迁移学习”，非常适合初学者快速获得好结果
model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

# 冻结特征提取层的参数 (可选)
# 如果不想训练前面的卷积层，只训练最后的分类层，可以取消下面两行的注释
# for param in model.features.parameters():
#     param.requires_grad = False

# 修改最后的全连接层 (Classifier)
# VGG16 原始输出是 1000 类 (ImageNet)，我们需要改为 CIFAR-10 的 10 类
num_features = model.classifier[6].in_features # 获取最后一层的输入维度 (4096)
model.classifier[6] = nn.Linear(num_features, 10) # 替换为输出 10 的全连接层

model = model.to(device) # 将模型移动到 GPU/MPS

# ==========================================
# 4. 定义损失函数与优化器 (Loss & Optimizer)
# ==========================================
criterion = nn.CrossEntropyLoss() # 分类任务标准损失函数
# 学习率设为 0.001，使用动量 SGD
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ==========================================
# 5. 训练与验证循环 (Training & Validation Loop)
# ==========================================
def train_model(num_epochs=5):
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} 开始...')
        
        # --- 训练阶段 ---
        model.train() # 设置为训练模式 (启用 Dropout, Batch Norm 更新)
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. 梯度清零
            optimizer.zero_grad()

            # 2. 前向传播
            outputs = model(inputs)

            # 3. 计算损失
            loss = criterion(outputs, labels)

            # 4. 反向传播
            loss.backward()

            # 5. 参数更新
            optimizer.step()

            running_loss += loss.item()
            
            # 每 100 个 batch 打印一次日志
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # --- 验证阶段 ---
        # 每个 Epoch 结束后在测试集上跑一次，看看效果
        model.eval() # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad(): # 验证阶段不需要计算梯度，节省显存
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Epoch {epoch+1} 结束. 测试集准确率: {acc:.2f}%')

    end_time = time.time()
    print(f'\n训练完成! 总耗时: {(end_time - start_time)/60:.2f} 分钟')

# ==========================================
# 6. 开始运行
# ==========================================
if __name__ == '__main__':
    # 为了演示，我们只训练 2-3 个 Epoch 即可看到效果
    # 如果在 CPU 上运行，建议将 Epoch 设为 1，或者减少数据集大小进行测试
    train_model(num_epochs=3)
    
    # 保存模型
    torch.save(model.state_dict(), 'vgg16_cifar10.pth')
    print("模型已保存为 vgg16_cifar10.pth")