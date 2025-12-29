import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.quantization import QuantStub, DeQuantStub
from tqdm.auto import tqdm
import numpy as np
import os
import time

# ==========================================
# 1. Config 配置
# ==========================================
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_root = "../dataset"
        # 请确保这里有预训练好的 FP32 权重路径
        self.load_model_path = "./record/result_2025-12-23_01-00/best_model.ckpt" 
        self.img_size = (32, 32)
        self.batch_size = 128
        self.num_workers = 0
        self.num_classes = 10
        self.seed = 2024
        self.qat_epochs = 3        # QAT 微调轮数
        self.qat_lr = 1e-4         # QAT 微调学习率 (通常比正常训练低)

# ==========================================
# 2. Utils 工具函数
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def measure_inference_latency(model, input_data, num_runs=100, num_warmup=10):
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(num_warmup):
            model(input_data)
        # 计时
        start_time = time.time()
        for _ in range(num_runs):
            model(input_data)
        end_time = time.time()
    return (end_time - start_time) / num_runs * 1000  # ms

def evaluate_accuracy(model, loader, device='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def fuse_model(model):
    """自动融合 Conv2d + BatchNorm2d + ReLU6"""
    modules_to_fuse = []
    for i in range(len(model.features) - 2):
        if (isinstance(model.features[i], nn.Conv2d) and
            isinstance(model.features[i+1], nn.BatchNorm2d) and
            isinstance(model.features[i+2], nn.ReLU6)): # 注意这里是 ReLU6
            modules_to_fuse.append([f'features.{i}', f'features.{i+1}'])
    
    if modules_to_fuse:
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        print(f"[Fusion] Fused {len(modules_to_fuse)} groups of Conv+BN+ReLU6.")
    return model

def run_qat_training(model, train_loader, device, epochs, lr):
    """QAT 微调循环"""
    print(f"\n[QAT] Start Fine-tuning for {epochs} epochs...")
    model.to(device)
    model.train()  # 必须是 train 模式，伪量化节点才能更新
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{epochs}", leave=False)
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': total_loss / (total/train_loader.batch_size), 'Acc': 100.*correct/total})
        
        print(f"  >>> Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")
    
    print("[QAT] Fine-tuning done.")
    return model

# ==========================================
# 3. Model 定义 (修改版)
# ==========================================
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.quant = QuantStub()       # 量化入口
        self.features = nn.Sequential(
            # 使用 ReLU6 替代 ReLU
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU6(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU6(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.dequant = DeQuantStub()   # 反量化出口

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# ==========================================
# 4. Main 主程序
# ==========================================
if __name__ == '__main__':
    cfg = Config()
    set_seed(cfg.seed)
    
    # -------------------------------------------------------
    # 1. 准备数据 (Train 用于 QAT 微调, Valid 用于测试)
    # -------------------------------------------------------
    print("Initializing Data Module...")
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
    ])
    # 这里的 train=True 是必须的，因为 QAT 需要反向传播
    train_set = CIFAR10(root=cfg.data_root, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    valid_set = CIFAR10(root=cfg.data_root, train=False, transform=transform, download=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # -------------------------------------------------------
    # 2. 准备模型
    # -------------------------------------------------------
    print("Initializing Model...")
    model = VGG16(num_classes=cfg.num_classes)
    
    # 加载预训练权重 (FP32)
    if os.path.exists(cfg.load_model_path):
        print(f"Loading checkpoint from {cfg.load_model_path}...")
        ckpt = torch.load(cfg.load_model_path, map_location='cpu')
        model.load_state_dict(ckpt)
    else:
        print("[Warning] Checkpoint not found! Training from scratch (random weights).")

    model.to(cfg.device)

    # -------------------------------------------------------
    # 3. 算子融合 (Fusion)
    # -------------------------------------------------------
    model.eval()  # 融合前通常先切换到 eval
    model = fuse_model(model)

    # -------------------------------------------------------
    # 4. 设置 QAT 配置 (prepare_qat)
    # -------------------------------------------------------
    # 使用针对 QAT 优化的 backend 设置
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    print("[Step 1] Preparing model for QAT (Inserting FakeQuantize nodes)...")
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)

    # -------------------------------------------------------
    # 5. QAT 微调 (Fine-tuning)
    # -------------------------------------------------------
    print("[Step 2] Running Quantization-Aware Training...")
    # 使用 GPU 进行训练速度更快
    model = run_qat_training(model, train_loader, device=cfg.device, epochs=cfg.qat_epochs, lr=cfg.qat_lr)

    # -------------------------------------------------------
    # 6. 转换 (Convert) -> INT8
    # -------------------------------------------------------
    print("[Step 3] Converting to CPU INT8 model...")
    model.eval()
    model.to('cpu') # 转换必须在 CPU 上进行
    model_int8 = torch.quantization.convert(model, inplace=False)

    # -------------------------------------------------------
    # 7. 最终 Benchmark
    # -------------------------------------------------------
    print("\n" + "="*40)
    print("Final Benchmark Comparison")
    print("="*40)
    
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')

    # 为了公平对比，我们重新加载一个干净的 FP32 模型（不做 QAT 的版本）
    # 或者直接对比 QAT 后的模型（尚未 convert）和最终 int8 模型
    # 这里我们对比 convert 后的 int8 模型和之前的 baseline
    
    print("Measuring INT8 Performance...")
    int8_lat = measure_inference_latency(model_int8, dummy_input)
    int8_acc = evaluate_accuracy(model_int8, valid_loader, device='cpu')
    
    print(f"\nFinal Results:")
    print(f"INT8 Accuracy: {int8_acc:.2f}%")
    print(f"INT8 Latency:  {int8_lat:.4f} ms")
    
    # 提示：你可以记录之前的 FP32 数据来进行 print 对比