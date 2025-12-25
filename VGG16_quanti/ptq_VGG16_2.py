import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
import numpy as np
import os
import time
from torch.quantization import QuantStub, DeQuantStub # 引入量化模块

# ==========================================
# 1. Config 模块
# ==========================================
class Config:
    def __init__(self):
        self.device = 'cpu' # 量化推理通常在 CPU 上进行验证
        self.data_root = "../dataset" 
        self.load_model_path = "./record/result_2025-12-23_01-00/best_model.ckpt" # 请确认路径
        self.img_size = (32, 32)
        self.batch_size = 128
        self.num_workers = 0 
        self.num_classes = 10
        self.seed = 0

# ==========================================
# 2. Utils 模块
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fuse_model(model):
    """算子融合: Conv+BN+ReLU -> ConvReLU"""
    modules_to_fuse = []
    # 遍历 features 序列，寻找 Conv + BN + ReLU 的组合
    for i in range(len(model.features) - 2):
        if (isinstance(model.features[i], nn.Conv2d) and
            isinstance(model.features[i+1], nn.BatchNorm2d) and
            isinstance(model.features[i+2], nn.ReLU)):
            modules_to_fuse.append([f'features.{i}', f'features.{i+1}', f'features.{i+2}'])
    
    # 调用 PyTorch 的工具进行融合
    if modules_to_fuse:
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        print(f"Fused {len(modules_to_fuse)} groups of modules.")
    return model

def measure_inference_latency(model, input_data, num_runs=100, num_warmup=10):
    """测量推理延迟"""
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
    return (end_time - start_time) / num_runs * 1000 # 转换为毫秒

def evaluate_accuracy(model, loader, device='cpu'):
    """评估准确率"""
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

# ==========================================
# 3. Data 模块
# ==========================================
def construct_dataloaders(cfg):
    print("Initializing Data Module...")
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
    ])
    # 只加载验证集用于量化校准和测试
    valid_set = CIFAR10(root=cfg.data_root, train=False, transform=transform, download=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return valid_loader

# ==========================================
# 4. Model 模块 (修改版)
# ==========================================
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.quant = QuantStub()       # <--- [新增] 量化入口
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
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
        self.dequant = DeQuantStub()   # <--- [新增] 反量化出口

    def forward(self, x):
        x = self.quant(x)              # <--- 1. 入口
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)            # <--- 2. 出口
        return x

# ==========================================
# 5. Main (量化流程)
# ==========================================
if __name__ == '__main__':
    cfg = Config()
    set_seed(cfg.seed)
    
    # 1. 准备数据和模型
    val_loader = construct_dataloaders(cfg)
    model_fp32 = VGG16(num_classes=cfg.num_classes)
    
    # 加载权重
    if os.path.exists(cfg.load_model_path):
        print(f"Loading checkpoint from {cfg.load_model_path}...")
        ckpt = torch.load(cfg.load_model_path, map_location='cpu') # 强制加载到 CPU
        model_fp32.load_state_dict(ckpt)
    else:
        print("[Warning] Checkpoint not found. Using random weights for demonstration.")

    model_fp32.to('cpu')
    model_fp32.eval()

    # 融合模型
    model_fp32 = fuse_model(model_fp32)
    # print("feature 0:", model_fp32.features[0])

    # 2. 设置量化配置 (Symmetric, ZP=0)
    # -------------------------------------------------------
    my_qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8, 
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric  # <--- [关键] 对称量化
        )
    )
    model_fp32.qconfig = my_qconfig
    
    # 3. 准备 (Prepare) - 插入 Observer
    print("\n[Step 1] Preparing model with Observers...")
    torch.quantization.prepare(model_fp32, inplace=True)

    # 4. 校准 (Calibration)
    print("[Step 2] Calibrating (feeding data to Observers)...")
    calibration_batches = 32
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(val_loader, total=calibration_batches)):
            if i >= calibration_batches: break
            model_fp32(imgs) # 前向传播，收集统计信息

    # 5. 转换 (Convert) - 真正转为 INT8
    print("[Step 3] Converting to INT8 model...")
    # inplace=False 保留 FP32 模型用于对比
    model_int8 = torch.quantization.convert(model_fp32, inplace=False) 

    # ==========================================
    # 6. 结果对比 (Benchmark)
    # ==========================================
    print("\n" + "="*40)
    print("Result Comparison (CPU)")
    print("="*40)
    
    # 这里的 FP32 模型因为已经被 prepare 过了，含有了 Observer，
    # 为了公平对比，我们通常重新实例化一个干净的 FP32 模型，或者在这里忽略 Observer 的开销。
    # 为简单起见，我们直接测试。
    
    dummy_input = torch.randn(1, 3, 32, 32).to('cpu')

    # A. 延迟测试
    print("Measuring Latency...")
    fp32_lat = measure_inference_latency(model_fp32, dummy_input)
    int8_lat = measure_inference_latency(model_int8, dummy_input)
    
    # B. 准确率测试
    print("Measuring Accuracy...")
    fp32_acc = evaluate_accuracy(model_fp32, val_loader)
    int8_acc = evaluate_accuracy(model_int8, val_loader)

    print(f"\nMetric\t\t| FP32\t\t| INT8")
    print("-" * 40)
    print(f"Accuracy (%)\t| {fp32_acc:.2f}%\t| {int8_acc:.2f}%")
    print(f"Latency (ms)\t| {fp32_lat:.2f} ms\t| {int8_lat:.2f} ms")
    print("-" * 40)
    
    speedup = fp32_lat / int8_lat
    acc_drop = fp32_acc - int8_acc
    print(f"\nAnalysis:")
    print(f"  >>> Speedup: {speedup:.2f}x")
    print(f"  >>> Accuracy Drop: {acc_drop:.2f}%")