import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from tqdm.auto import tqdm
import os
import copy

# 导入你源码中的配置和数据加载模块
try:
    from VGG16_quanti import Config, VGG16, construct_dataloaders
except ImportError:
    print("请确保 VGG16_quanti.py 在当前目录下")
    exit()

# ==========================================
# 1. 定义量化版 VGG16 (插入探针)
# ==========================================
class QuantizableVGG16(VGG16):
    def __init__(self, num_classes=10):
        super(QuantizableVGG16, self).__init__(num_classes)
        # [关键] 插入 QuantStub (将 FP32 输入转为 Int8)
        self.quant = QuantStub()
        # [关键] 插入 DeQuantStub (将 Int8 输出转回 FP32)
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 1. 进入 Int8 域
        x = self.quant(x)
        
        # 2. 执行卷积 (此时全是 Int8 运算)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # 3. 回到 FP32 域
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """算子融合: Conv+BN+ReLU -> ConvReLU"""
        print("[Fusion] 正在融合算子...")
        self.eval()
        
        # 融合 Features (Conv + BN + ReLU)
        # 根据 VGG16 结构手动指定索引
        fusion_layers = []
        # Block 1
        fusion_layers.append(['0', '1', '2'])
        fusion_layers.append(['3', '4', '5'])
        # Block 2
        fusion_layers.append(['7', '8', '9'])
        fusion_layers.append(['10', '11', '12'])
        # Block 3
        fusion_layers.append(['14', '15', '16'])
        fusion_layers.append(['17', '18', '19'])
        fusion_layers.append(['20', '21', '22'])
        # Block 4
        fusion_layers.append(['24', '25', '26'])
        fusion_layers.append(['27', '28', '29'])
        fusion_layers.append(['30', '31', '32'])
        # Block 5
        fusion_layers.append(['34', '35', '36'])
        fusion_layers.append(['37', '38', '39'])
        fusion_layers.append(['40', '41', '42'])

        for module_indices in fusion_layers:
            fuse_modules(self.features, module_indices, inplace=True)

        # 融合 Classifier (Linear + ReLU)
        fuse_modules(self.classifier, ['0', '1'], inplace=True)
        fuse_modules(self.classifier, ['3', '4'], inplace=True)

# ==========================================
# 2. PTQ 主流程
# ==========================================
def run_ptq(cfg):
    # -------------------------------------------
    # Step 0: 准备
    # -------------------------------------------
    # 注意: PyTorch 量化目前主要在 CPU 上运行 (FBGEMM)
    device = 'cpu' 
    
    # 加载浮点模型
    model_fp32 = QuantizableVGG16(num_classes=cfg.num_classes)
    if os.path.exists(cfg.load_model_path):
        state_dict = torch.load(cfg.load_model_path, map_location='cpu')
        model_fp32.load_state_dict(state_dict, strict=False)
    else:
        print("Error: 找不到模型权重")
        return

    model_fp32.to(device)
    model_fp32.eval()
    
    # 融合算子 (这一步很重要，能提升速度和精度)
    model_fp32.fuse_model()

    # -------------------------------------------
    # Step 1: 配置量化后端 (关键!)
    # -------------------------------------------
    # Windows/Linux x86 使用 'fbgemm'
    # ARM (Mac M1/M2, 手机) 使用 'qnnpack'
    backend = 'fbgemm' 
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    print(f"\n[Config] 使用后端: {backend}")

    # -------------------------------------------
    # Step 2: 插入 Observer (Prepare)
    # -------------------------------------------
    print("[Prepare] 插入 Observer...")
    torch.ao.quantization.prepare(model_fp32, inplace=True)

    # -------------------------------------------
    # Step 3: 校准 (Calibrate)
    # -------------------------------------------
    print("[Calibrate] 正在校准 (喂入数据)...")
    _, val_loader = construct_dataloaders(cfg)
    # 校准只需要少量数据
    calibration_batches = 32
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(val_loader, total=calibration_batches)):
            if i >= calibration_batches: break
            model_fp32(imgs) # 前向传播，Observer 会自动记录数据分布

    # -------------------------------------------
    # Step 4: 转换 (Convert) -> 真正的 Int8 模型
    # -------------------------------------------
    print("\n[Convert] 转换为 Int8 模型...")
    # 这一步会将 nn.Conv2d 替换为 nn.quantized.Conv2d
    model_int8 = torch.ao.quantization.convert(model_fp32, inplace=False)

    # -------------------------------------------
    # Step 5: 检查结构
    # -------------------------------------------
    print("\n[Inspect] 检查第一层卷积:")
    # 注意：融合后层级结构可能变了，通常在 features[0]
    layer0 = model_int8.features[0]
    print(f"Layer Type: {type(layer0)}") # 应该是 <class 'torch.ao.nn.quantized.modules.conv.Conv2d'>
    print(f"Weight Dtype: {layer0.weight().dtype}") # 应该是 torch.qint8
    print(f"Int8 Weight Shape: {layer0.weight().shape}")
    
    # -------------------------------------------
    # Step 6: 评估精度
    # -------------------------------------------
    print("\n[Eval] 评估 Int8 模型精度...")
    acc = evaluate(model_int8, val_loader, device)
    print(f"Int8 Accuracy: {acc:.2f}%")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    cfg = Config()
    cfg.device = 'cpu' # 强制 CPU
    run_ptq(cfg)