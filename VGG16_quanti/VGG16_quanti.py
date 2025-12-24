import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time


# ==========================================
# 1. Config 模块 (参数配置寄存器)
# ==========================================
class Config:
    def __init__(self):
        # --- 自动归档设置 ---
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.timestamp = current_time
        self.output_dir = f"./record/result_{current_time}"
        self.ckpt_name = "best_model.ckpt"
        
        # --- 模型加载设置 ---
        #  "./record/result_2025-12-23_01-00/best_model.ckpt"
        self.load_model_path = "./record/result_2025-12-23_01-00/best_model.ckpt"
        
        # --- 基础环境 ---
        self.seed = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_root = "../dataset" 
        
        # --- 训练参数 ---
        self.img_size       = (32, 32)
        self.num_classes    = 10
        self.n_epochs       = 50
        self.batch_size     = 128
        self.learning_rate  = 0.0002
        self.weight_decay   = 1e-5
        self.num_workers    = 0  # 数据加载线程数
        self.optimizer      = 'Adam'
        self.optim_hparams  = {'lr': self.learning_rate, 'weight_decay': self.weight_decay}
        
        # --- 功能开关 ---
        self.use_augmentation = False # 是否开启训练集数据增强
        self.earlystop = 10 # Early Stopping (多少个 epoch 没提升就停止)


    def create_output_dir(self):
        """创建文件夹并备份参数"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        config_path = os.path.join(self.output_dir, "config.txt")
        with open(config_path, 'w') as f:
            f.write(f"Experiment Log - {self.timestamp}\n")
            f.write("========================================\n")
            # 遍历 Config 类中所有的属性并写入
            for key, value in self.__dict__.items():
                f.write(f"{key}: {value}\n")
            f.write("========================================\n")
        print(f"Config saved to {config_path}")

# ==========================================
# 2. Utils 模块 (工具箱)
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_curves(record, output_dir):
    """保存波形图"""
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(record['train_loss'], label='Train Loss', c='tab:orange')
    plt.plot(record['val_loss'], label='Val Loss', c='tab:blue')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend() 
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(record['train_acc'], label='Train Acc', c='tab:orange')
    plt.plot(record['val_acc'], label='Val Acc', c='tab:blue')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "acc_curve.png"))
    plt.close()

def print_model_weights_info(model):
    """打印模型权重统计信息"""
    print("\n" + "="*100)
    print(f"{'Layer Name':<30} | {'Shape':<20} | {'Params':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 100)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:<30} | {str(list(param.shape)):<20} | {num_params:<10} | {param.data.mean():.4f}     | {param.data.std():.4f}     | {param.data.min():.4f}     | {param.data.max():.4f}")
    print("-" * 100)
    print(f"Total Trainable Params: {total_params}")
    print("="*100 + "\n")

# ==========================================
# 3. Data 模块 (I/O 接口与信号处理)
# ==========================================
def get_transforms(mode, img_size):
    """
    定义数据增强策略 
    mode: 'train' or 'test' or 'default'
    """
    if mode == 'train':
        # 这里完整保留了你之前的高级数据增强策略
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomChoice(
                [transforms.AutoAugment(),
                 transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                 transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.ToTensor(),
        ])
    else:
        # 验证/测试集只需要 Resize 和 ToTensor
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

def construct_dataloaders(cfg):
    """
    数据加载器构造函数
    类似于 Verilog 中的 Test Pattern Generator
    """
    print("Initializing Data Module...")
    
    # 1. 获取变换策略
    if cfg.use_augmentation:
        train_tfm = get_transforms('train', cfg.img_size)
        test_tfm = get_transforms('test', cfg.img_size)
    else:
        train_tfm = get_transforms('default', cfg.img_size)
        test_tfm = get_transforms('default', cfg.img_size)

    # 2. 读取数据集
    # 使用 torchvision 自带的 CIFAR10 数据集
    train_set = CIFAR10(root=cfg.data_root, train=True, transform=train_tfm)
    valid_set = CIFAR10(root=cfg.data_root, train=False, transform=test_tfm)

    # 3. 封装为 Loader
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_set, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    print(f"Data Module Ready. Train set: {len(train_set)}, Val set: {len(valid_set)}")
    return train_loader, valid_loader

# ==========================================
# 4. Model 模块 (DUT)
# ==========================================
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1 3*32*32->64*32*32->64*16*16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2 64*16*16->128*16*16->128*8*8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3 128*8*8->256*8*8->256*4*4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 256*4*4->512*4*4->512*2*2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5 512*2*2->512*2*2->512*1*1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 512*1*1->4096->4096->num_classes
        self.avgpool = nn.AvgPool2d(kernel_size=1,stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 5. Trainer 模块 (仿真控制平台)
# ==========================================
class Trainer:
    def __init__(self, config, model, train_loader, val_loader):
        self.cfg = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optim_hparams)
        self.best_acc = 0.0
        self.record = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(self):
        self.model.train()
        total_loss = []
        total_acc = []
        for imgs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
            # Zero the gradients.
            self.optimizer.zero_grad()
            # Forward the data.
            logits = self.model(imgs)
            # Calculate the loss.
            loss = self.criterion(logits, labels)
            # Backpropagate the loss.
            loss.backward()
            # Clip the gradient norms for stable training.
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters.
            self.optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            total_loss.append(loss.item())
            total_acc.append(acc.item())
        return sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)

    def validate(self):
        # set model to evalutation mode
        self.model.eval()
        total_loss = []
        total_acc = []
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                total_loss.append(loss.item())
                total_acc.append((logits.argmax(dim=-1) == labels).float().mean().item())
        return sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)

    def run(self):
        print(f"Start Training. Results -> {self.cfg.output_dir}")
        stale = 0 # 计数器：记录有多少个 epoch 没有提升
        
        for epoch in range(self.cfg.n_epochs):
            t_loss, t_acc = self.train_epoch()
            v_loss, v_acc = self.validate()
            
            self.record['train_loss'].append(t_loss)
            self.record['train_acc'].append(t_acc)
            self.record['val_loss'].append(v_loss)
            self.record['val_acc'].append(v_acc)
            
            print(f"[{epoch+1:03d}/{self.cfg.n_epochs}] Train Loss:{t_loss:.4f}, Acc:{t_acc:.4f} | Valid Loss:{v_loss:.4f}, Acc:{v_acc:.4f}")
            
            if v_acc > self.best_acc:
                self.best_acc = v_acc
                stale = 0 # 重置计数器
                save_path = os.path.join(self.cfg.output_dir, self.cfg.ckpt_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> Best saved: {v_acc:.4f}")
            else:
                stale += 1
                if stale > self.cfg.earlystop:
                    print(f"No improvement for {self.cfg.earlystop} epochs. Early stopping...")
                    break
                    
        return self.record

# ==========================================
# 6. Main (Top Module)
# ==========================================
if __name__ == '__main__':
    # 1. 实例化配置
    cfg = Config()
    cfg.create_output_dir()
    set_seed(cfg.seed)
    
    try:
        # 2. 准备数据 
        train_loader, val_loader = construct_dataloaders(cfg)
        
        # 3. 实例化模型 
        model = VGG16(num_classes=cfg.num_classes)
        
        # --- 加载预训练模型 ---
        # if cfg.load_model_path is not None:
        #     if os.path.exists(cfg.load_model_path):
        #         print(f"Loading model checkpoint from {cfg.load_model_path}...")
        #         # 加载权重
        #         ckpt = torch.load(cfg.load_model_path, map_location=cfg.device)
        #         model.load_state_dict(ckpt)
        #         print("Model loaded successfully!")
        #     else:
        #         print(f"[Warning] Model path {cfg.load_model_path} does not exist. Training from scratch.")
        
        # 打印模型权重信息
        print_model_weights_info(model)

        # # 4. 运行训练 
        # trainer = Trainer(cfg, model, train_loader, val_loader)
        # record = trainer.run()
        
        # # 5. 导出波形
        # save_curves(record, cfg.output_dir)
        # print("Done.")
        
    except Exception as e:
        print(f"\n[Error]: {e}")