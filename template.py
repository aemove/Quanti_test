import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from PIL import Image


# TODO: 代码 return total_loss / len(self.val_loader), total_acc / len(self.val_loader) 不太对劲，关注一下
# ==========================================
# 1. Config 模块 (参数配置寄存器)
# ==========================================
class Config:
    def __init__(self):
        # --- 自动归档设置 ---
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./result_{current_time}"
        self.ckpt_name = "best_model.ckpt"
        
        # --- 基础环境 ---
        self.seed = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_root = "./ML_HW/hw3/food-11/" 
        
        # --- 训练参数 ---
        self.img_size = (160, 160)
        self.num_classes = 11
        self.n_epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.0005
        self.weight_decay = 1e-5
        self.num_workers = 0  # 数据加载线程数

        self.optimizer = 'Adam'
        self.optim_hparams = {'lr': self.learning_rate, 'weight_decay': self.weight_decay}

    def create_output_dir(self):
        """创建文件夹并生成 config.txt"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

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
    plt.plot(record['train_loss'], label='Train Loss', c='tab:red')
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
    plt.plot(record['val_acc'], label='Val Acc', c='tab:green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "acc_curve.png"))
    plt.close()

# ==========================================
# 3. Data 模块 (I/O 接口与信号处理)
# ==========================================
def get_transforms(mode, img_size):
    """
    定义数据增强策略 (Signal Conditioning)
    mode: 'train' or 'test'
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
    train_tfm = get_transforms('train', cfg.img_size)
    test_tfm = get_transforms('test', cfg.img_size)
    
    # 2. 读取数据集
    # 假设路径结构为 root/training/labeled 和 root/validation
    train_path = os.path.join(cfg.data_root, "training/labeled")
    valid_path = os.path.join(cfg.data_root, "validation")
    
    # 检查路径是否存在
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(f"Data path not found. Please check: {cfg.data_root}")

    train_set = DatasetFolder(train_path, loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder(valid_path, loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

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
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    print(f"Data Module Ready. Train set: {len(train_set)}, Val set: {len(valid_set)}")
    return train_loader, valid_loader

# ==========================================
# 4. Model 模块 (DUT)
# ==========================================
class DynamicCNN(nn.Module):
    def __init__(self):
        super(DynamicCNN, self).__init__()
        self.cnn = nn.Sequential(
            # input (3, 160, 160)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # output (64, 80, 80)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # output (128, 40, 40)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # output (256, 20, 20)
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # output (512, 10, 10)
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            # output (1024, 5, 5)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 * 5 * 5, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out

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
                save_path = os.path.join(self.cfg.output_dir, self.cfg.ckpt_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> Best saved: {v_acc:.4f}")
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
        model = DynamicCNN(input_shape=(3, *cfg.img_size), num_classes=cfg.num_classes)
        
        # 4. 运行训练 
        trainer = Trainer(cfg, model, train_loader, val_loader)
        record = trainer.run()
        
        # 5. 导出波形
        save_curves(record, cfg.output_dir)
        print("Done.")
        
    except Exception as e:
        print(f"\n[Error]: {e}")