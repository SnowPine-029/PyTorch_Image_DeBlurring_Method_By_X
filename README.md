# PyTorch_Image_DeBlurring_Method_By_X
基于深度学习训练模型实现图像去模糊化的方法
import os
import sys
import time
import math
import random
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==================== Windows特定优化 ====================
def windows_system_optimization():
    """Windows系统优化设置"""
    # 设置环境变量以减少内存问题
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # 设置PyTorch共享内存策略
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)
    except:
        pass
    
    print("[系统优化] Windows优化设置已应用")

# ==================== 简单退化模拟模块 ====================
class SimpleDegradationSimulator:
    """简化的退化模拟器"""
    
    def degrade_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        模拟低分辨率相机退化
        简化的退化过程，减少内存使用
        """
        B, C, H, W = img.shape
        
        # 1. 下采样到1/4大小
        down_h, down_w = H // 4, W // 4
        down_h = max(down_h, 32)
        down_w = max(down_w, 32)
        
        downsampled = F.interpolate(img, size=(down_h, down_w), 
                                   mode='bilinear', align_corners=False)
        
        # 2. 上采样回原尺寸
        upsampled = F.interpolate(downsampled, size=(H, W), 
                                 mode='bilinear', align_corners=False)
        
        # 3. 轻微高斯模糊
        kernel_size = 3
        sigma = 0.5
        
        # 手动实现3x3高斯模糊
        kernel = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32, device=img.device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        
        blurred = F.conv2d(upsampled, kernel, padding=1, groups=C)
        
        return torch.clamp(blurred, 0, 1)

# ==================== 简化数据集类 ====================
class SimpleImageDataset(Dataset):
    """简化图像数据集"""
    
    def __init__(self, image_dir: str, patch_size: int = 256):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        
        # 收集所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(self.image_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        if not self.image_paths:
            raise ValueError(f"在 {image_dir} 中未找到图像文件")
            
        print(f"[数据集] 找到 {len(self.image_paths)} 张图像")
        
        # 初始化退化模拟器
        self.degrader = SimpleDegradationSimulator()
        
    def __len__(self) -> int:
        # 每张图像生成多个patch以增加数据量
        return len(self.image_paths) * 5
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算实际图像索引
        img_idx = idx % len(self.image_paths)
        
        try:
            # 加载图像
            img_path = self.image_paths[img_idx]
            image = Image.open(img_path).convert('RGB')
            
            # 转换为张量
            image = TF.to_tensor(image)
            
            # 随机裁剪
            H, W = image.shape[1], image.shape[2]
            
            if H < self.patch_size or W < self.patch_size:
                # 图像太小，调整大小
                image = TF.resize(image, (self.patch_size, self.patch_size))
                h_start, w_start = 0, 0
            else:
                # 随机裁剪
                h_start = random.randint(0, H - self.patch_size)
                w_start = random.randint(0, W - self.patch_size)
            
            hr_patch = image[:, h_start:h_start+self.patch_size, 
                            w_start:w_start+self.patch_size]
            
            # 模拟退化
            lr_patch = self.degrader.degrade_image(hr_patch.unsqueeze(0)).squeeze(0)
            
            # 简单的数据增强（减少内存使用）
            if random.random() > 0.5:
                hr_patch = TF.hflip(hr_patch)
                lr_patch = TF.hflip(lr_patch)
            
            return lr_patch, hr_patch
            
        except Exception as e:
            print(f"[数据集错误] 处理图像时出错: {e}")
            # 返回一个简单的占位符
            dummy_img = torch.zeros((3, self.patch_size, self.patch_size))
            return dummy_img, dummy_img

# ==================== 轻量级U-Net模型 ====================
class LightweightUNet(nn.Module):
    """轻量级U-Net，适合小数据集"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output = nn.Conv2d(32, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        enc1 = self.enc1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        # 瓶颈层
        bottleneck = self.bottleneck(enc2_pool)
        
        # 解码器
        up2 = self.up2(bottleneck)
        dec2_input = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2_input)
        
        up1 = self.up1(dec2)
        dec1_input = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1_input)
        
        # 输出
        output = self.output(dec1)
        
        return torch.sigmoid(output)

# ==================== 简单损失函数 ====================
class SimpleLoss(nn.Module):
    """简单损失函数：L1 + SSIM"""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def ssim_loss(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """简化SSIM损失"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.std()
        sigma2 = img2.std()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        
        return 1 - ssim
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        # 权重调整
        total_loss = l1 + 0.1 * ssim
        
        return total_loss

# ==================== 安全训练器 ====================
class SafeImageTrainer:
    """安全训练器，避免Windows多进程问题"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 应用Windows优化
        windows_system_optimization()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[设备] 使用设备: {self.device}")
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
        
        # 创建日志目录
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return 100.0
        return 20 * math.log10(1.0 / math.sqrt(mse))
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器（使用单进程避免Windows问题）"""
        
        # 创建数据集
        dataset = SimpleImageDataset(
            image_dir=self.config['data_dir'],
            patch_size=self.config.get('patch_size', 256)
        )
        
        # 分割训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 在Windows上使用单进程数据加载
        num_workers = 0  # Windows上使用0以避免多进程问题
        batch_size = self.config.get('batch_size', 2)  # 使用小批次
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # 验证时也使用单进程
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"[数据] 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        print(f"[数据] 批次大小: {batch_size}, Workers: {num_workers}")
        
        return train_loader, val_loader
    
    def train(self):
        """训练模型"""
        print("\n" + "="*60)
        print("开始训练图像清晰化模型（Windows优化版）")
        print("="*60 + "\n")
        
        try:
            # 创建数据加载器
            train_loader, val_loader = self._create_data_loaders()
            
            # 创建模型
            model = LightweightUNet().to(self.device)
            
            print(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
            
            # 损失函数和优化器
            criterion = SimpleLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            
            # 学习率调度器
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            # 训练循环
            num_epochs = self.config.get('num_epochs', 50)
            best_val_loss = float('inf')
            
            train_losses = []
            val_losses = []
            psnr_values = []
            
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
                    # 移动到设备
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
                    # 前向传播
                    pred_imgs = model(lr_imgs)
                    loss = criterion(pred_imgs, hr_imgs)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 优化器步进
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # 每5个批次打印一次进度
                    if batch_idx % 5 == 0:
                        print(f"[训练] Epoch {epoch+1}/{num_epochs} | "
                              f"Batch {batch_idx}/{len(train_loader)} | "
                              f"Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
                train_losses.append(avg_train_loss)
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_psnr = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for lr_imgs, hr_imgs in val_loader:
                        lr_imgs = lr_imgs.to(self.device)
                        hr_imgs = hr_imgs.to(self.device)
                        
                        pred_imgs = model(lr_imgs)
                        loss = criterion(pred_imgs, hr_imgs)
                        
                        # 计算PSNR
                        psnr = self._calculate_psnr(pred_imgs, hr_imgs)
                        
                        val_loss += loss.item()
                        val_psnr += psnr
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
                avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0.0
                
                val_losses.append(avg_val_loss)
                psnr_values.append(avg_val_psnr)
                
                # 更新学习率
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                
                # 打印进度
                print(f"[Epoch {epoch+1:03d}] "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"PSNR: {avg_val_psnr:.2f}dB | "
                      f"LR: {current_lr:.6f}")
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint(model, optimizer, epoch, best_val_loss, is_best=True)
                
                # 定期保存检查点
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(model, optimizer, epoch, avg_val_loss, is_best=False)
            
            print(f"\n[训练完成] 最佳验证损失: {best_val_loss:.6f}")
            
            # 保存训练日志
            self._save_training_log(train_losses, val_losses, psnr_values)
            
            return model
            
        except Exception as e:
            print(f"\n[错误] 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                        epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if is_best:
            filename = checkpoint_dir / "best_model.pth"
            print(f"[检查点] 保存最佳模型到 {filename} (损失: {val_loss:.6f})")
        else:
            filename = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
        
        torch.save(checkpoint, filename)
        
        # 保存轻量版模型（仅权重）
        torch.save(model.state_dict(), checkpoint_dir / "final_model_weights.pth")
    
    def _save_training_log(self, train_losses, val_losses, psnr_values):
        """保存训练日志"""
        with open(self.log_dir / "training_log.csv", "w") as f:
            f.write("epoch,train_loss,val_loss,psnr\n")
            for i in range(len(train_losses)):
                f.write(f"{i+1},{train_losses[i]:.6f},{val_losses[i]:.6f},{psnr_values[i]:.2f}\n")
        
        print(f"[日志] 训练日志已保存到 {self.log_dir / 'training_log.csv'}")
    
    def load_model(self, model_path: str = "checkpoints/best_model.pth"):
        """加载已训练的模型"""
        try:
            # 创建模型
            model = LightweightUNet().to(self.device)
            
            if os.path.exists(model_path):
                # 检查是完整检查点还是仅权重文件
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    # 完整检查点
                    model.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 0)
                    val_loss = checkpoint.get('val_loss', 0.0)
                    print(f"[模型] 加载检查点模型 (Epoch {epoch}, Loss: {val_loss:.6f})")
                else:
                    # 仅权重文件
                    model.load_state_dict(checkpoint)
                    print(f"[模型] 加载权重文件: {model_path}")
                
                print(f"[模型] 模型加载成功")
                return model
            else:
                print(f"[错误] 找不到模型文件: {model_path}")
                print(f"[提示] 请先训练模型或确保模型路径正确")
                return None
                
        except Exception as e:
            print(f"[错误] 加载模型时出错: {e}")
            return None
    
    def enhance_image(self, model: nn.Module, image_path: str, output_path: str = None):
        """增强单张图像"""
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"[错误] 找不到图像文件: {image_path}")
                return None
                
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (W, H)
            
            print(f"[增强] 处理图像: {image_path}, 原始尺寸: {original_size}")
            
            # 转换为张量
            img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            
            # 应用模型
            model.eval()
            with torch.no_grad():
                enhanced_tensor = model(img_tensor)
            
            # 转换回PIL图像
            enhanced_tensor = enhanced_tensor.squeeze(0).cpu().clamp(0, 1)
            enhanced_image = TF.to_pil_image(enhanced_tensor)
            
            # 调整回原始尺寸
            if enhanced_image.size != original_size:
                enhanced_image = enhanced_image.resize(original_size, Image.BICUBIC)
            
            # 保存图像
            if output_path:
                enhanced_image.save(output_path)
                print(f"[增强] 图像已保存到: {output_path}")
            
            return enhanced_image
            
        except Exception as e:
            print(f"[错误] 增强图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

# ==================== 模式选择函数 ====================
def run_training_mode():
    """运行训练模式"""
    # 配置参数（针对Windows优化）
    config = {
        # 数据路径
        'data_dir': 'clear_photos',  # 你的清晰照片文件夹路径
        
        # 训练参数（调整为小规模）
        'num_epochs': 50,  # 减少epoch数量
        'batch_size': 2,   # 使用小批次
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'patch_size': 128,  # 使用较小的patch
        
        # 路径
        'log_dir': 'training_logs',
        
        # 随机种子
        'seed': 42,
    }
    
    try:
        # 检查数据目录
        if not os.path.exists(config['data_dir']):
            print(f"\n[错误] 找不到数据目录: {config['data_dir']}")
            print(f"请创建 '{config['data_dir']}' 文件夹，并将清晰照片放入其中")
            os.makedirs(config['data_dir'], exist_ok=True)
            print("文件夹已创建，请添加照片后重新运行程序")
            return
        
        # 检查是否有图像文件
        image_files = list(Path(config['data_dir']).glob("*.jpg")) + \
                     list(Path(config['data_dir']).glob("*.jpeg")) + \
                     list(Path(config['data_dir']).glob("*.png"))
        
        if len(image_files) == 0:
            print(f"\n[错误] 在 '{config['data_dir']}' 中没有找到图像文件")
            print("支持的格式: .jpg, .jpeg, .png")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 创建训练器
        trainer = SafeImageTrainer(config)
        
        # 开始训练
        model = trainer.train()
        
        if model is not None:
            print("\n" + "="*60)
            print("训练完成！模型已保存到 checkpoints/ 目录")
            print("="*60)
            
            # 询问是否立即测试模型
            test_choice = input("\n是否使用刚训练的模型测试一张图像? (y/n): ").lower()
            if test_choice == 'y':
                test_image_path = input("请输入测试图像路径 (或按回车使用默认模糊图像): ").strip()
                if not test_image_path:
                    test_image_path = "blurry_photo.jpg"
                    
                output_path = "enhanced_test.jpg"
                
                if os.path.exists(test_image_path):
                    enhanced_image = trainer.enhance_image(
                        model=model,
                        image_path=test_image_path,
                        output_path=output_path
                    )
                    
                    if enhanced_image:
                        print(f"测试完成！增强后的图像已保存为: {output_path}")
                    else:
                        print("测试失败")
                else:
                    print(f"[提示] 找不到图像文件: {test_image_path}")
                    print(f"请将你的模糊照片放在当前目录并命名为 'blurry_photo.jpg'")
        
        return model
        
    except Exception as e:
        print(f"\n[严重错误] 训练过程失败: {e}")
        import traceback
        traceback.print_exc()

def run_inference_mode():
    """运行推理模式（使用已有模型）"""
    print("\n" + "="*60)
    print("推理模式：使用已有模型增强图像")
    print("="*60)
    
    try:
        # 配置参数
        config = {
            'log_dir': 'inference_logs',
            'seed': 42,
        }
        
        # 创建训练器实例（用于加载模型和处理图像）
        trainer = SafeImageTrainer(config)
        
        # 询问模型路径
        model_path = input(f"请输入模型路径 (或按回车使用默认路径 'checkpoints/best_model.pth'): ").strip()
        if not model_path:
            model_path = "checkpoints/best_model.pth"
        
        # 加载模型
        model = trainer.load_model(model_path)
        if model is None:
            return
        
        # 询问要处理的图像
        while True:
            print("\n" + "-"*60)
            image_path = input("请输入要增强的图像路径 (或输入 'q' 退出): ").strip()
            
            if image_path.lower() == 'q':
                break
            
            # 生成输出路径
            input_path = Path(image_path)
            output_path = f"{input_path.stem}_enhanced{input_path.suffix}"
            
            # 处理图像
            enhanced_image = trainer.enhance_image(
                model=model,
                image_path=image_path,
                output_path=output_path
            )
            
            if enhanced_image:
                print(f"✓ 图像增强完成！增强后的图像保存为: {output_path}")
            
            # 询问是否继续处理其他图像
            continue_choice = input("\n是否继续处理其他图像? (y/n): ").lower()
            if continue_choice != 'y':
                break
        
        print("\n推理模式结束。")
        
    except Exception as e:
        print(f"\n[严重错误] 推理过程失败: {e}")
        import traceback
        traceback.print_exc()

# ==================== 主程序 ====================
def main():
    """主函数"""
    # 显示欢迎信息
    print("="*70)
    print("图像清晰化深度学习模型 - Windows优化版（用户选项版）")
    print("="*70)
    print("系统信息:")
    print(f"- 操作系统: {sys.platform}")
    print(f"- Python版本: {sys.version.split()[0]}")
    print(f"- PyTorch版本: {torch.__version__}")
    print(f"- CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA版本: {torch.version.cuda}")
        print(f"- 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("-"*70)
    
    # 检查必要的包
    try:
        import numpy as np
        from PIL import Image
        print("✓ 必要的包已安装")
    except ImportError as e:
        print(f"✗ 缺少必要的包: {e}")
        print("请运行: pip install numpy pillow torch torchvision")
        sys.exit(1)
    
    # 用户选择模式
    print("\n请选择运行模式:")
    print("  1 - 训练新的模型")
    print("  2 - 使用已有模型增强图像")
    print("  0 - 退出程序")
    
    while True:
        choice = input("\n请输入选项 (0, 1, 或 2): ").strip()
        
        if choice == '0':
            print("程序退出。")
            break
        elif choice == '1':
            run_training_mode()
            break
        elif choice == '2':
            run_inference_mode()
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
