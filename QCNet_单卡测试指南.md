# QCNet 单卡小数据测试指南

## 🎯 目标
测试你修改的模块是否有问题，使用单张GPU和少量数据快速验证。

## 📋 完整操作流程

### 1. 环境准备

```bash
# 激活环境
conda activate QCNet

# 验证GPU可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

# 验证PyG安装
python -c "import torch_geometric; print('PyG installed successfully')"
```

### 2. 数据准备

#### 2.1 Argoverse 2 数据集下载

**方法1: 官方完整下载（推荐用于正式使用）**
```bash
# 安装Argoverse 2 API
pip install av2

# 下载数据集（需要大量存储空间和时间）
# 训练集: ~1TB, 验证集: ~200GB, 测试集: ~200GB
python -c "
from av2.datasets.motion_forecasting import scenario_serialization
import os

# 创建数据目录
os.makedirs('/path/to/argoverse_v2', exist_ok=True)

# 下载验证集（相对较小，用于测试）
scenario_serialization.download_scenarios(
    dataset_type='val',
    output_dir='/path/to/argoverse_v2'
)
"
```

**方法2: 手动下载（适合测试）**
```bash
# 创建数据目录
mkdir -p /path/to/test_data/argoverse_v2
cd /path/to/test_data/argoverse_v2

# 从官方下载少量样本数据用于测试
# 访问: https://www.argoverse.org/av2.html#download-link
# 或使用wget下载部分文件
wget -c "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/val.tar.gz"

# 解压
tar -xzf val.tar.gz

# 清理压缩包
rm val.tar.gz
```

**方法3: 个人用户超轻量数据集（强烈推荐！）**
```bash
# 创建测试数据目录
mkdir -p /path/to/test_data/argoverse_v2/{train,val,test}
cd /path/to/test_data/argoverse_v2

# 下载单个样本文件（总共只有几MB）
# 验证集 - 只下载2个文件用于验证
wget -O val/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/val/0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
wget -O val/sample2.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/val/0000b175-3fc6-46a2-9d57-3e28e3e10140.parquet"

# 训练集 - 只下载1个文件用于训练测试
wget -O train/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/train/0000b329-3351-4e99-8677-68cc4c0e9ce4.parquet"

# 测试集 - 只下载1个文件用于测试
wget -O test/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/test/0000b0cd-6f82-4cba-81a7-6dc3ae5a7ea4.parquet"

echo "✅ 超轻量数据集下载完成！总大小约5-10MB"
```

**方法4: 创建最小测试数据集（从完整数据集复制）**
```bash
# 如果你已经有完整数据集的访问权限
mkdir -p /path/to/test_data/argoverse_v2/{train,val,test}

# 从完整数据集中只复制极少量文件
cp /path/to/full_argoverse_v2/val/*.parquet /path/to/test_data/argoverse_v2/val/ | head -2
cp /path/to/full_argoverse_v2/train/*.parquet /path/to/test_data/argoverse_v2/train/ | head -1  
cp /path/to/full_argoverse_v2/test/*.parquet /path/to/test_data/argoverse_v2/test/ | head -1
```

#### 2.2 数据集结构说明
下载完成后，数据结构应如下：
```
argoverse_v2/
├── train/
│   ├── 0000b329-3351-4e99-8677-68cc4c0e9ce4.parquet
│   ├── 0000b819-e28a-471a-bc81-09f34e6e5395.parquet
│   └── ...
├── val/
│   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet
│   ├── 0000b175-3fc6-46a2-9d57-3e28e3e10140.parquet  
│   └── ...
└── test/
    ├── 0000b0cd-6f82-4cba-81a7-6dc3ae5a7ea4.parquet
    ├── 0000b123-4567-8901-2345-6789abcdef01.parquet
    └── ...
```

#### 2.3 最小数据集准备（推荐用于测试）
```bash
# 创建测试数据目录
mkdir -p /path/to/test_data/argoverse_v2/{train,val,test}

# 如果下载了完整验证集，只使用前几个文件进行测试
cd /path/to/argoverse_v2/val
ls *.parquet | head -5 | xargs -I {} cp {} /path/to/test_data/argoverse_v2/val/
ls *.parquet | head -3 | xargs -I {} cp {} /path/to/test_data/argoverse_v2/train/
ls *.parquet | head -2 | xargs -I {} cp {} /path/to/test_data/argoverse_v2/test/
```

#### 2.4 数据验证
```bash
# 检查数据结构
ls -la /path/to/test_data/argoverse_v2/
# 应该包含: train/, val/, test/ 目录

# 检查文件数量
echo "Train files: $(ls /path/to/test_data/argoverse_v2/train/*.parquet 2>/dev/null | wc -l)"
echo "Val files: $(ls /path/to/test_data/argoverse_v2/val/*.parquet 2>/dev/null | wc -l)"  
echo "Test files: $(ls /path/to/test_data/argoverse_v2/test/*.parquet 2>/dev/null | wc -l)"

# 检查单个文件（使用pandas）
python -c "
import pandas as pd
import glob
val_files = glob.glob('/path/to/test_data/argoverse_v2/val/*.parquet')
if val_files:
    df = pd.read_parquet(val_files[0])
    print(f'File: {val_files[0]}')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
else:
    print('No parquet files found!')
"
```

### 3. 快速功能测试

#### 3.1 最小参数训练测试（推荐使用）
```bash
# 测试训练流程（只跑几个epoch验证没有错误）
python train_qcnet.py \
  --root /path/to/test_data/argoverse_v2/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --test_batch_size 1 \
  --devices 1 \
  --dataset argoverse_v2 \
  --num_historical_steps 25 \
  --num_future_steps 30 \
  --num_recurrent_steps 2 \
  --pl2pl_radius 75 \
  --time_span 10 \
  --pl2a_radius 25 \
  --a2a_radius 25 \
  --num_t2m_steps 15 \
  --pl2m_radius 75 \
  --a2m_radius 75 \
  --hidden_dim 64 \
  --num_modes 3 \
  --num_map_layers 1 \
  --num_agent_layers 1 \
  --num_dec_layers 1 \
  --num_heads 4 \
  --head_dim 16 \
  --num_freq_bands 32 \
  --max_epochs 2 \
  --num_workers 2 \
  --lr 1e-3 \
  --T_max 2
```

#### 3.2 模型推理测试（如果有预训练模型）
```bash
# 如果有预训练模型，测试推理
python val.py \
  --model QCNet \
  --root /path/to/test_data/argoverse_v2/ \
  --ckpt_path lightning_logs/version_0/checkpoints/epoch-xxx.ckpt \
  --batch_size 1 \
  --devices 1 \
  --num_workers 2
```

### 4. 调试模式运行

#### 4.1 创建调试脚本
创建文件 `debug_test.py`：

```python
# debug_test.py - 创建这个文件用于调试
import torch
import pytorch_lightning as pl
from predictors import QCNet
from datamodules import ArgoverseV2DataModule

# 设置调试模式
torch.autograd.set_detect_anomaly(True)
pl.seed_everything(2023, workers=True)

# 最小参数配置
config = {
    'dataset': 'argoverse_v2',
    'input_dim': 2,
    'hidden_dim': 32,  # 极小配置
    'output_dim': 2,
    'output_head': False,
    'num_historical_steps': 10,  # 极小配置
    'num_future_steps': 10,      # 极小配置
    'num_modes': 2,              # 极小配置
    'num_recurrent_steps': 1,
    'num_freq_bands': 16,
    'num_map_layers': 1,
    'num_agent_layers': 1,
    'num_dec_layers': 1,
    'num_heads': 2,
    'head_dim': 16,
    'dropout': 0.1,
    'pl2pl_radius': 50.0,
    'time_span': 10,
    'pl2a_radius': 25.0,
    'a2a_radius': 25.0,
    'num_t2m_steps': 10,
    'pl2m_radius': 50.0,
    'a2m_radius': 50.0,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'T_max': 2,
    'submission_dir': './',
    'submission_file_name': 'test_submission'
}

try:
    # 测试模型初始化
    print("🔥 Testing model initialization...")
    model = QCNet(**config)
    print("✅ Model initialized successfully")
    
    # 测试数据模块
    print("🔥 Testing data module...")
    datamodule = ArgoverseV2DataModule(
        root='/path/to/test_data/argoverse_v2/',  # 请替换为你的数据路径
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0  # 设为0避免多进程问题
    )
    print("✅ Data module created successfully")
    
    # 测试trainer
    print("🔥 Testing trainer...")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1,
        fast_dev_run=True,  # 只跑一个batch
        enable_checkpointing=False,
        logger=False
    )
    print("✅ Trainer created successfully")
    
    # 运行一个快速测试
    print("🔥 Running fast dev test...")
    trainer.fit(model, datamodule)
    print("✅ Fast dev test completed successfully")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
```

#### 4.2 运行调试脚本
```bash
python debug_test.py
```

### 5. 逐步测试策略

#### 5.1 测试单个模块
创建文件 `test_modules.py`：

```python
# test_modules.py
import torch
from modules import QCNetEncoder, QCNetDecoder
from torch_geometric.data import HeteroData

def test_encoder():
    print("🔥 Testing QCNetEncoder...")
    
    # 创建虚拟数据 (这里需要根据实际数据格式调整)
    data = HeteroData()
    # 添加必要的数据字段...
    
    encoder = QCNetEncoder(
        dataset='argoverse_v2',
        input_dim=2,
        hidden_dim=32,
        num_historical_steps=10,
        pl2pl_radius=50.0,
        time_span=10,
        pl2a_radius=25.0,
        a2a_radius=25.0,
        num_freq_bands=16,
        num_map_layers=1,
        num_agent_layers=1,
        num_heads=2,
        head_dim=16,
        dropout=0.1
    )
    
    try:
        output = encoder(data)
        print("✅ Encoder test passed")
        return True
    except Exception as e:
        print(f"❌ Encoder test failed: {e}")
        return False

def test_decoder():
    print("🔥 Testing QCNetDecoder...")
    # 类似的测试逻辑...
    pass

if __name__ == "__main__":
    test_encoder()
    # test_decoder()
```

### 6. 错误排查清单

#### 6.1 常见错误检查
```bash
# 检查1: CUDA内存
nvidia-smi

# 检查2: Python包版本
pip list | grep torch
pip list | grep geometric

# 检查3: 数据路径
ls -la /path/to/test_data/argoverse_v2/

# 检查4: 权限问题
touch test_file && rm test_file
```

#### 6.2 内存优化设置
```bash
# 如果内存不足，进一步减少参数
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 7. 批处理测试脚本

创建文件 `test_modifications.sh`：

```bash
#!/bin/bash
# test_modifications.sh

echo "🚀 开始测试QCNet修改..."

# 设置变量
DATA_ROOT="/path/to/test_data/argoverse_v2"  # 请替换为你的数据路径
TEST_LOG="test_$(date +%Y%m%d_%H%M%S).log"

# 创建日志文件
touch $TEST_LOG

echo "📝 日志文件: $TEST_LOG"

# 测试1: 基本导入测试
echo "🔥 测试1: 基本导入..." | tee -a $TEST_LOG
python -c "
try:
    from predictors import QCNet
    from modules import QCNetEncoder, QCNetDecoder
    print('✅ 导入成功')
except Exception as e:
    print(f'❌ 导入失败: {e}')
" 2>&1 | tee -a $TEST_LOG

# 测试2: 快速训练测试
echo "🔥 测试2: 快速训练..." | tee -a $TEST_LOG
timeout 300 python train_qcnet.py \
  --root $DATA_ROOT \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --test_batch_size 1 \
  --devices 1 \
  --dataset argoverse_v2 \
  --num_historical_steps 10 \
  --num_future_steps 10 \
  --num_recurrent_steps 1 \
  --pl2pl_radius 25 \
  --pl2a_radius 15 \
  --a2a_radius 15 \
  --pl2m_radius 25 \
  --a2m_radius 25 \
  --hidden_dim 32 \
  --num_modes 2 \
  --num_map_layers 1 \
  --num_agent_layers 1 \
  --num_dec_layers 1 \
  --num_heads 2 \
  --max_epochs 1 \
  --num_workers 0 2>&1 | tee -a $TEST_LOG

if [ $? -eq 0 ]; then
    echo "✅ 快速训练测试通过" | tee -a $TEST_LOG
else
    echo "❌ 快速训练测试失败" | tee -a $TEST_LOG
fi

echo "🎉 测试完成，查看日志: $TEST_LOG"
```

### 8. 运行测试

```bash
# 给脚本执行权限
chmod +x test_modifications.sh

# 运行测试
./test_modifications.sh
```

### 9. 成功标志

如果看到以下输出，说明你的修改没有问题：
- ✅ 模型初始化成功
- ✅ 数据加载成功  
- ✅ 前向传播成功
- ✅ 反向传播成功
- ✅ 至少完成一个训练步骤

### 10. 故障排除

如果出现错误：

#### 10.1 CUDA内存不足
```bash
# 进一步减少参数
python train_qcnet.py \
  --train_batch_size 1 \
  --hidden_dim 16 \
  --num_historical_steps 5 \
  --num_future_steps 5 \
  --num_modes 2 \
  --num_heads 2 \
  # ... 其他参数
```

#### 10.2 模块导入错误
- 检查你修改的模块语法
- 确保`__init__.py`正确导入
- 运行: `python -c "from modules import *"`

#### 10.3 数据格式错误
- 检查数据集路径和格式
- 使用更小的数据集进行测试
- 验证数据结构: `ls -la /path/to/data/`

## 📊 参数对比表

| 参数 | 原始值 | 轻量值 | 极小值 |
|------|-------|-------|-------|
| batch_size | 4 | 1 | 1 |
| hidden_dim | 128 | 64 | 32 |
| num_modes | 6 | 3 | 2 |
| num_historical_steps | 50 | 25 | 10 |
| num_future_steps | 60 | 30 | 10 |
| num_heads | 8 | 4 | 2 |
| radius | 150 | 75 | 25 |

## 🚀 快速开始

### 个人用户一键开始（推荐）
```bash
# 1. 进入项目目录
cd QCNet

# 2. 下载超轻量数据集（只需5-10MB）
mkdir -p ~/test_data/argoverse_v2/{train,val,test}
cd ~/test_data/argoverse_v2

# 下载样本文件
wget -O val/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/val/0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet" &
wget -O val/sample2.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/val/0000b175-3fc6-46a2-9d57-3e28e3e10140.parquet" &
wget -O train/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/train/0000b329-3351-4e99-8677-68cc4c0e9ce4.parquet" &
wget -O test/sample1.parquet "https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/sample_scenarios/test/0000b0cd-6f82-4cba-81a7-6dc3ae5a7ea4.parquet" &
wait

echo "数据下载完成！"
cd - # 回到QCNet目录

# 3. 直接测试你的修改
python train_qcnet.py \
  --root ~/test_data/argoverse_v2/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --test_batch_size 1 \
  --devices 1 \
  --dataset argoverse_v2 \
  --num_historical_steps 10 \
  --num_future_steps 10 \
  --num_recurrent_steps 1 \
  --pl2pl_radius 25 \
  --pl2a_radius 15 \
  --a2a_radius 15 \
  --pl2m_radius 25 \
  --a2m_radius 25 \
  --hidden_dim 32 \
  --num_modes 2 \
  --num_map_layers 1 \
  --num_agent_layers 1 \
  --num_dec_layers 1 \
  --num_heads 2 \
  --max_epochs 1 \
  --num_workers 0
```

### 标准流程

1. **克隆项目后**：
```bash
cd QCNet
cp debug_test.py ./
cp test_modifications.sh ./
```

2. **修改路径**：
   - 在所有脚本中将 `/path/to/test_data/argoverse_v2/` 替换为你的实际数据路径

3. **运行测试**：
```bash
# 快速验证
python debug_test.py

# 完整测试
./test_modifications.sh
```

4. **检查结果**：
   - 查看控制台输出
   - 检查生成的日志文件

## 📊 可视化功能

### 11.1 创建可视化脚本

创建文件 `visualize_predictions.py`：

```python
# visualize_predictions.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import seaborn as sns
from torch_geometric.data import DataLoader
from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder

class QCNetVisualizer:
    def __init__(self, model_path, data_root, device='cuda:0'):
        self.device = device
        self.model = QCNet.load_from_checkpoint(model_path, map_location=device)
        self.model.eval()
        
        # 创建数据集
        self.dataset = ArgoverseV2Dataset(
            root=data_root, 
            split='val',
            transform=TargetBuilder(self.model.num_historical_steps, self.model.num_future_steps)
        )
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_single_prediction(self, idx=0, save_path=None):
        """可视化单个预测结果"""
        data = self.dataset[idx]
        
        with torch.no_grad():
            # 转换为batch格式
            data = data.to(self.device)
            pred = self.model(data.unsqueeze(0))
        
        # 提取数据
        hist_pos = data['agent']['position'][:self.model.num_historical_steps].cpu().numpy()
        future_pos = data['agent']['target'][:self.model.num_future_steps].cpu().numpy()
        
        # 预测结果
        pred_pos = pred['loc_refine_pos'][0].cpu().numpy()  # [num_modes, num_future_steps, 2]
        probs = torch.softmax(pred['pi'][0], dim=-1).cpu().numpy()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：轨迹预测
        ax1.plot(hist_pos[:, 0], hist_pos[:, 1], 'b-o', linewidth=2, markersize=4, label='历史轨迹')
        ax1.plot(future_pos[:, 0], future_pos[:, 1], 'g-o', linewidth=2, markersize=4, label='真实未来轨迹')
        
        # 绘制多模态预测
        colors = plt.cm.Set3(np.linspace(0, 1, len(pred_pos)))
        for i, (traj, prob) in enumerate(zip(pred_pos, probs)):
            ax1.plot(traj[:, 0], traj[:, 1], '--', color=colors[i], 
                    linewidth=2, alpha=0.8, label=f'预测模式{i+1} (概率: {prob:.2f})')
        
        ax1.set_xlabel('X 坐标 (m)')
        ax1.set_ylabel('Y 坐标 (m)')
        ax1.set_title('QCNet 轨迹预测结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 右图：概率分布
        ax2.bar(range(len(probs)), probs, color=colors)
        ax2.set_xlabel('预测模式')
        ax2.set_ylabel('概率')
        ax2.set_title('预测模式概率分布')
        ax2.set_xticks(range(len(probs)))
        ax2.set_xticklabels([f'模式{i+1}' for i in range(len(probs))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
        
    def visualize_training_metrics(self, log_dir='lightning_logs'):
        """可视化训练指标"""
        # 查找最新的version目录
        log_path = Path(log_dir)
        if not log_path.exists():
            print("未找到训练日志目录")
            return
            
        version_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')]
        if not version_dirs:
            print("未找到训练日志")
            return
            
        latest_version = max(version_dirs, key=lambda x: int(x.name.split('_')[1]))
        
        # 读取metrics.csv
        metrics_file = latest_version / 'metrics.csv'
        if not metrics_file.exists():
            print("未找到metrics.csv文件")
            return
            
        df = pd.read_csv(metrics_file)
        
        # 创建训练曲线图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        train_loss = df[df['train_total_loss'].notna()]
        val_loss = df[df['val_minFDE'].notna()]
        
        if not train_loss.empty:
            axes[0, 0].plot(train_loss['step'], train_loss['train_total_loss'], label='训练损失')
            axes[0, 0].set_title('训练损失曲线')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('损失值')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 验证指标
        if not val_loss.empty:
            axes[0, 1].plot(val_loss['epoch'], val_loss['val_minFDE'], 'r-', label='minFDE')
            axes[0, 1].set_title('验证指标 - minFDE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('minFDE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 其他指标
        if 'val_minADE' in df.columns:
            val_ade = df[df['val_minADE'].notna()]
            if not val_ade.empty:
                axes[1, 0].plot(val_ade['epoch'], val_ade['val_minADE'], 'g-', label='minADE')
                axes[1, 0].set_title('验证指标 - minADE')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('minADE')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        if 'val_MR' in df.columns:
            val_mr = df[df['val_MR'].notna()]
            if not val_mr.empty:
                axes[1, 1].plot(val_mr['epoch'], val_mr['val_MR'], 'm-', label='Miss Rate')
                axes[1, 1].set_title('验证指标 - Miss Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Miss Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("训练指标图已保存为 training_metrics.png")

# 使用示例
if __name__ == "__main__":
    # 使用预训练模型（如果有的话）
    model_path = "lightning_logs/version_0/checkpoints/epoch-0.ckpt"  # 替换为实际路径
    data_root = "~/test_data/argoverse_v2/"  # 替换为实际数据路径
    
    if Path(model_path).exists():
        visualizer = QCNetVisualizer(model_path, data_root)
        
        # 可视化预测结果
        visualizer.visualize_single_prediction(idx=0, save_path="prediction_result.png")
        
        # 可视化训练指标
        visualizer.visualize_training_metrics()
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
```

### 11.2 实时可视化脚本

创建文件 `real_time_visualizer.py`：

```python
# real_time_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import torch
from datasets import ArgoverseV2Dataset
from predictors import QCNet

class RealTimeVisualizer:
    def __init__(self, model_path, data_root):
        self.model = QCNet.load_from_checkpoint(model_path)
        self.model.eval()
        self.dataset = ArgoverseV2Dataset(root=data_root, split='val')
        
        # 初始化图形
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('QCNet 实时预测可视化')
        self.ax.grid(True, alpha=0.3)
        
    def update_plot(self, frame):
        self.ax.clear()
        
        # 获取数据
        idx = frame % len(self.dataset)
        data = self.dataset[idx]
        
        with torch.no_grad():
            pred = self.model(data.unsqueeze(0))
        
        # 绘制结果
        hist_pos = data['agent']['position'][:self.model.num_historical_steps].numpy()
        self.ax.plot(hist_pos[:, 0], hist_pos[:, 1], 'b-o', label='历史轨迹')
        
        # 绘制预测
        pred_pos = pred['loc_refine_pos'][0].numpy()
        colors = plt.cm.Set3(np.arange(len(pred_pos)))
        
        for i, traj in enumerate(pred_pos):
            self.ax.plot(traj[:, 0], traj[:, 1], '--', color=colors[i], 
                        label=f'预测{i+1}')
        
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        self.ax.legend()
        self.ax.set_title(f'样本 {idx}: QCNet 实时预测')
        
    def start(self):
        anim = FuncAnimation(self.fig, self.update_plot, interval=2000, cache_frame_data=False)
        plt.show()
        return anim

# 使用方法
if __name__ == "__main__":
    visualizer = RealTimeVisualizer("path/to/model.ckpt", "path/to/data")
    visualizer.start()
```

### 11.3 高级多场景可视化（类似你的图片）

创建并运行 `advanced_visualizer.py`（已创建），实现专业级可视化：

```bash
# 安装可视化依赖
pip install matplotlib seaborn

# 运行高级多场景可视化（类似你的图片效果）
python advanced_visualizer.py

# 运行基础可视化
python visualize_predictions.py

# 实时可视化（如果有训练好的模型）
python real_time_visualizer.py
```

### 11.4 快速可视化命令

### 11.4 训练过程中的可视化

在 `debug_test.py` 中添加TensorBoard日志：

```python
# 在debug_test.py中修改trainer配置
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="qcnet_test")

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=1,
    logger=logger,  # 添加logger
    # ... 其他参数
)

# 训练后启动TensorBoard
# tensorboard --logdir=tb_logs
```

### 11.5 可视化结果示例

运行可视化脚本后，你会看到：

1. **轨迹预测图**：
   - 蓝色线：历史轨迹
   - 绿色线：真实未来轨迹  
   - 彩色虚线：多模态预测结果

2. **概率分布图**：
   - 显示每个预测模式的概率

3. **训练曲线**：
   - 损失下降情况
   - 验证指标变化

4. **实时预测**：
   - 动态显示不同样本的预测结果

5. **高级多场景可视化**（类似你提供的图片）：
   - 黑色背景的专业可视化界面
   - 多场景并排显示
   - 道路网络和车道线渲染
   - 橙色高亮区域标注
   - 交互式场景切换

## 💡 小贴士

- **内存不够**：逐步减少参数，从batch_size开始
- **速度优化**：设置`num_workers=0`避免多进程开销  
- **调试模式**：使用`fast_dev_run=True`只跑一个batch
- **日志查看**：每次测试都会生成带时间戳的日志文件
- **可视化调试**：使用可视化功能直观查看模型预测效果

这个指南应该能帮你快速验证修改是否有问题！🎉 