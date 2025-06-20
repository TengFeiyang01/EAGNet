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

## 💡 小贴士

- **内存不够**：逐步减少参数，从batch_size开始
- **速度优化**：设置`num_workers=0`避免多进程开销  
- **调试模式**：使用`fast_dev_run=True`只跑一个batch
- **日志查看**：每次测试都会生成带时间戳的日志文件

这个指南应该能帮你快速验证修改是否有问题！🎉 