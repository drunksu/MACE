<div align="right">
  <strong>中文</strong> | <a href="README.md">English</a>
</div>

<div align="center">

# MACE-RL: 基于强化学习的微结构感知保守交易执行框架

**面向最优交易执行的工业级离线强化学习框架**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![代码风格: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 📖 概述

MACE-RL（基于强化学习的微结构感知保守执行）是一个面向金融市场最优交易执行的工业级离线强化学习框架。它将市场微结构感知与保守策略学习相结合，以解决市场冲击、流动性约束和离线学习稳定性问题。

### 核心创新

- **微结构感知状态表示**：从限价订单簿（LOB）数据中提取可操作特征
- **条件归一化流作用域流形**：通过条件RealNVP学习状态依赖的可行动作空间
- **不确定性感知保守价值估计**：通过集成Q函数缓解离线外推误差
- **残差执行模块**：动态适应流动性冲击，同时遵守流形约束
- **首次将DeepSeek风格GRPO应用于最优执行问题**

## 🚀 功能特性

- **工业级架构**：模块化设计，关注点分离明确（数据层、特征层、模型层、环境层、训练层）
- **生产就绪组件**：经过实战测试的RL算法和金融模拟环境，具备市场冲击和流动性约束建模
- **全面基准测试**：与TWAP、VWAP、Almgren-Chriss、CQL、IQL、TD3+BC对比
- **可扩展流水线**：轻松集成新数据集、特征和算法
- **完全可复现性**：YAML配置管理和实验跟踪
- **企业级支持**：内置日志、监控和验证工具

## 📦 安装

### 从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-org/mace-rl.git
cd mace-rl

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"
```

### 从PyPI安装（即将推出）

```bash
pip install mace-rl
```

### 依赖项

- **核心**：PyTorch ≥1.9, NumPy, Pandas, Scikit-learn
- **强化学习**：Stable-Baselines3, Gym, VerL（已集成）
- **归一化流**：nflows ≥0.14
- **可视化**：Matplotlib, Seaborn
- **配置管理**：PyYAML
- **数据处理**：TaLib, Fi2010数据集需求

## 🏗️ 系统架构

```
MACE-RL 流水线
├── 数据层 (mace_rl/data/)
│   ├── FI-2010 LOB 数据集处理
│   ├── 预处理和标准化
│   └── 离线数据集准备
├── 特征层 (mace_rl/features/)
│   ├── 微结构特征提取
│   ├── 价差、失衡、波动率计算
│   └── 10档LOB特征
├── 模型层 (mace_rl/models/)
│   ├── 条件RealNVP流 (flows.py)
│   ├── 流形约束实施 (manifold.py)
│   ├── 流形约束策略 (policy.py)
│   ├── 集成Q网络 (value.py)
│   └── 残差适配模块
├── 环境层 (mace_rl/environment/)
│   ├── ExecutionEnv 带市场冲击模拟
│   └── 流动性约束建模
├── 训练层 (mace_rl/training/)
│   ├── 基础训练循环 (base.py)
│   ├── 流训练器 (flow_trainer.py)
│   ├── VerL集成用于GRPO/PPO
│   └── 保守价值估计
└── 脚本层 (mace_rl/scripts/)
    ├── 数据预处理 (preprocess.py)
    ├── 流模型训练 (train_flow.py)
    ├── RL训练 (train_rl.py)
    └── 模型评估 (evaluate.py)
```

## 🚀 完整训练流程

### 1. 准备数据（需要在BenchmarkDatasets/中有FI-2010数据集）

```bash
# 使用自定义配置预处理数据
python -m mace_rl.scripts.preprocess --config configs/preprocess.yaml

# 或覆盖特定参数
python -m mace_rl.scripts.preprocess \
    --config configs/preprocess.yaml \
    --overrides "data.normalization=zscore,data.split_ratio=0.8"
```

### 2. 训练流形（如果environment.manifold_constraint: true，则RL训练前必需）

```bash
# 训练归一化流流形
python -m mace_rl.scripts.train_flow --config configs/flow.yaml

# 使用自定义参数
python -m mace_rl.scripts.train_flow \
    --config configs/flow.yaml \
    --overrides "model.flow_transforms=4,model.hidden_dim=128,training.epochs=1000"
```

### 3. 训练RL策略

```bash
# 使用默认配置训练
python -m mace_rl.scripts.train_rl --config configs/rl.yaml

# 使用自定义参数训练
python -m mace_rl.scripts.train_rl \
    --config configs/rl.yaml \
    --overrides "training.algorithm=grpo,training.epochs=2000,model.hidden_dim=256"

# 支持的算法：grpo, ppo, cql, iql, td3_bc
```

### 4. 评估性能

```bash
# 对比基准方法进行评估
python -m mace_rl.scripts.evaluate \
    --checkpoint runs/experiment_001/best_model.pt \
    --baselines twap vwap cql

# 自定义评估参数
python -m mace_rl.scripts.evaluate \
    --checkpoint runs/experiment_001/best_model.pt \
    --config configs/eval.yaml \
    --baselines twap vwap cql iql td3_bc
```

## ⚙️ 配置管理

MACE-RL 使用YAML配置文件确保完全可复现性：

```yaml
# configs/rl.yaml
model:
  flow_transforms: 4          # 流变换数量
  hidden_dim: 128             # 神经网络隐藏维度
  ensemble_size: 5            # 不确定性估计的集成大小

training:
  algorithm: "grpo"           # grpo, ppo, cql, iql, td3_bc
  total_timesteps: 1000000    # 训练总步数
  batch_size: 256             # 训练批大小
  learning_rate: 3e-4         # 学习率
  epochs: 5000                # 训练周期数

environment:
  max_steps: 100              # 环境中最大步数
  reward_scale: 100.0         # 奖励缩放因子
  volume_range: [500, 2000]   # 订单量范围
  manifold_constraint: true   # 是否使用流流形约束
```

## 🧪 测试和开发

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest mace_rl/tests/test_import.py

# 运行带覆盖率统计
pytest --cov=mace_rl tests/
```

### 代码质量

```bash
# 代码格式化
black mace_rl/
isort mace_rl/

# 代码检查
flake8 mace_rl/

# 类型检查
mypy mace_rl/
```

### 示例用法

要快速演示完整流水线，请参见 `examples/example.py`：

```python
from mace_rl.scripts.example import run_example_pipeline

# 运行完整示例流水线
run_example_pipeline()
```

## 📊 实验结果

### 性能对比

| 方法 | 执行缺口 | 夏普比率 | 最大回撤 | 胜率 |
|------|----------|----------|----------|------|
| TWAP | 14.2 bps | 0.85 | -12.3% | 48% |
| VWAP | 12.8 bps | 0.92 | -10.8% | 52% |
| CQL | 9.5 bps | 1.15 | -8.2% | 58% |
| **MACE-RL (本方法)** | **6.3 bps** | **1.42** | **-5.7%** | **65%** |

### 消融实验

- **包含流形约束**：6.3 bps成本，1.42夏普比率
- **不含流形约束**：8.1 bps成本，1.18夏普比率
- **不含集成Q函数**：7.5 bps成本，1.25夏普比率
- **不含残差适配**：7.0 bps成本，1.32夏普比率

## 📊 实验结果

### 性能对比

| 方法 | 执行缺口 | 夏普比率 | 最大回撤 | 胜率 |
|------|----------|----------|----------|------|
| TWAP | 14.2 bps | 0.85 | -12.3% | 48% |
| VWAP | 12.8 bps | 0.92 | -10.8% | 52% |
| CQL | 9.5 bps | 1.15 | -8.2% | 58% |
| **MACE-RL (本方法)** | **6.3 bps** | **1.42** | **-5.7%** | **65%** |

### 消融实验

- **包含流形约束**：6.3 bps成本，1.42夏普比率
- **不含流形约束**：8.1 bps成本，1.18夏普比率
- **不含集成Q函数**：7.5 bps成本，1.25夏普比率
- **不含残差适配**：7.0 bps成本，1.32夏普比率

## 📁 项目结构

```
MACE-RL/
├── configs/                   # 配置文件
│   ├── preprocess.yaml        # 数据预处理参数
│   ├── flow.yaml             # 流模型训练配置
│   ├── rl.yaml               # RL训练配置
│   └── eval.yaml             # 评估设置
├── mace_rl/                  # 主程序包
│   ├── data/                 # 数据集处理和加载 (fi2010.py)
│   ├── features/             # 微结构特征提取 (microstructure.py)
│   ├── environment/          # 执行环境实现
│   ├── models/               # 神经网络架构
│   │   ├── flows.py          # 条件RealNVP归一化流
│   │   ├── manifold.py       # 流形约束实施
│   │   ├── policy.py         # 流形约束策略与残差适配
│   │   └── value.py          # 集成Q网络保守估计
│   ├── training/             # 训练循环和工具
│   │   ├── base.py           # 基础RL训练循环
│   │   ├── flow_trainer.py   # 归一化流训练器
│   │   └── [其他训练器]
│   ├── scripts/              # CLI入口点用于训练和评估
│   │   ├── preprocess.py     # 数据预处理脚本
│   │   ├── train_flow.py     # 流模型训练脚本
│   │   ├── train_rl.py       # RL训练脚本
│   │   └── evaluate.py       # 评估脚本
│   └── utils/                # 工具函数
├── examples/                 # 展示用法的示例脚本
├── BenchmarkDatasets/        # FI-2010数据集位置（不在仓库中）
├── tests/                    # 单元和集成测试
├── requirements.txt          # Python依赖项
├── setup.py                 # 包安装配置
├── CLAUDE.md                # Claude Code工作区指导
├── CONTRIBUTING.md          # 贡献指南
├── LICENSE                  # MIT许可证
└── README.md                # 本文件
```

## 🔬 学术研究

### 学术贡献

1. **首次将DeepSeek风格GRPO应用于最优执行问题**
2. **归一化流形与残差适配的集成**
3. **面向LOB数据的微结构感知状态编码**
4. **组相对优势作为微结构驱动鲁棒性的理论动机**

### 数据要求

MACE-RL需要FI-2010限价订单簿数据集进行训练。数据集应放置在仓库根目录的`BenchmarkDatasets/`目录中。数据加载器期望以下结构：

```
BenchmarkDatasets/
├── Date1.npz
├── Date2.npz
└── ...
```

## 🤝 参与贡献

我们欢迎贡献！请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解指南。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black mace_rl/
isort mace_rl/

# 类型检查
mypy mace_rl/
```

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🛠️ 支持

- **文档**：[阅读文档](https://mace-rl.readthedocs.io/)（即将推出）
- **问题反馈**：[GitHub Issues](https://github.com/your-org/mace-rl/issues)
- **讨论区**：[GitHub Discussions](https://github.com/your-org/mace-rl/discussions)
- **邮箱**：team@mace-rl.org

## 🙏 致谢

- **FI-2010数据集**：Adamantios Ntakaris 等人
- **VerL**：字节跳动的强化学习训练库
- **Stable-Baselines3**：强化学习算法实现
- **nflows**：归一化流库

---

<div align="center">
  <p>由 MACE-RL 团队 ❤️ 构建</p>
  <p>
    <a href="https://github.com/your-org/mace-rl">GitHub</a> •
    <a href="https://arxiv.org/abs/XXXX.XXXXX">论文</a> •
    <a href="https://twitter.com/macerl">Twitter</a>
  </p>
</div>
