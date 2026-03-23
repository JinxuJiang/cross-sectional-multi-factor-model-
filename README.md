# 截面多因子量化选股系统

> 可复现、可维护、可快速迭代的截面多因子量化选股系统。基于机器学习（LightGBM/XGBoost/Transformer）捕捉因子与收益率的非线性关系。

---

## 核心准则

> **逻辑领先于代码，相关性服务于因果。**

- 每个因子必须有清晰的因果逻辑（Causal Logic）
- 通过低相关因子流实现 Alpha 与 Beta 的分离

---

## 系统架构

```
                    截面多因子量化选股系统

  Module 1        Module 2         Module 3        Module 4

    Data    →   Alpha    →   Alpha   →   Optimizer
   Engine      Factory      Combiner     & Backtest
  数据引擎      因子工厂       AI训练       优化回测

  原始数据        因子计算       非线性建模      组合优化
  PIT对齐        数据清洗       滚动训练        交易成本
```

| 模块 | 目录 | 核心职责 | 关键技术 |
|:---|:---|:---|:---|
| 数据引擎 | `01数据/` | 原始数据搬运与存储 | Parquet、PIT对齐、等比前复权 |
| 因子工厂 | `02因子库/` | 因子计算与数据清洗 | MAD去极值、标准化、行业中性化 |
| AI训练 | `03模型训练层/` | 非线性建模与集成 | LightGBM、Walk-forward |
| 回测优化 | `04回测层/` | 组合优化与绩效分析 | Alphalens、Backtrader |

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n quant python=3.9
conda activate quant

# 安装依赖
pip install pandas numpy pyarrow lightgbm xgboost backtrader alphalens
```

### 2. 数据下载（需QMT账号）

```bash
cd 01数据
python data_main.py --full
```

### 3. 因子计算

```bash
cd 02因子库
python update_all.py
```

### 4. 模型训练

```bash
cd 03模型训练层
python main_train_v1.py --config configs/default_config.yaml
```

### 5. 回测

```bash
cd 04回测层
python backtrader_eval_1.1.py --exp_id exp_001
```

---

## 目录结构

```
截面多因子模型/
├── 01数据/                    # 【Module 1: 数据引擎】
│   ├── Base_DataEngine.py     # QMT数据下载
│   ├── monthly_update.py      # 月度更新
│   ├── data_main.py           # 主入口
│   └── data/                  # 【运行时生成】原始数据存储
│
├── 02因子库/                  # 【Module 2: 因子工厂】
│   ├── src/
│   │   ├── data_engine/       # 数据加载与PIT对齐
│   │   ├── alpha_factory/     # 因子计算
│   │   │   ├── technical/     # 技术因子
│   │   │   └── financial/     # 财务因子
│   │   └── processors/        # 清洗流程
│   └── processed_data/        # 【运行时生成】处理后数据
│       ├── market_data/
│       ├── financial_data/
│       └── factors/
│
├── 03模型训练层/              # 【Module 3: AI训练】
│   ├── models/                # 模型定义
│   ├── training/              # Walk-forward训练器
│   ├── dataset/               # 数据构造
│   ├── configs/               # 配置文件
│   ├── main_train_v1.py       # 训练主入口
│   └── experiments/           # 【运行时生成】实验输出
│
├── 04回测层/                  # 【Module 4: 回测优化】
│   ├── backtrader_eval_*.py   # 策略回测
│   ├── alphalens_analysis.py  # 因子分析
│   ├── generate_live_signals.py
│   └── reports/               # 【运行时生成】回测报告
│
├── docs/                      # 详细文档
├── xtquant/                   # QMT API库
└── README.md                  # 本文档
```

---

## 数据规范

### One Factor, One File

所有因子存储为独立的 Parquet 宽表：

```
processed_data/
├── market_data/              # 行情基础数据
├── financial_data/           # 财务基础数据（PIT对齐）
└── factors/                  # 因子数据
    ├── technical/            # 技术因子
    └── financial/            # 财务因子
```

### 格式标准

| 维度 | 规范 |
|:---|:---|
| 索引 | 0-N 整数 |
| 行 | 时间（date） |
| 列 | 股票代码（ticker） |
| 值 | 对应股票在当天的因子值 |

---

## 详细文档

| 模块 | 设计原理 | 工程规范 | 运维日志 |
|:---|:---|:---|:---|
| 项目总体 | [总体要求](docs/00.1_项目总体要求.md) | [文档架构](docs/00.2_项目文档架构.md) | - |
| 01数据层 | [设计原理](docs/01.1_设计原理与逻辑架构.md) | [工程规范](docs/01.2_工程实现与规范.md) | [变更日志](docs/01.3_运维与变更日志.md) |

---

## 后续路线图

- [ ] 遗传算法自动因子挖掘
- [ ] Transformer模型接入
- [ ] MLOps (MLflow) 实验追踪
- [ ] 云部署 (Azure)
- [ ] 另类数据接入

---

*项目维护: 蒋大王*  
*最后更新: 2026-03-23*
