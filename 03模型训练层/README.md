# 03 模型训练层 (Alpha Combiner) 🤖

> 截面多因子量化选股系统 - AI模型训练与多模型融合层

---

## ✨ 核心能力

| 能力 | 说明 |
|:---|:---|
| 🔄 Walk-forward训练 | 滑动窗口滚动训练，Train/Valid/Test三重隔离，防止数据泄露 |
| 🎯 双模型类型 | LightGBM回归 + LambdaRank排序学习，支持不同目标函数 |
| 📉 EMA平滑 | 自适应半衰期平滑预测，根据自相关系数动态调整窗口长度 |
| 🔗 多模型融合 | IC加权融合多周期模型(5d/20d/60d)，支持动态权重与固定权重切换 |

---

## 🏗️ 架构概览

```
配置(config.yaml)
├── label.horizon → 标签周期(T+N)
├── walk_forward.gap_* → 训练/验证/测试隔离
└── model.* → 模型参数

数据流:
因子层 → DataConstructorV1 → WalkForwardSplitterV1 → WalkForwardTrainerV1
                                              ↓
                                    [fuse_predictions.py]
                                              ↓
                                    多模型IC加权融合
```

---

## 📁 目录结构

```
03模型训练层/
├── configs/                     # 配置文件
│   ├── horizon5_config.yaml     # 5日预测周期
│   ├── horizon20_config.yaml    # 20日预测周期（推荐）
│   ├── horizon60_config.yaml    # 60日预测周期
│   └── rank_config.yaml         # LambdaRank排序模型
│
├── dataset/                     # 数据构造
│   ├── data_constructor_v1.py   # X,y构造 + 标签计算
│   └── splitter_v1.py           # Walk-forward切分
│
├── models/                      # 模型实现
│   ├── base_model.py            # 模型抽象基类
│   ├── lightgbm_model.py        # LightGBM回归模型
│   └── lightgbm_rank_model.py   # LambdaRank排序模型
│
├── training/                    # 训练框架
│   └── walk_forward_trainer_v1.py
│
├── main_train_v1.py             # 单模型训练入口
├── fuse_predictions.py          # 多模型融合入口
│
└── experiments/                 # 实验输出（运行时生成，gitignore）
    └── {exp_id}/
        ├── smoothed_predictions.parquet      # EMA平滑后test预测
        ├── smoothed_live_predictions.parquet # EMA平滑后实盘预测
        ├── summary.parquet                   # IC/Rank IC/特征重要性
        ├── models/model_fold_*.pkl           # 各Fold模型
        └── *.png                             # 可视化图表
```

---

## 🔄 数据流与逻辑

### 🏷️ 标签计算流程（V1关键修复）

```
T日收盘价 ──→ T+1开盘价 ───────────────→ T+(h+1)开盘价
   │             │                           │
   │             │ 买入价                      │ 卖出价
   │             ▼                           ▼
   │        ┌─────────────────────────────────────┐
   │        │  label = open[t+h+1]/open[t+1] - 1  │
   │        │  (真实可执行，无未来信息)            │
   │        └─────────────────────────────────────┘
   │
   ▼
计算当日因子特征（使用T日收盘前的已知信息）
```

**关键设计**：
- 买入：T+1开盘价（T日收盘后无法以T日收盘价买入）
- 卖出：T+(horizon+1)开盘价
- 确保标签计算不泄露未来信息

### 🔄 Walk-forward训练流程

```
时间轴:  2010    2015    2020    2024
                            
Fold 1: [Train]──→[Valid]──→[Test]
           3年        6月       3月
Fold 2:   [Train]──→[Valid]──→[Test]
             3年        6M       3M
Fold 3:     [Train]──→[Valid]──→[Test]
               3年        6M       3M
                ...

隔离机制:
- gap_train_valid = horizon + 1
- gap_valid_test = horizon + 1
确保标签计算不越界
```

### 📉 EMA平滑逻辑

```
原始预测:  p[t-2]   p[t-1]   p[t]     p[t+1]
            │        │        │        │
            ▼        ▼        ▼        ▼
         ┌────────────────────────────────────┐
         │  EMA平滑:                          │
         │  α = 1 - exp(ln(0.5)/autocorr)    │
         │  smoothed[t] = α*p[t] + (1-α)*smoothed[t-1] │
         └────────────────────────────────────┘
                    │
                    ▼
平滑预测:  s[t-2]   s[t-1]   s[t]     s[t+1]
          (更稳定，换手率更低)
```

**自适应半衰期**：
- 趋势强（自相关高）→ 长窗口（20+天）
- 波动大（自相关低）→ 短窗口（5-10天）

### 🔗 多模型融合流程

```
输入: 各模型的 smoothed_predictions.parquet
       │
       ├── model_5d  (短周期)
       ├── model_20d (基准周期)
       └── model_60d (长周期)
       │
       ▼
┌─────────────────────────────────────────┐
│ IC计算（用base模型的actual统一计算）      │
│ 例：base=20d，则5d的IC = pred_5d vs actual_20d │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 权重计算                                │
│ • 前lag天(lag=base horizon): 等权        │
│ • 之后: 历史IC均值作为权重               │
│ • Test期: 动态权重（有actual）           │
│ • Live期: 固定权重（无actual）           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
           融合后预测分数
           smoothed_predictions.parquet
```

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
pip install lightgbm pandas numpy pyarrow pyyaml
```

### 2️⃣ 运行入口

```bash
# 单模型训练 - 20d模型(推荐)
python 03模型训练层/main_train_v1.py \
    --config 03模型训练层/configs/horizon20_config.yaml \
    --exp-id test_20d_v1 \
    -y

# 单模型训练 - 5d短周期
python 03模型训练层/main_train_v1.py \
    --config 03模型训练层/configs/horizon5_config.yaml \
    --exp-id test_5d_v1 \
    -y

# 单模型训练 - 60d长周期
python 03模型训练层/main_train_v1.py \
    --config 03模型训练层/configs/horizon60_config.yaml \
    --exp-id test_60d_v1 \
    -y
```

**多模型融合**：
```bash
# 融合5d+20d+60d，以20d为基准
python 03模型训练层/fuse_predictions.py \
    --exps test_5d_v1 test_20d_v1 test_60d_v1 \
    --base-idx 1 \
    --output-exp ensemble_5_20_60_v1
```

---

## 🔑 关键设计

### 🏷️ 标签计算V1修复

| 版本 | 买入价 | 卖出价 | 问题 |
|:---|:---|:---|:---|
| V0 | T日close | T+horizon日close | 无法以T日收盘价买入，未来函数 |
| **V1** | **T+1日open** | **T+(horizon+1)日open** | **真实可执行，无未来信息** |

**虽然IC数值降低，但实盘可信度更高。**

### 🛡️ 双重Gap防泄露

```yaml
walk_forward:
  gap_train_valid: 21   # = horizon + 1
  gap_valid_test: 21    # = horizon + 1
```

确保：
1. 训练集最后一个标签的计算不越界
2. 验证集和测试集完全隔离
3. 标签计算不泄露未来信息

### 📉 EMA平滑优势

| 指标 | 原始预测 | EMA平滑后 |
|:---|:---|:---|
| IC | 较高 | 略降（~10%） |
| 换手率 | 高 | **降低30-50%** |
| 信号稳定性 | 波动大 | **更稳定** |
| 实盘交易成本 | 高（频繁调仓） | **更低** |

**建议**：回测时对比 `--use-smooth` 前后的结果，选择合适的参数。

### 🔗 融合权重计算

**分界日期前（Test期）**：
- 各模型有足够的test数据
- 动态计算IC权重

**分界日期后（Live期）**：
- 长周期模型（60d）test结束，无actual_return
- 使用历史IC均值作为固定权重

---

## 📊 数据规范

### 输入

| 数据 | 路径 | 格式 |
|:---|:---|:---|
| 因子数据 | 02因子库/processed_data/factors/**/*.parquet | 宽表 [date × stock_code] |
| 行情数据 | 02因子库/processed_data/market_data/*.parquet | 宽表 [date × stock_code] |
| 配置文件 | configs/*.yaml | YAML格式 |

### 输出

**单模型输出**：
```
experiments/{exp_id}/
├── smoothed_predictions.parquet      # EMA平滑后test预测
│   └── 字段: date, stock_code, pred_score_smooth, actual_return, fold_id
├── smoothed_live_predictions.parquet # EMA平滑后实盘预测
│   └── 字段: date, stock_code, pred_score_smooth, fold_id
├── summary.parquet                   # 各Fold性能汇总
│   └── 字段: fold_id, ic, rank_ic, feature_importance
└── models/
    └── model_fold_{i}.pkl            # 各Fold训练好的模型
```

**融合模型输出**：
```
experiments/{ensemble_id}/
├── smoothed_predictions.parquet      # 融合后test预测
├── smoothed_live_predictions.parquet # 融合后实盘预测
└── fusion_config.yaml                # 融合配置+各模型权重
```

---

## 📚 详细文档

- [03.1_设计原理与逻辑架构](../docs/03.1_设计原理与逻辑架构.md) - 架构设计、数据流、Walk-forward详解
- [03.2_工程实现与规范](../docs/03.2_工程实现与规范.md) - API说明、配置规范、开发注意事项
- [03.3_运维与变更日志](../docs/03.3_运维与变更日志.md) - 检查点、性能基准、变更记录

---

*最后更新: 2026-03-26*  
*维护者: 蒋大王*
