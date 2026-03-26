# 03模型训练层 (Alpha Combiner)

> Walk-forward滚动训练 + LightGBM模型 + 多模型IC加权融合

## 核心能力

| 能力 | 说明 |
|:---|:---|
| Walk-forward训练 | 滑动窗口滚动训练，防止数据泄露 |
| 双模型类型 | LightGBM回归 + LambdaRank排序学习 |
| EMA平滑 | 自适应半衰期平滑预测，降低噪声 |
| 多模型融合 | IC加权融合多周期模型(5d/20d/60d) |

## 架构概览

```
配置(config.yaml)
├── label.horizon → 标签周期(T+N)
├── walk_forward.gap_* → 训练/验证/测试隔离
└── model.* → 模型参数

d数据流:
因子层 → DataConstructorV1 → WalkForwardSplitterV1 → WalkForwardTrainerV1
                                              ↓
                                    [fuse_predictions.py]
                                              ↓
                                    多模型IC加权融合
```

### 关键设计

**1. 标签计算(V1修复)**
- 买入: T+1开盘价
- 卖出: T+(horizon+1)开盘价
- 收益: `open[t+h+1]/open[t+1] - 1`
- 真实可执行，无未来信息

**2. 双重Gap防泄露**
- `gap = label_horizon + 1`
- Train/Valid/Test三重隔离
- 确保标签计算不越界

**3. EMA平滑**
- 自适应半衰期: `halflife = ln(0.5)/ln(autocorr)`
- 趋势强→长窗口(20+天)，波动大→短窗口(5-10天)
- 平滑后预测更稳定

**4. 多模型融合**
- **输入**: 各模型的`smoothed_predictions.parquet`
- **IC计算**: 用base模型的actual统一计算
- **滞后权重**: lag=base horizon，前lag天等权，之后历史IC均值
- **分界处理**: Test期动态权重，Live期固定权重

## 快速开始

### 单模型训练

```bash
# 20d模型(推荐)
python main_train_v1.py --config configs/horizon20_config.yaml --exp-id test_20d_v1 -y

# 5d短周期
python main_train_v1.py --config configs/horizon5_config.yaml --exp-id test_5d_v1 -y

# 60d长周期
python main_train_v1.py --config configs/horizon60_config.yaml --exp-id test_60d_v1 -y
```

### 多模型融合

```bash
# 融合5d+20d+60d，以20d为基准
python fuse_predictions.py \
    --exps test_5d_v1 test_20d_v1 test_60d_v1 \
    --base-idx 1 \
    --output-exp ensemble_5_20_60_v1
```

## 输出文件

```
experiments/{exp_id}/
├── smoothed_predictions.parquet      # EMA平滑后test预测
├── smoothed_live_predictions.parquet # EMA平滑后实盘预测
├── summary.parquet                   # IC/Rank IC/特征重要性
├── models/model_fold_*.pkl           # 各Fold模型
└── *.png                             # IC趋势图、特征重要性图
```

融合输出:
```
experiments/{ensemble_id}/
├── smoothed_predictions.parquet      # 融合后test预测
├── smoothed_live_predictions.parquet # 融合后实盘预测
└── fusion_config.yaml                # 融合配置+权重
```

## 配置说明

关键配置项(`configs/*.yaml`):

```yaml
data:
  label:
    horizon: 20              # 预测周期
    use_open_price: true     # 使用开盘价(V1关键)

walk_forward:
  train_window: "3Y"         # 训练集3年
  valid_window: "6M"         # 验证集6个月
  test_window: "3M"          # 测试集3个月
  gap_train_valid: 21        # 隔离天数(=horizon+1)
  gap_valid_test: 21

model:
  name: "lightgbm"           # lightgbm或lightgbm_rank
  params:
    learning_rate: 0.02
    num_leaves: 63
```

## 文档导航

- [03.1_设计原理与逻辑架构.md](./docs/03.1_设计原理与逻辑架构.md) - 架构设计、数据流、核心决策
- [03.2_工程实现与规范.md](./docs/03.2_工程实现与规范.md) - API说明、数据格式、开发规范
- [03.3_运维与变更日志.md](./docs/03.3_运维与变更日志.md) - 版本变更、运维记录

## 模块结构

```
03模型训练层/
├── configs/              # 配置文件
│   ├── horizon5_config.yaml
│   ├── horizon20_config.yaml
│   ├── horizon60_config.yaml
│   └── rank_config.yaml
├── dataset/              # 数据构造
│   ├── data_constructor_v1.py    # X,y构造+标签计算
│   └── splitter_v1.py            # Walk-forward切分
├── models/               # 模型实现
│   ├── lightgbm_model.py         # 回归模型
│   └── lightgbm_rank_model.py    # LambdaRank
├── training/             # 训练框架
│   └── walk_forward_trainer_v1.py
├── main_train_v1.py      # 训练入口
├── fuse_predictions.py   # 多模型融合
└── experiments/          # 实验输出(gitignore)
```

## 常见问题

**Q: 为什么IC比之前低？**  
A: V1使用真实交易时点(开盘价)，旧版使用收盘价。虽然数值降低，但实盘可信度更高。

**Q: 融合权重为什么用base模型的actual？**  
A: 统一调仓周期，使IC可比。如以20d为base，5d模型的IC=pred_5d vs actual_20d。

**Q: 分界日期后为什么固定权重？**  
A: 长模型(60d)test结束后只有live预测，无actual_return，无法计算新IC。

---

**版本**: v1.0  
**更新**: 2026-03-25
