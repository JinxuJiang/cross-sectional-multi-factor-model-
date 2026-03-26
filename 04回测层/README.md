# 04 回测层 (Backtest & Optimization) 📈

> 截面多因子量化选股系统 - 预测分数验证与策略绩效评估层

---

## ✨ 核心能力

| 能力 | 说明 |
|:---|:---|
| 🔬 双轨验证 | Alphalens因子检验(IC/IR/分层收益) + Backtrader策略回测(净值/夏普/回撤) |
| 🚫 涨停过滤 | 自动过滤开盘涨停股(涨幅>=9.9%)，避免买入不可成交股票 |
| 📉 平滑预测支持 | 支持半衰期平滑后的预测回测，显著降低换手率 |
| 🛡️ 风险控制 | 15%止损线 + 主板过滤(60/00开头) + 等权90%仓位管理 |

---

## 🏗️ 架构概览

```
模型预测分数 (predictions.parquet)
           │
           ├──→ Alphalens 分析层 ──→ 因子层面验证
           │                              │
           │                         • Rank IC / IR
           │                         • 分层收益（单调性）
           │                         • 换手率/稳定性
           │                              │
           │                              ▼
           │                    ┌──────────────────┐
           │                    │ 因子是否有效？    │
           │                    │ IC > 0.03？      │
           │                    │ IR > 0.5？       │
           │                    └────────┬─────────┘
           │                             │
           ▼                             ▼
    Backtrader 回测层 ──→ 策略层面验证
                              │
                         • 调仓执行（每月首交易日）
                         • 涨停过滤 + 主板过滤
                         • 成本/滑点/止损
                              │
                              ▼
                    ┌──────────────────┐
                    │ 策略是否可盈利？  │
                    │ 夏普 > 1？       │
                    │ 回撤 < 20%？     │
                    └──────────────────┘
```

---

## 📁 目录结构

```
04回测层/
├── alphalens_analysis.py       # Alphalens因子有效性分析
├── backtrader.eval.py          # Backtrader策略回测 V1.1
├── utils.py                    # 公共工具函数
│
├── reports/                    # 回测报告输出目录（运行时生成）
│   └── {exp_id}/
│       ├── alphalens_report.html       # Alphalens分析汇总报告
│       ├── ic_analysis.png             # IC时间序列/分布图
│       ├── returns_analysis.png        # 分层收益分析图
│       ├── turnover_analysis.png       # 换手率/稳定性分析图
│       ├── backtest_report.html        # Backtrader回测汇总报告
│       ├── equity_curve.png            # 净值曲线
│       ├── trades.csv                  # 详细交易记录
│       └── performance.json            # 绩效指标（JSON格式）
│
└── test/
    └── check_future_data.py    # 未来函数检查工具
```

---

## 🔄 数据流与逻辑

### 🔬 Alphalens 分析流

```
predictions.parquet (模型预测)
        │
        ▼
┌──────────────────┐
│  load_predictions │  ← 加载预测分数（支持平滑预测）
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ load_close_prices │  ← 加载收盘价（计算远期收益）
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ get_clean_factor_and_forward_returns    │  ← Alphalens核心函数
│   • factor: MultiIndex (date, asset)    │
│   • prices: DataFrame (date × asset)    │
│   • periods: (20,)  # 20日远期收益      │
└──────────────────┬──────────────────────┘
                   │
                   ├──→ factor_information_coefficient() → Rank IC
                   ├──→ mean_return_by_quantile() → 分层收益
                   └──→ factor_rank_autocorrelation() → 换手率
```

### 📈 Backtrader 回测流

```
predictions.parquet + market_data/*.parquet
        │
        ▼
┌──────────────────┐
│ load_and_merge_data │  ← 合并预测+行情数据
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ generate_signals    │  ← 生成调仓信号（每月首交易日）
│  • 主板过滤        │     只保留60/00开头
│  • 涨停过滤        │     过滤open_return >= 9.9%
│  • Top N 选股      │     按prediction排序
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│ MyMultiFactorStrategy   │  ← Backtrader策略类
│  • next()每日执行       │
│    - 检查15%止损（优先） │
│    - 调仓日买卖         │
│  • notify_order()记录交易│
└────────┬────────────────┘
         │
         ▼
┌──────────────────┐
│ 绩效分析器         │  ← Sharpe/DrawDown/Returns/TradeAnalyzer
└────────┬─────────┘
         │
         ▼
    performance.json + equity_curve.png + backtest_report.html
```

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
pip install alphalens-reloaded backtrader matplotlib pandas numpy
```

### 2️⃣ 运行入口

```bash
# ========== Alphalens 因子分析 ==========

# 分析原始预测
python 04回测层/alphalens_analysis.py --exp-id exp_001

# 分析平滑后的预测（推荐，换手率更低）
python 04回测层/alphalens_analysis.py --exp-id exp_001 --use-smooth


# ========== Backtrader 策略回测 ==========

# 使用原始预测回测
python 04回测层/backtrader.eval.py --exp-id exp_001

# 使用平滑预测回测（推荐）
python 04回测层/backtrader.eval.py --exp-id exp_001 --use-smooth
```

**回测参数**（编辑 `backtrader.eval.py` 中 `STRATEGY_PARAMS`）：
```python
STRATEGY_PARAMS = {
    'stocks_per_batch': 20,           # 每次选股数量
    'start_date': datetime(2020, 1, 1),  # 回测开始日期
    'end_date': datetime(2026, 3, 31),   # 回测结束日期
    'initial_cash': 50000,           # 初始资金
    'commission': 0.002               # 手续费率 (0.2%)
}
```

---

## 🔑 关键设计

### 🔬 双轨验证的必要性

| 维度 | Alphalens | Backtrader |
|:---|:---|:---|
| 关注点 | 因子预测能力 | 策略可执行性 |
| 核心指标 | IC/IR、分层收益 | 夏普比率、最大回撤 |
| 交易假设 | 理想化（无成本、无滑点） | 真实交易（成本+滑点+止损） |
| 用途 | 模型迭代参考 | 实盘前最终验证 |
| 通过标准 | IC > 0.03, IR > 0.5 | 夏普 > 1, 回撤 < 20% |

**结论**：两者互补，缺一不可。
- IC高但实盘差 → 执行问题（成本、滑点、涨停等）
- IC低但实盘好 → 运气/过拟合

### 🚫 涨停过滤逻辑

**问题**：回测时若选到涨停股，实际无法买入，导致回测收益虚高。

**解决方案**：
```python
# 计算T+1开盘涨幅
merged['open_return'] = merged['open_t1'] / merged['close_t'] - 1

# 过滤开盘涨停（>=9.9%）
tradable_stocks = merged.loc[merged['open_return'] < 0.099, 'stock_code']
```

**效果**：更真实的回测结果，避免过度乐观。

### 🛡️ 止损逻辑

```python
def next(self):
    # 每日检查止损（在调仓逻辑之前执行）
    for data in self.datas:
        pos = self.getposition(data)
        if pos.size > 0:
            cost_price = pos.price
            current_price = data.close[0]
            
            # 跌超15%止损
            if current_price < cost_price * (1 - 0.15):
                self.order_target_percent(data=data, target=0)
```

**优先级**：止损 > 调仓，风险控制优先。

### 📉 平滑预测的优势

| 指标 | 原始预测 | 平滑预测 |
|:---|:---|:---|
| Rank IC | 较高 | 略降（~10%） |
| 换手率 | 高 | **降低30-50%** |
| 信号稳定性 | 波动大 | **更稳定** |
| 实盘交易成本 | 高（频繁调仓） | **更低** |

**建议**：回测时对比 `--use-smooth` 前后的结果，选择合适的参数。

---

## 📊 数据规范

### 输入

| 数据 | 路径 | 格式 |
|:---|:---|:---|
| 预测分数 | 03模型训练层/experiments/{exp_id}/predictions.parquet | 长表 [date, stock_code, pred_score] |
| 平滑预测 | 03模型训练层/experiments/{exp_id}/smoothed_predictions.parquet | 长表 [date, stock_code, pred_score_smooth] |
| 行情数据 | 02因子库/processed_data/market_data/*.parquet | 宽表 [date × stock_code] |

### 输出

**Alphalens报告**：
| 文件 | 说明 |
|:---|:---|
| alphalens_report.html | 汇总报告（IC/IR/分层收益/换手率） |
| ic_analysis.png | IC时间序列、分布、累计曲线 |
| returns_analysis.png | 分位数收益柱状图、多空收益曲线 |
| turnover_analysis.png | 因子稳定性分析 |

**Backtrader报告**：
| 文件 | 说明 |
|:---|:---|
| backtest_report.html | 汇总报告（净值曲线、风险指标、交易记录） |
| equity_curve.png | 策略净值曲线图 |
| trades.csv | 详细交易记录（日期/代码/买卖/价格/股数） |
| performance.json | 绩效指标（夏普/回撤/胜率等） |

### performance.json 字段

```json
{
  "exp_id": "exp_001",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "initial_cash": 50000,
  "final_value": 82345.67,
  "total_return_pct": 64.69,
  "annual_return_pct": 12.34,
  "sharpe_ratio": 1.23,
  "max_drawdown_pct": 15.67,
  "total_trades": 480,
  "win_rate": 52.3
}
```

---

## 📚 详细文档

- [04.1_设计原理与逻辑架构](../docs/04.1_设计原理与逻辑架构.md) - 架构设计、数据流、核心决策
- [04.2_工程实现与规范](../docs/04.2_工程实现与规范.md) - API说明、数据规范、开发注意事项
- [04.3_运维与变更日志](../docs/04.3_运维与变更日志.md) - 检查点、性能基准、变更记录

---

*最后更新: 2026-03-26*  
*维护者: 蒋大王*
