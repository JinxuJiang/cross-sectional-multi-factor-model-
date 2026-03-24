# 02因子库 (Alpha Factory)

> 截面多因子量化选股系统 - 因子计算与数据清洗层

## 简介

本模块是量化选股系统的**特征工程核心**，负责从原始行情/财务数据计算技术/财务因子，并进行标准化清洗，输出可用于模型训练的标准化因子。

**核心设计原则**：
- **因果逻辑优先**：每个因子必须有清晰的经济学解释
- **无未来函数**：PIT对齐确保财报公告后才可用
- **One Factor One File**：每个因子独立存储，便于按需加载
- **内嵌清洗**：原始因子不输出，直接输出标准化因子

## 目录结构

```
02因子库/
├── update_all.py                    # 一键全量更新脚本
│
├── src/                             # 源代码
│   ├── data_engine/                 # 数据预处理（PIT对齐）
│   │   ├── market_data_loader.py    # 行情数据宽表化
│   │   ├── financial_data_loader.py # 财务数据TTM+PIT对齐
│   │   ├── pit_aligner.py           # PIT对齐核心算法
│   │   ├── industry_loader.py       # 行业数据加载
│   │   ├── main_prepare_market_data.py      # 行情数据入口
│   │   └── main_prepare_financial_data.py   # 财务数据入口
│   │
│   ├── alpha_factory/               # 因子计算
│   │   ├── technical/               # 技术因子家族
│   │   │   ├── momentum.py          # 动量因子 (ret1/5/20/60/120)
│   │   │   ├── volatility.py        # 波动率因子 (std20/60, atr20)
│   │   │   ├── liquidity.py         # 流动性因子 (amihud)
│   │   │   ├── price_volume.py      # 价量因子
│   │   │   └── main_compute_technical.py    # 统一计算入口
│   │   │
│   │   └── financial/               # 财务因子家族
│   │       ├── valuation.py         # 估值因子 (PE/PB/PS)
│   │       ├── profitability.py     # 盈利因子 (ROE/ROA)
│   │       ├── growth.py            # 成长因子
│   │       ├── quality.py           # 质量因子
│   │       ├── safety.py            # 安全因子
│   │       ├── investment.py        # 投资因子
│   │       ├── efficiency.py        # 效率因子
│   │       └── main_compute_financial.py    # 统一计算入口
│   │
│   └── processors/                  # 数据清洗流程
│       ├── pipeline.py              # 清洗流程串联
│       ├── outlier.py               # MAD去极值
│       ├── missing_value.py         # 缺失值填补
│       ├── neutralizer.py           # OLS中性化
│       └── standardizer.py          # Z-Score标准化
│
├── processed_data/                  # 输出数据
│   ├── market_data/                 # 行情基础宽表
│   │   ├── close.parquet
│   │   ├── volume.parquet
│   │   └── ...
│   ├── financial_data/              # 财务基础宽表（PIT对齐）
│   │   ├── cap_stk.parquet
│   │   ├── net_profit_ttm.parquet
│   │   ├── industry.parquet
│   │   └── ...
│   └── factors/                     # 因子数据（清洗后）
│       ├── technical/               # 技术因子
│       │   ├── ret20.parquet
│       │   ├── std20.parquet
│       │   └── ...
│       └── financial/               # 财务因子
│           ├── pe.parquet
│           ├── roe.parquet
│           └── ...
│
└── test/                            # 测试脚本
    ├── test_main_technical_fixed.py
    ├── test_financial_loader_simple.py
    └── ...
```

## 快速开始

### 环境要求

- Python 3.8+
- 内存：16GB+（全量计算时）
- 磁盘：~3GB（输出数据）

### 一键全量更新

```bash
# 从项目根目录运行
python 02因子库/update_all.py
```

这将按顺序执行：
1. 行情数据宽表化
2. 财务数据PIT对齐
3. 技术因子计算+清洗
4. 财务因子计算+清洗

**预计耗时**：~30分钟（i7-12700H，5000只×3000日）

### 分步执行（推荐首次使用）

```bash
# Step 1: 准备基础数据
cd 02因子库/src/data_engine
python main_prepare_market_data.py --overwrite
python main_prepare_financial_data.py --overwrite

# Step 2: 计算技术因子
cd ../alpha_factory/technical
python main_compute_technical.py

# Step 3: 计算财务因子
cd ../financial
python main_compute_financial.py
```

### 计算单个因子

```python
from src.alpha_factory.technical.momentum import MomentumFactors

mf = MomentumFactors()
mf.factor_ret20(save=True)  # 计算并保存ret20因子
```

## 核心概念

### 1. 数据流向

```
┌─────────────────────────────────────────────────────────────────────┐
│                        原始数据 (01数据层)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │market_data/  │  │financial_data│  │industry_map  │              │
│  │(个股日频)    │  │(个股季度)    │  │(行业映射)    │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼────────────────┼────────────────┼────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Engine 数据引擎                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │market_data_loader│  │financial_data_  │  │industry_loader  │      │
│  │  (宽表转换)      │  │  (TTM+PIT对齐)   │  │  (宽表转换)      │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────┬────────────────┬────────────────┬─────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PIT对齐后的基础数据                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │market_data/      │  │financial_data/   │  │industry.parquet │   │
│  │ close.parquet    │  │ cap_stk.parquet  │  │ (申万31行业)     │   │
│  │ volume.parquet   │  │ net_profit_ttm   │  └─────────────────┘   │
│  │ ...              │  │ ...              │                        │
│  └──────────────────┘  └──────────────────┘                        │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Alpha Factory 因子工厂                            │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐                               │
│   │ technical/   │  │ financial/   │                               │
│   │ 技术因子家族  │  │ 财务因子家族  │                               │
│   │ • momentum   │  │ • valuation  │                               │
│   │ • volatility │  │ • profitability│                              │
│   │ • liquidity  │  │ • growth     │                               │
│   └──────┬───────┘  └──────┬───────┘                               │
│          │                 │                                       │
│          └────────┬────────┘                                       │
│                   ▼                                                │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │                    Processors 清洗流程                    │    │
│   │  MAD去极值 → 填补缺失 → OLS中性化 → Z-Score标准化        │    │
│   │   (3倍MAD)   (行业中位数)  (剥离行业+市值Beta)  (N(0,1))  │    │
│   └────────────────────────┬─────────────────────────────────┘    │
│                            │                                       │
│                            ▼                                       │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │              清洗后因子 (One Factor One File)             │    │
│   │  factors/technical/    factors/financial/                │    │
│   │   ret20.parquet         pe.parquet                       │    │
│   │   std20.parquet         roe.parquet                      │    │
│   │   (均值≈0, 标准差≈1)   (均值≈0, 标准差≈1)              │    │
│   └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. PIT（Point-in-Time）对齐

**问题**：财务数据是季度报告，如何避免未来函数？

**方案**：按公告日（m_anntime）前向填充

```
财报1(2010Q1, 4/29公告) ─────── 财报2(2010Q2, 8/25公告)
     │                              │
     ▼                              ▼
2010-04-29 ~ 2010-08-24       2010-08-25 ~ 2010-10-27
   使用财报1数据                 使用财报2数据
```

**关键原则**：公告日后市场才知晓数据，因此公告日前只能用上一期数据。

### 3. 因子清洗流程（四步顺序不能变）

```
原始因子（含极端值、行业偏误）
   ↓
1. MAD去极值（3倍MAD缩尾）      ← 必须先做
   ↓
2. 缺失值填补（行业中位数）      ← 缺行业/市值的剔除
   ↓
3. OLS中性化（剥离行业+市值Beta）
   ↓
4. Z-Score标准化（均值0，标准差1） ← 必须最后做
   ↓
清洗后因子
```

### 4. 技术因子 vs 财务因子

| 对比项 | 技术因子 | 财务因子 |
|:---|:---|:---|
| 数据来源 | 行情数据（日频） | 财务数据（季度） |
| 预处理 | 宽表转换 | TTM计算 + PIT对齐 |
| 计算逻辑 | 时序运算（shift/rolling） | 财务公式（乘除法） |
| 市值依赖 | 不需要 | 需要（close × cap_stk） |
| 未来函数风险 | 低（注意使用preClose） | 高（必须PIT对齐） |

## 使用指南

### 新增因子

```python
# Step 1: 在对应家族添加方法
def factor_xxx(self, save=True):
    """
    因子名称 - 一句话描述
    
    因果逻辑：
    ---------
    解释为什么这个因子能预测收益
    
    公式：
    ------
    因子 = xxx / yyy
    """
    # 实现计算逻辑
    result = ...
    return result

# Step 2: 添加到注册表
TECHNICAL_FACTORS = {
    'xxx': {'family': 'momentum', 'method': 'factor_xxx', 'desc': '描述'},
}

# Step 3: 运行测试
python main_compute_technical.py --factors xxx
```

### 检查因子质量

```python
import pandas as pd

# 加载因子
factor = pd.read_parquet('processed_data/factors/technical/ret20.parquet')

# 检查分布
print(f"均值: {factor.mean():.4f}")      # 应≈0
print(f"标准差: {factor.std():.4f}")     # 应≈1
print(f"缺失值: {factor.isna().sum().sum()}")

# 检查未来函数（最后一天是否全为NaN，取决于因子类型）
print(f"最后一天非NaN数: {factor.iloc[-1].notna().sum()}")
```

### 跳过清洗快速测试

```bash
# 只计算原始因子，不清洗（节省内存）
python main_compute_technical.py --skip-clean
```

## 现有因子清单

### 技术因子（22个）

| 家族 | 因子 | 说明 |
|:---|:---|:---|
| 动量 | ret1/5/20/60/120 | N日收益率 |
| 动量 | ret20_60 | 动量差（ret20 - ret60） |
| 波动率 | std20/60 | N日标准差 |
| 波动率 | atr20 | 20日平均真实波幅 |
| 波动率 | volatility_regime | 波动率状态 |
| 流动性 | amihud | Amihud非流动性 |
| 价量 | close_position | 收盘价日内位置 |

### 财务因子（25个）

| 家族 | 因子 | 说明 |
|:---|:---|:---|
| 估值 | pe/pb/ps/ey | 市盈率/市净率/市销率/盈利收益率 |
| 盈利 | roe/roa/opm/gross_margin | 净资产收益率/总资产收益率/营业利润率/毛利率 |
| 成长 | profit_growth/revenue_growth | 净利润/营收增长率 |
| 质量 | accrual/cashflow_to_profit | 应计利润比/现金流利润比 |
| 安全 | debt_to_equity/current_ratio/cash_ratio | 产权比率/流动比率/现金比率 |
| 投资 | asset_growth/capex_to_assets | 总资产增长率/资本支出强度 |
| 效率 | asset_turnover/working_capital_ratio | 资产周转率/营运资本占比 |

## 相关文档

| 文档 | 内容 |
|:---|:---|
| [02.1_设计原理与逻辑架构.md](../docs/02.1_设计原理与逻辑架构.md) | 数据流详解、PIT对齐算法、TTM计算 |
| [02.2_工程实现与规范.md](../docs/02.2_工程实现与规范.md) | API接口、数据规范、开发注意事项 |
| [02.3_运维与变更日志.md](../docs/02.3_运维与变更日志.md) | 检查点、性能基准、变更记录 |

## 注意事项

1. **内存管理**：全量计算时内存峰值约2GB，建议关闭其他应用
2. **数据对齐**：所有宽表必须行列对齐（日期×股票），否则计算会出错
3. **清洗顺序**：四步清洗顺序不能变，否则影响结果
4. **PIT原则**：财务因子必须使用公告日对齐，不能用报告期

## 常见问题

**Q: 计算时提示"有效样本少于3，跳过中性化"？**

A: 正常。早期数据（2010年）TTM数据不足，导致PE等因子为NaN。从2011年开始数据逐渐完善。

**Q: 如何只计算部分因子？**

A: 使用 `--family` 或 `--factors` 参数：
```bash
python main_compute_technical.py --family momentum
python main_compute_financial.py --factors pe pb roe
```

**Q: 修改代码后如何重算？**

A: 删除旧数据后重新运行：
```bash
rm processed_data/factors/technical/ret20.parquet
python main_compute_technical.py --factors ret20
```

---

*最后更新: 2026-03-24*  
*维护者: 蒋大王*
