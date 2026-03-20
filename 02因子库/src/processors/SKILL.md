# Processors 因子清洗模块 - 知识点与架构文档

> 创建时间: 2026-02-27
> 用途: 记录因子清洗流程的关键决策、算法细节和使用规范

---

## 一、模块架构

```
processors/
├── __init__.py              # 包初始化，导出主要函数
├── outlier.py               # 去极值: MAD缩尾法
├── missing_value.py         # 缺失值填补: 行业中位数法
├── neutralizer.py           # 中性化: OLS残差法
├── standardizer.py          # 标准化: Z-Score法
└── pipeline.py              # 流程串联: 组合以上步骤
```

### 职责分离原则

| 模块 | 职责 | 不涉及 |
|------|------|--------|
| outlier.py | 识别并处理极端值 | 不处理缺失值、不做中性化 |
| missing_value.py | 填补缺失值 | 不改变非缺失值 |
| neutralizer.py | 剥离行业/市值Beta | 不去极值、不标准化 |
| standardizer.py | 统一量纲 | 不改变分布形状 |
| pipeline.py | 按序调用各模块 | 不实现具体算法 |

---

## 二、处理流程（4步标准流程）

```
原始因子数据 (Raw Factor)
    │
    ▼
┌─────────────────────────────────────┐
│ Step 1: Outlier (去极值)            │
│ - 方法: MAD缩尾法 (5倍MAD)          │
│ - 目的: 防止异常值扭曲后续计算       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 2: Missing Value (填补缺失)    │
│ - 规则1: 缺失市值的整行剔除          │
│ - 规则2: 其他缺失用行业当日中位数填补│
│ - 目的: 确保OLS数据矩阵完整          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 3: Neutralizer (中性化)        │
│ - 方法: OLS残差法                   │
│ - 控制变量: 行业dummy + 对数市值     │
│ - 计算方式: 每日截面回归             │
│ - 目的: 剥离Beta，提取Alpha          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 4: Standardizer (标准化)       │
│ - 方法: Z-Score                     │
│ - 分布: N(0,1)                      │
│ - 计算方式: 每日截面标准化           │
│ - 目的: 统一量纲，便于AI模型输入     │
└─────────────────────────────────────┘
    │
    ▼
清洗后因子数据 (Clean Factor)
```

---

## 三、各步骤详细规范

### 3.1 Step 1: Outlier (去极值)

**算法**: MAD (Median Absolute Deviation) 缩尾法

**公式**:
```
median = factor.median()
mad = (factor - median).abs().median()
lower_bound = median - 5 × 1.4826 × mad
upper_bound = median + 5 × 1.4826 × mad

# 处理
factor[factor < lower_bound] = lower_bound
factor[factor > upper_bound] = upper_bound
```

**关键参数**:
- **MAD倍数**: 3倍（更严格，去除更极端的值）
- **常数1.4826**: 使MAD与标准差在正态分布下一致

**注意事项**:
- 使用**当日截面数据**计算MAD
- 缺失值不参与MAD计算，但保持位置
- 缩尾而非删除（保留样本量）

---

### 3.2 Step 2: Missing Value (填补缺失)

**处理规则**（优先级从高到低）:

| 情况 | 处理方式 | 原因 |
|------|----------|------|
| 缺失市值 | **整行剔除** | 市值是中性化必需变量 |
| 缺失行业 | **整行剔除** | 行业是中性化必需变量 |
| 缺失因子值 | 用**行业当日中位数**填补 | 同行业公司具有相似性 |
| 缺失其他字段 | 用**行业当日中位数**填补 | 保持一致性 |

**填补方法**:
```python
# 按行业分组，用中位数填补
grouped = df.groupby('industry')
df[factor] = grouped[factor].transform(lambda x: x.fillna(x.median()))
```

**重要原则**:
- **保留索引**: 不重新排序，不改变股票代码索引
- **当日截面**: 只使用当天同行业的数据计算中位数
- **剔除优先**: 关键变量（市值、行业）缺失时直接剔除，不填补

---

### 3.3 Step 3: Neutralizer (中性化)

**算法**: OLS线性回归残差法

**模型**:
```
因子 = β₀ + β₁×行业_dummy₁ + ... + βₙ×行业_dummyₙ + βₘ×对数市值 + ε
输出: ε (残差，即中性化后的Alpha)
```

**变量说明**:
- **因变量**: 待中性化的因子（经过Step1、Step2处理后）
- **自变量1**: 行业虚拟变量（dummy variables）
- **自变量2**: 对数市值（log(market_cap)）

**对数市值原因**:
- 市值分布右偏，取对数后更接近正态
- 减少极端大市值的影响

**截面回归说明**:
```python
for date in trading_dates:
    # 只取当天数据
    day_data = factor.loc[date]
    
    # 当天截面OLS
    model = OLS(y=day_data.factor, X=day_data[行业dummy, 对数市值])
    residuals = model.resid
    
    # 保存结果
    neutralized_factor.loc[date] = residuals
```

**剔除规则**:
- 回归前检查：缺失行业或市值的已在前一步剔除
- 回归后检查：若某行业当天只有1只股票，该行业dummy会共线，需特殊处理（通常为剔除或合并）

---

### 3.4 Step 4: Standardizer (标准化)

**算法**: Z-Score标准化

**公式**:
```
z = (x - μ) / σ

其中:
μ = 当日截面的因子均值
σ = 当日截面的因子标准差
```

**输出分布**:
- 均值 = 0
- 标准差 = 1
- 近似 N(0,1) 正态分布

**特殊情况处理**:
```python
if σ < 1e-10:  # 标准差接近0
    factor_clean = 0  # 所有值设为0，表示无差异
else:
    factor_clean = (x - μ) / σ
```

**原因**: 如果标准差为0（所有值相同），标准化无意义，设为0表示该因子当天对所有股票无区分度。

---

## 四、统一接口规范

### 4.1 数据格式约定

**输入数据**:
- **类型**: `pd.Series` 或 `pd.DataFrame`
- **索引**: 
  - 如果是Series: index=股票代码 (如 '000001.SZ')
  - 如果是DataFrame: index=日期, columns=股票代码
- **列名**: 因子名称（如 'PE_TTM'）

**辅助数据**:
```python
# 行业数据 (必需用于中性化和填补)
industry: pd.Series
# index: 股票代码
# value: 行业分类 (如 '银行', '医药')

# 市值数据 (必需用于中性化)
market_cap: pd.Series  
# index: 股票代码
# value: 总市值 (元)
```

### 4.2 函数签名规范

**单个因子处理**（推荐用于调试）:
```python
def process_factor(
    factor: pd.Series,           # 因子值，index=股票代码
    industry: pd.Series,         # 行业分类
    market_cap: pd.Series,       # 市值
    date: Optional[str] = None   # 日期（用于日志）
) -> pd.Series:
    """处理单个截面的因子数据"""
    pass
```

**批量处理**（推荐用于生产）:
```python
def process_factor_wide(
    factor_df: pd.DataFrame,      # 宽表格式，index=日期, columns=股票代码
    industry_df: pd.DataFrame,    # 行业宽表
    market_cap_df: pd.DataFrame,  # 市值宽表
) -> pd.DataFrame:
    """处理宽表格式的因子数据"""
    pass
```

---

## 五、实现顺序与依赖关系

### 5.1 实现优先级

```
Phase 1 (基础算法):
├── outlier.py          # 最简单，独立实现
├── standardizer.py     # 最简单，独立实现
└── missing_value.py    # 需要行业分组逻辑

Phase 2 (核心算法):
└── neutralizer.py      # 依赖前三个，需要OLS

Phase 3 (流程串联):
└── pipeline.py         # 依赖前四个，负责编排
```

### 5.2 依赖关系图

```
                    pipeline.py
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   outlier.py    missing_value.py   neutralizer.py   standardizer.py
        │                │                │                │
        └────────────────┴────────────────┴────────────────┘
                               │
                               ▼
                        common utilities
                        (统计函数、检查工具)
```

---

## 六、使用示例（预期）

### 6.1 分步使用（调试时用）

```python
from processors import mad_winsorize, fill_missing, neutralize, zscore

# 原始因子
raw_pe = load_factor('PE_TTM')

# Step 1: 去极值
pe_outlier = mad_winsorize(raw_pe, n_mad=3)

# Step 2: 填补缺失
pe_filled = fill_missing(pe_outlier, industry, method='industry_median')

# Step 3: 中性化
pe_neutral = neutralize(pe_filled, industry, market_cap)

# Step 4: 标准化
pe_clean = zscore(pe_neutral)

# 保存
save_factor(pe_clean, 'PE_TTM_CLEAN')
```

### 6.2 一键清洗（生产用）

```python
from processors.pipeline import clean_factor

# 原始因子宽表
raw_factor = load_factor('PE_TTM')  # index=日期, columns=股票代码
industry = load_industry()          # 同上格式
market_cap = load_market_cap()      # 同上格式

# 一键清洗
clean_factor = clean_factor(
    raw_factor, 
    industry, 
    market_cap,
    steps=['outlier', 'missing', 'neutralize', 'standardize']
)

# 保存
save_factor(clean_factor, 'PE_TTM_CLEAN')
```

---

## 七、常见陷阱与注意事项

### 7.1 数据对齐问题

**陷阱**: 因子数据、行业数据、市值数据的索引不一致

**解决**: 每个步骤开始时使用 `reindex` 对齐
```python
industry = industry.reindex(factor.index)
market_cap = market_cap.reindex(factor.index)
```

### 7.2 全缺失行业处理

**陷阱**: 某天某行业的所有股票因子都缺失，中位数无法计算

**解决**: 检查行业股票数量，如果为0或全缺失，跳过该行业或整体剔除

### 7.3 中性化共线性

**陷阱**: 某天某行业只有1只股票，行业dummy与其他变量共线，OLS报错

**解决**: 检查行业股票数，少于2只的行业合并为"其他"类别

### 7.4 内存问题

**陷阱**: 宽表格式 (3918天 × 5000股票) 占用内存大

**建议**: 
- 使用 `float32` 而非 `float64`
- 分日期循环处理，避免一次性加载全量数据到OLS
- 使用 PyArrow  backend 的 pandas

### 7.5 未来函数泄露

**红线**: 清洗时绝对不能用未来数据！

**检查点**:
- MAD只用当天数据 ✓
- 行业中位数只用当天数据 ✓
- OLS只用当天截面 ✓
- 标准化只用当天数据 ✓

**禁止**:
- 用历史均值填补 ✗
- 用前后日期插值 ✗
- 跨天滚动计算 ✗

---

## 八、待确认事项

在编写代码前，请确认以下事项：

- [x] MAD倍数确认为3倍
- [ ] 中性化控制变量确认（行业 + 对数市值）
- [ ] 行业分类数据格式确认（申万一级？）
- [ ] 市值数据格式确认（总市值？流通市值？）
- [ ] 输出因子命名规范（如 PE_TTM → PE_TTM_CLEAN？）

---

## 九、版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-02-27 | 初始版本，定义4步流程和接口规范 |
