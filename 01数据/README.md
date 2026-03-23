# 01 数据层 (Data Engine)

> 从 QMT 下载原始行情与财务数据，存储为标准化 Parquet 格式。

---

## 目录结构

### 代码文件（本层提交的内容）
```
01数据/
├── Base_DataEngine.py      # 核心数据引擎，封装 QMT API
├── monthly_update.py       # 月度增量更新入口
├── data_main.py            # 首次全量下载入口
└── README.md               # 本文档
```

### 运行时自动生成的目录
首次运行后自动创建，**无需手动创建，也不提交到Git**：
```
01数据/
└── data/                           # 【运行时自动生成】
    └── raw_data/
        ├── market_data/            # 行情数据（个股 parquet）
        ├── financial_data/         # 财务数据（个股 parquet）
        ├── industry_map.csv        # 申万一级行业映射
        ├── stock_info.parquet      # 股票基础信息
        └── update_log.json         # 更新记录
```

---

## 快速开始

### 1. 环境准备

```bash
# 依赖: xtquant, pandas, pyarrow
pip install xtquant pandas pyarrow
```

**权限要求**: 迅投 QMT 投研端 VIP 账号

### 2. 首次全量下载

```bash
cd 01数据
python data_main.py --full
```

**说明**: 
- 自动连接 QMT 行情服务器
- 下载全 A 股 (~5000+) 行情与财务数据
- 耗时约 2-4 小时（取决于网络）

### 3. 月度增量更新

```bash
cd 01数据
python data_main.py --monthly
```

**更新策略**:

| 数据类型 | 策略 | 原因 | 耗时 |
|:---|:---|:---|:---|
| 行情 | 全量覆盖 | 复权价格漂移需修正 | 5分钟 |
| 财务 | 智能合并 | 历史数据静态不变 | 40-60分钟 |
| 元数据 | 重新下载 | 股票列表会变化 | <1分钟 |

---

## 数据规范

### 行情数据

| 属性 | 说明 |
|:---|:---|
| **位置** | `data/raw_data/market_data/{code}.parquet` |
| **字段** | `time`, `open`, `high`, `low`, `close`, `volume`, `amount`, `preClose`, `suspendFlag` |
| **复权** | 等比前复权 (`front_ratio`) |
| **频率** | 日频 |

### 财务数据

| 属性 | 说明 |
|:---|:---|
| **位置** | `data/raw_data/financial_data/{code}.parquet` |
| **关键字段** | `report_date`: 报告期, `m_anntime`: 公告日（PIT 对齐用） |
| **格式** | PIT (Point-in-Time)，四表合并后 323 列 |
| **特点** | 大量 NaN 正常（不同表字段互补）|

---

## 核心类说明

| 类/函数 | 文件 | 用途 |
|:---|:---|:---|
| `DataEngine` | `Base_DataEngine.py` | 连接 QMT、下载数据、错误重试 |
| `MonthlyDataUpdater` | `monthly_update.py` | 继承 DataEngine，实现智能增量更新 |

### 代码示例

```python
from Base_DataEngine import DataEngine

engine = DataEngine()

# 下载行情
engine.download_market_data(
    stock_list=['000001.SZ', '600000.SH'],
    start_time='20200101',
    end_time='20241231'
)

# 下载财务
engine.download_financial_data(
    stock_list=['000001.SZ', '600000.SH'],
    start_time='20200101'
)
```

---

## 常用命令

```bash
# 首次使用：全量下载所有历史数据
cd 01数据
python data_main.py --full

# 日常维护：月度增量更新
python data_main.py --monthly

# 避免获取当日未收盘数据
python data_main.py --full --end-date 20260318

# 自定义财务数据起始日期
python data_main.py --monthly --financial-start 20240101

# 查看上次更新时间
cat data/raw_data/update_log.json
```

---

## 注意事项

| 注意点 | 说明 |
|:---|:---|
| **分批下载** | 每批 300 只股票，避免 API 超时 |
| **错误重试** | 单只股票失败自动重试 3 次，不影响其他 |
| **PIT 对齐** | 财务数据用 `m_anntime` 对齐，同比计算用 `report_date` |
| **两步清洗** | 同一报告期保留首次公告（防未来函数），同天多报告保留最新报告期 |
| **时间格式** | `time` 字段是 UTC 毫秒戳，需转换为北京时间 |
| **前复权漂移** | 每月需全量重新下载行情，修正历史复权价格 |

---

## 详细文档

- [设计原理与逻辑架构](../docs/01.1_设计原理与逻辑架构.md)
- [工程实现与规范](../docs/01.2_工程实现与规范.md)
- [运维与变更日志](../docs/01.3_运维与变更日志.md)
