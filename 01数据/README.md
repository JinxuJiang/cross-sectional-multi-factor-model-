# QMT 日线数据下载系统

基于 **投研端 - 原生Python API** 的日线数据下载工具。

> **特点**: 可以在 VSCode/PyCharm 等IDE中直接运行，**不需要启动QMT客户端界面**。

## 🔧 环境准备

### 1. 安装依赖

```bash
pip install xtquant pandas pyarrow
```

### 2. 确认投研端权限

- 需要投研端VIP权限
- 确保账号可以连接VIP行情服务器

## 🚀 快速开始

### 步骤 1：示例测试（直接运行）

```bash
python main.py --sample
```

这会：
1. 自动连接行情服务器（默认绍兴电信）
2. 下载6只示例股票的日线数据
3. 保存到 `data/kline/daily/` 目录

### 步骤 2：下载全市场

```bash
python main.py --all
```

### 步骤 3：每日增量更新

```bash
python main.py --incremental
```

---

## 📁 项目结构

```
D:\数据\
├── data/                              # 数据目录
│   ├── kline/daily/                   # 日线数据
│   │   ├── sh/                        # 上海市场 (600xxx.parquet)
│   │   └── sz/                        # 深圳市场 (000xxx.parquet)
│   └── meta/                          # 元数据
│       ├── stock_list.csv             # 股票列表
│       └── download_log.json          # 下载记录
├── src/                               # 源代码
│   ├── config.py                      # 配置（含行情服务器地址）
│   ├── downloader.py                  # 下载器
│   ├── stock_pool.py                  # 股票池管理
│   └── utils.py                       # 工具函数
├── tests/
│   └── test_sample.py                 # 测试脚本
├── main.py                            # 主入口
└── README.md
```

---

## 📖 使用说明

### 命令行选项

```bash
python main.py --help

# 常用命令
python main.py --sample                          # 下载示例股票测试
python main.py --all                             # 下载全市场
python main.py --incremental                     # 增量更新
python main.py --list                            # 查看已下载
python main.py --server 郑州联通 --all            # 使用指定服务器
```

### 行情服务器选择

| 服务器 | 地址 | 适用网络 |
|--------|------|---------|
| 绍兴电信 | 115.231.218.73:55310 | 电信用户 |
| 郑州联通 | 42.228.16.211:55300 | 联通用户 |
| 郑州电信 | 36.99.48.20:55300 | 电信用户 |

### 在Python代码中使用

```python
from src.downloader import DailyDataDownloader
from src.stock_pool import StockPoolManager

# 连接行情服务器并下载
downloader = DailyDataDownloader()
downloader.download_and_save('000001.SZ')

# 批量下载
downloader.download_batch(['000001.SZ', '600000.SH'])

# 获取全市场股票池
pool = StockPoolManager()
all_stocks = pool.get_all_a_shares()

# 下载全市场
pool = StockPoolManager()
stocks = pool.get_all_a_shares()
batches = pool.batch_split(stocks, 50)

for i, batch in enumerate(batches):
    print(f"下载第 {i+1}/{len(batches)} 批")
    downloader.download_batch(batch)
```

---

## 📊 数据格式

### 单只股票数据（Parquet文件）

| 字段 | 说明 |
|------|------|
| `date` | 日期 |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |
| `preClose` | 前收盘价 |
| `change_pct` | 涨跌幅(%) |

### 读取数据

```python
import pandas as pd

# 读取单只股票
df = pd.read_parquet('data/kline/daily/sz/000001.parquet')
print(df.tail())
```

---

## ⚠️ 注意事项

1. **需要投研端VIP权限** - 普通权限可能无法连接VIP行情服务器
2. **首次运行会自动连接行情服务器** - 请确保网络畅通
3. **首次下载较慢** - 全市场约5000只股票，每只需要1-3秒
4. **增量更新** - 下载记录保存在 `data/meta/download_log.json`

---

## 🔄 完整工作流

```
┌────────────────────────────────────────┐
│  1. 确保已安装xtquant                   │
│     pip install xtquant                │
├────────────────────────────────────────┤
│  2. 示例测试                             │
│     python main.py --sample             │
├────────────────────────────────────────┤
│  3. 下载全市场                           │
│     python main.py --all                │
├────────────────────────────────────────┤
│  4. 每日收盘后增量更新                    │
│     python main.py --incremental        │
└────────────────────────────────────────┘
```

---

## 🔧 核心API

### connect - 连接行情服务器

```python
import xtdata

# 连接投研端行情服务器
xtdata.connect('115.231.218.73', 55310)
```

### download_history_data - 下载历史数据

```python
xtdata.download_history_data(
    stock_code='000001.SZ',
    period='1d',
    start_time='20240101',
    end_time=''
)
```

### get_market_data - 获取市场数据

```python
data = xtdata.get_market_data(
    field_list=['open', 'high', 'low', 'close'],
    stock_list=['000001.SZ'],
    period='1d',
    count=-1
)
```

### get_stock_list_in_sector - 获取板块股票列表

```python
stocks = xtdata.get_stock_list_in_sector('沪深A股')
```

---

## 📝 任务进度

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | 搭建文件夹结构 | ✅ |
| 2 | 编写投研端原生Python下载器 | ✅ |
| 3 | 编写股票池管理 | ✅ |
| 4 | **示例测试** | ⏳ 运行 `python main.py --sample` |
| 5 | **全量下载** | ⏳ 运行 `python main.py --all` |
| 6 | 分钟线数据（后续） | 📋 |
| 7 | 财务数据（后续） | 📋 |

---

## 📚 参考文档

- 投研使用教程: https://dict.thinktrader.net/freshman/ty_rookie.html
- QMT使用指南: https://dict.thinktrader.net/freshman/rookie.html
- QMT API 知识库见 `01_QMT_API知识库/` 目录
