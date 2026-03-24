# -*- coding: utf-8 -*-
"""
财务数据加载器

将原始财务数据（季度报告）转换为 PIT 对齐后的日度宽表。

处理流程：
---------
1. 读取 market_data/close.parquet 获取交易日历
2. 遍历 financial_data/ 下的所有个股 parquet 文件
3. 对每个股票：
   - 读取原始财务数据
   - 如需TTM：先计算季度TTM（累计值→单季度→4季度求和）
   - PIT对齐到交易日历
4. 拼接成宽表（行：日期，列：股票代码）
5. 保存为 parquet 文件

输出字段：
---------
期末值字段（资产负债表）:
    - cap_stk: 总股本
    - tot_assets: 总资产
    - tot_shrhldr_eqy: 归属于母公司股东权益合计 (原始字段: tot_shrhldr_eqy_excl_min_int)
    - total_current_assets: 流动资产

TTM字段（利润表，4季度滚动求和）:
    - net_profit_ttm: 归母净利润_TTM (原始字段: net_profit_excl_min_int_inc)
    - revenue_ttm: 营业收入_TTM (原始字段: revenue)
    - oper_profit_ttm: 营业利润_TTM (原始字段: oper_profit)

注意：
- ROE_TTM 后续从 net_profit_ttm / tot_shrhldr_eqy 计算得出
- 原始数据中无 ROE 字段
- TTM计算逻辑：先算季度TTM，再PIT对齐到日度（确保同财报期内TTM值不变）
"""

import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd

from pit_aligner import PITAligner


class FinancialDataLoader:
    """
    财务数据加载器
    
    功能：
    1. 加载原始财务数据
    2. 计算 TTM 字段（在季度层面正确计算）
    3. PIT 对齐到交易日历
    4. 输出宽表格式
    
    参数：
    -----
    raw_data_path : str, optional
        原始财务数据路径
    market_data_path : str, optional
        行情数据路径（用于获取交易日历）
    output_path : str, optional
        输出路径
    """
    
    # 需要处理的字段配置
    # format: (输出字段名, 是否需要TTM, 原始数据字段名)
    # 如果原始字段名与输出名不同，需要在第三个位置指定
    FIELD_CONFIG = [
        # 资产负债表 - 期末值，不需要TTM
        ('cap_stk', False, 'cap_stk'),
        ('tot_assets', False, 'tot_assets'),
        ('tot_shrhldr_eqy', False, 'tot_shrhldr_eqy_excl_min_int'),  # 归属于母公司股东权益
        ('total_current_assets', False, 'total_current_assets'),
        
        # 利润表 - 需要TTM（4季度滚动求和）
        ('net_profit', True, 'net_profit_excl_min_int_inc'),  # 归母净利润
        ('revenue', True, 'revenue'),
        ('oper_profit', True, 'oper_profit'),
        
        # PershareIndex 每股指标 - 直接PIT对齐（QMT原始字段名）
        # 注意：以下字段来自QMT的PershareIndex表，使用实际字段名
        # [2026-03-24] 删除冗余字段：s_fa_ocfps, s_fa_bps, s_fa_eps_basic, s_fa_eps_diluted,
        # s_fa_undistributedps, net_profit_margin, inventory_turnover, du_return_on_equity,
        # equity_roe, net_roe, total_roe, actual_tax_rate, gear_ratio
        ('sales_gross_profit', False, 'sales_gross_profit'),    # 销售毛利率(对应gross_margin)
        
        # 新增基础数据字段（用于扩充因子库）
        # 资产负债表
        ('tot_liab', False, 'tot_liab'),                          # 总负债
        ('total_current_liability', False, 'total_current_liability'),  # 流动负债
        ('cash_equivalents', False, 'cash_equivalents'),          # 货币资金
        
        # 现金流量表（TTM）
        ('operating_cash_flow', True, 'net_cash_flows_oper_act'), # 经营现金流
        ('capex', True, 'cash_pay_acq_const_fiolta'),             # 购建固定资产现金
    ]
    
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        market_data_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        初始化财务数据加载器
        """
        # 路径设置
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent  # 因子库
        project_root = factor_lib_root.parent  # 截面多因子模型
        
        if raw_data_path is None:
            self.raw_data_path = project_root / '01数据' / 'data' / 'raw_data' / 'financial_data'
        else:
            self.raw_data_path = Path(raw_data_path)
            
        if market_data_path is None:
            self.market_data_path = factor_lib_root / 'processed_data' / 'market_data'
        else:
            self.market_data_path = Path(market_data_path)
            
        if output_path is None:
            self.output_path = factor_lib_root / 'processed_data' / 'financial_data'
        else:
            self.output_path = Path(output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 交易日历（从 market_data/close.parquet 读取）
        self.trading_calendar: List[datetime.date] = []
        self._load_trading_calendar()
        
        # PIT 对齐器
        self.aligner = PITAligner(self.trading_calendar)
        
    def _load_trading_calendar(self):
        """
        从 market_data/close.parquet 加载交易日历
        """
        close_file = self.market_data_path / 'close.parquet'
        if not close_file.exists():
            raise FileNotFoundError(
                f"收盘价数据不存在: {close_file}\n"
                f"请先运行 main_prepare_market_data.py 准备行情数据"
            )
        
        table = pq.read_table(close_file, columns=['time'])
        # 转换为 datetime 对象，确保与 market_data 格式一致
        time_list = table.column('time').to_pylist()
        # 如果是 datetime.date，转换为 datetime.datetime
        self.trading_calendar = []
        for t in time_list:
            if isinstance(t, datetime.date) and not isinstance(t, datetime.datetime):
                self.trading_calendar.append(datetime.datetime.combine(t, datetime.time.min))
            else:
                self.trading_calendar.append(t)
        print(f"已加载交易日历: {len(self.trading_calendar)} 个交易日")
        print(f"  日期范围: {self.trading_calendar[0]} ~ {self.trading_calendar[-1]}")
    
    def _get_stock_files(self) -> List[Path]:
        """
        获取所有财务数据文件列表
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"原始数据路径不存在: {self.raw_data_path}")
        
        files = sorted(self.raw_data_path.glob('*.parquet'))
        print(f"找到 {len(files)} 个财务数据文件")
        return files
    
    def _read_financial_data(self, file_path: Path) -> List[Dict]:
        """
        读取单个股票的财务数据
        
        返回：
        ------
        List[Dict] : 财务记录列表
        """
        try:
            table = pq.read_table(file_path)
            
            # 提取所有需要的列
            columns = table.column_names
            
            # 检查是否有公告日期字段
            if 'm_anntime' not in columns:
                print(f"警告: {file_path.name} 缺少 m_anntime 字段")
                return []
            
            # 转换为字典列表
            records = []
            n_rows = len(table)
            
            for i in range(n_rows):
                record = {}
                for col in columns:
                    val = table.column(col)[i].as_py()
                    record[col] = val
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return []
    
    def _calculate_ttm_from_cumulative(
        self,
        records: List[Dict],
        value_field: str
    ) -> List[Dict]:
        """
        从累计值计算 TTM (Trailing Twelve Months)
        
        正确算法：
        1. 从累计值计算单季度值（相邻报告期相减，跨年第一季度直接取累计值）
        2. 对单季度值滚动4期求和得到TTM
        
        参数：
        -----
        records : List[Dict]
            财务记录列表，每个记录包含 report_date 和累计值
        value_field : str
            需要计算TTM的字段名（如 'net_profit_excl_min_int_inc'）
            
        返回：
        ------
        List[Dict] : 添加了单季度和TTM字段的记录
            每个记录新增：
            - {value_field}_quarter: 单季度值
            - {value_field}_ttm: TTM值
        """
        if not records:
            return []
        
        # 1. 按报告期排序（report_date 格式：YYYYMMDD）
        sorted_records = sorted(
            records,
            key=lambda x: str(x.get('report_date', ''))
        )
        
        n = len(sorted_records)
        quarter_values = []  # 单季度值
        ttm_values = []      # TTM值
        
        for i in range(n):
            record = sorted_records[i]
            current_value = record.get(value_field)
            current_report = str(record.get('report_date', ''))
            
            if current_value is None or pd.isna(current_value):
                quarter_values.append(np.nan)
                ttm_values.append(np.nan)
                continue
            
            # 解析当前报告期的年份和月份
            try:
                current_year = int(current_report[:4])
                current_month = int(current_report[4:6]) if len(current_report) >= 6 else 0
            except:
                quarter_values.append(np.nan)
                ttm_values.append(np.nan)
                continue
            
            # 2. 计算单季度值
            # 判断是否为第一季度（3月份）
            is_q1 = (current_month == 3)
            
            if i == 0:
                # 第一个记录，无法计算季度变化，直接取累计值作为单季度
                quarter_value = current_value
            else:
                prev_record = sorted_records[i - 1]
                prev_value = prev_record.get(value_field)
                prev_report = str(prev_record.get('report_date', ''))
                
                try:
                    prev_year = int(prev_report[:4])
                except:
                    prev_year = current_year
                
                # 判断是否为跨年Q1（新年第一季度，累计值重新计数）
                if is_q1 and current_year != prev_year:
                    # 新年第一季度，累计值重新计数，直接取累计值作为单季度
                    quarter_value = current_value
                else:
                    # 同一年内，单季度 = 当前累计 - 上期累计
                    if prev_value is not None and not pd.isna(prev_value):
                        quarter_value = current_value - prev_value
                    else:
                        quarter_value = current_value
            
            quarter_values.append(quarter_value)
            
            # 3. 计算TTM（最近4个单季度之和）
            if i < 3:
                # 数据不足4个季度，TTM为NaN
                ttm_values.append(np.nan)
            else:
                # 取最近4个单季度值
                recent_quarters = quarter_values[i-3:i+1]
                # 检查是否有NaN
                if any(pd.isna(v) for v in recent_quarters):
                    ttm_values.append(np.nan)
                else:
                    ttm_values.append(sum(recent_quarters))
        
        # 4. 合并结果
        result = []
        for i, record in enumerate(sorted_records):
            new_record = record.copy()
            new_record[f'{value_field}_quarter'] = quarter_values[i]
            new_record[f'{value_field}_ttm'] = ttm_values[i]
            result.append(new_record)
        
        return result
    
    def _process_single_stock(
        self,
        file_path: Path
    ) -> Optional[Dict[str, List[Tuple]]]:
        """
        处理单个股票的所有字段
        
        流程：
        1. 读取原始财务数据
        2. 对于需要TTM的字段：
           - 先计算季度TTM（累计值→单季度→4季度求和）
           - 再PIT对齐到日度
        3. 对于不需要TTM的字段：直接PIT对齐
        
        返回：
        ------
        Dict[str, List[Tuple]] : {字段名: 对齐后的数据}
        """
        stock_code = file_path.stem  # 000001.SZ
        
        # 读取财务数据
        records = self._read_financial_data(file_path)
        if not records:
            return None
        
        result = {}
        
        # 处理每个字段
        for field_name, need_ttm, source_field in self.FIELD_CONFIG:
            # 确定源字段名
            src_field = source_field if source_field else field_name
            
            if need_ttm:
                # 需要TTM：先计算季度TTM，再对齐
                # 1. 计算TTM（在季度层面）
                records_with_ttm = self._calculate_ttm_from_cumulative(records, src_field)
                
                # 2. PIT对齐TTM字段到日度
                ttm_field = f'{src_field}_ttm'
                aligned = self.aligner.align(
                    records_with_ttm,
                    'm_anntime',
                    [ttm_field],
                    stock_code
                )
                output_field = f"{field_name}_ttm"
            else:
                # 不需要TTM：直接对齐
                aligned = self.aligner.align(
                    records,
                    'm_anntime',
                    [src_field],
                    stock_code
                )
                output_field = field_name
            
            result[output_field] = aligned
        
        return result
    
    def _build_wide_table(
        self,
        all_stock_data: Dict[str, Dict[str, List[Tuple]]],
        field_name: str
    ) -> pa.Table:
        """
        将多个股票的单个字段数据拼接成宽表
        
        参数：
        -----
        all_stock_data : Dict[str, Dict]
            {股票代码: {字段名: 数据列表}}
        field_name : str
            要构建宽表的字段名
            
        返回：
        ------
        pa.Table : 宽表，列名为 [date, 股票代码1, 股票代码2, ...]
        """
        print(f"\n构建宽表: {field_name}")
        
        # 收集所有股票的数据
        stock_values = {}  # {stock_code: [value1, value2, ...]}
        
        for stock_code, fields_data in all_stock_data.items():
            if field_name not in fields_data:
                continue
            
            data = fields_data[field_name]
            # 提取数值（跳过日期）
            if field_name.endswith('_ttm'):
                # TTM 字段在最后一个位置（因为对齐时只传了一个字段）
                values = [row[-1] for row in data]
            else:
                # 普通字段在位置1
                values = [row[1] for row in data]
            
            stock_values[stock_code] = values
        
        if not stock_values:
            raise ValueError(f"没有有效数据用于字段 {field_name}")
        
        print(f"  共 {len(stock_values)} 个股票有数据")
        
        # 构建 PyArrow Table
        # 确保日期为 datetime64[ns] 类型
        arrays = [pa.array(self.trading_calendar, type=pa.timestamp('ns'))]  # 第一列是日期
        names = ['time']
        
        for stock_code, values in stock_values.items():
            # 处理 NaN
            pa_values = [None if (v != v or np.isnan(v)) else float(v) for v in values]
            arrays.append(pa.array(pa_values, type=pa.float64()))
            names.append(stock_code)
        
        return pa.table(arrays, names=names)
    
    def prepare_all_fields(self, fields: Optional[List[str]] = None, overwrite: bool = False):
        """
        批量处理所有财务字段
        
        参数：
        -----
        fields : List[str], optional
            要处理的字段列表，如 ['cap_stk', 'net_profit_ttm']，默认处理所有
        overwrite : bool
            是否覆盖已存在的文件
        """
        # 获取所有股票文件
        stock_files = self._get_stock_files()
        
        # 收集所有股票的数据
        print("\n读取并处理财务数据...")
        all_stock_data = {}
        
        for i, file_path in enumerate(stock_files):
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(stock_files)} 个股票...")
            
            result = self._process_single_stock(file_path)
            if result:
                all_stock_data[file_path.stem] = result
        
        print(f"\n成功处理 {len(all_stock_data)} 个股票")
        
        # 为每个输出字段构建宽表并保存
        output_files = []
        
        # 确定要处理的字段
        field_configs_to_process = self.FIELD_CONFIG
        if fields is not None:
            # 过滤指定字段
            field_configs_to_process = [
                (fn, need_ttm, src) for fn, need_ttm, src in self.FIELD_CONFIG
                if fn in fields or f"{fn}_ttm" in fields
            ]
        
        for field_name, need_ttm, _ in field_configs_to_process:
            output_field = f"{field_name}_ttm" if need_ttm else field_name
            output_file = self.output_path / f"{output_field}.parquet"
            
            # 检查是否已存在
            if output_file.exists() and not overwrite:
                print(f"文件已存在，跳过: {output_file}")
                output_files.append(output_file)
                continue
            
            try:
                wide_table = self._build_wide_table(all_stock_data, output_field)
                pq.write_table(wide_table, output_file)
                print(f"已保存: {output_file} ({wide_table.num_rows} 行 × {wide_table.num_columns} 列)")
                output_files.append(output_file)
            except Exception as e:
                print(f"处理字段 {output_field} 失败: {e}")
                continue
        
        print(f"\n全部完成！共生成 {len(output_files)} 个文件")
        return output_files
