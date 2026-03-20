# -*- coding: utf-8 -*-
"""
市值数据准备脚本

计算并保存市值数据：市值 = 收盘价 × 总股本

用法：
    python prepare_market_cap.py
    
可选参数：
    --overwrite : 覆盖已存在的文件

输出：
    processed_data/financial_data/market_cap.parquet
    格式：宽表，time × stock_code，值为市值
"""

import argparse
import sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


def prepare_market_cap(overwrite: bool = False) -> Path:
    """
    计算市值并保存
    
    参数：
    -----
    overwrite : bool
        是否覆盖已存在的文件
        
    返回：
    ------
    Path : 输出文件路径
    """
    # 路径设置
    current_file = Path(__file__).resolve()
    factor_lib_root = current_file.parent.parent.parent  # 因子库
    
    processed_data_path = factor_lib_root / 'processed_data'
    close_file = processed_data_path / 'market_data' / 'close.parquet'
    cap_stk_file = processed_data_path / 'financial_data' / 'cap_stk.parquet'
    output_file = processed_data_path / 'financial_data' / 'market_cap.parquet'
    
    print("=" * 60)
    print("市值数据准备工具")
    print("=" * 60)
    
    # 检查是否已存在
    if output_file.exists() and not overwrite:
        print(f"\n文件已存在，跳过: {output_file}")
        print("使用 --overwrite 参数强制覆盖")
        return output_file
    
    # 检查输入文件
    print(f"\n读取数据...")
    if not close_file.exists():
        raise FileNotFoundError(f"收盘价数据不存在: {close_file}\n请先运行 main_prepare_market_data.py")
    if not cap_stk_file.exists():
        raise FileNotFoundError(f"总股本数据不存在: {cap_stk_file}\n请先运行 main_prepare_financial_data.py")
    
    # 读取数据
    print(f"  读取: {close_file}")
    close_df = pq.read_table(close_file).to_pandas()
    print(f"  读取: {cap_stk_file}")
    cap_stk_df = pq.read_table(cap_stk_file).to_pandas()
    
    print(f"  收盘价形状: {close_df.shape}")
    print(f"  总股本形状: {cap_stk_df.shape}")
    
    # 设置 time 为索引
    if 'time' in close_df.columns:
        close_df = close_df.set_index('time')
    if 'time' in cap_stk_df.columns:
        cap_stk_df = cap_stk_df.set_index('time')
    
    # 对齐并计算市值
    print(f"\n计算市值...")
    common_cols = close_df.columns.intersection(cap_stk_df.columns)
    common_index = close_df.index.intersection(cap_stk_df.index)
    
    print(f"  共同股票: {len(common_cols)}")
    print(f"  共同日期: {len(common_index)}")
    
    market_cap_df = (close_df.loc[common_index, common_cols] * 
                     cap_stk_df.loc[common_index, common_cols])
    
    print(f"  市值形状: {market_cap_df.shape}")
    print(f"  非空值比例: {market_cap_df.notna().sum().sum() / (market_cap_df.shape[0] * market_cap_df.shape[1]) * 100:.2f}%")
    
    # 保存为 parquet
    print(f"\n保存结果...")
    
    # 重置索引，time 作为列
    market_cap_reset = market_cap_df.reset_index()
    
    # 构建 PyArrow Table
    arrays = [pa.array(market_cap_reset['time'], type=pa.timestamp('ns'))]
    names = ['time']
    
    for col in market_cap_reset.columns[1:]:
        arrays.append(pa.array(market_cap_reset[col], type=pa.float64()))
        names.append(col)
    
    table = pa.table(arrays, names=names)
    pq.write_table(table, output_file)
    
    print(f"  已保存: {output_file}")
    print(f"  行数: {table.num_rows}, 列数: {table.num_columns}")
    
    print("\n" + "=" * 60)
    print("市值数据准备完成！")
    print("=" * 60)
    
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='准备市值数据：市值 = 收盘价 × 总股本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 计算并保存市值
  python prepare_market_cap.py
  
  # 强制覆盖已存在文件
  python prepare_market_cap.py --overwrite
        """
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的文件'
    )
    
    args = parser.parse_args()
    
    try:
        prepare_market_cap(overwrite=args.overwrite)
        return 0
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
