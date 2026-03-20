"""
生成实时交易信号工具 (与 backtrader.eval_1.1.py 逻辑一致)
==========================================================

使用说明:
---------
本工具使用与回测完全一致的选股逻辑:
1. 只保留主板股票 (60/00开头)
2. 过滤T+1开盘涨停的股票 (涨幅>=9.9%)
3. 按预测分数选Top N
4. 等权分配90%仓位

使用示例:
---------
python generate_live_signals.py --exp-id exp_001 --date 2026-03-10

输出:
-----
- 终端显示 Top N 股票列表及目标仓位
- 生成 live_signals_YYYYMMDD.csv 文件，可直接用于下单参考
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import gc


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent


def load_predictions_and_market_data(exp_id: str):
    """
    加载预测数据和市场数据（与 backtrader.eval_1.1.py 一致）
    """
    PROJECT_ROOT = get_project_root()
    
    # 路径配置
    paths = {
        'open': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'open.parquet',
        'close': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'close.parquet',
        'high': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'high.parquet',
        'low': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'low.parquet',
        'volume': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'volume.parquet',
        'pred': PROJECT_ROOT / '03模型训练层' / 'experiments' / exp_id / 'predictions.parquet',
        'live_pred': PROJECT_ROOT / '03模型训练层' / 'experiments' / exp_id / 'live_predictions.parquet',
    }
    
    print("--- 步骤1: 加载预测数据 ---")
    prediction = pd.read_parquet(paths['pred'], columns=['date', 'stock_code', 'pred_score'])
    live_prediction = pd.read_parquet(paths['live_pred'], columns=['date', 'stock_code', 'pred_score'])
    
    pred_total = pd.concat([prediction, live_prediction], axis=0)
    pred_total = pred_total.rename(columns={'date': 'time', 'pred_score': 'prediction'})
    pred_total['time'] = pd.to_datetime(pred_total['time'])
    pred_total['prediction'] = pred_total['prediction'].astype('float32')
    
    del prediction, live_prediction
    gc.collect()
    
    # 宽表转长表的辅助函数
    def wide_to_long(df_wide, value_name, time_col='time'):
        if time_col not in df_wide.columns:
            if df_wide.index.name == time_col:
                df_wide = df_wide.reset_index()
            else:
                df_wide.index.name = time_col
                df_wide = df_wide.reset_index()
        
        df_wide = df_wide.set_index(time_col)
        df_long = df_wide.stack().reset_index()
        df_long.columns = [time_col, 'stock_code', value_name]
        
        # 只保留标准格式的股票代码
        df_long = df_long[df_long['stock_code'].str.match(r'^\d{6}\.(SZ|SH|BJ)$', na=False)]
        df_long[value_name] = df_long[value_name].astype('float32')
        df_long = df_long.dropna(subset=[value_name])
        return df_long
    
    # 合并市场数据
    main_df = pred_total
    for col in ['open', 'close', 'high', 'low', 'volume']:
        print(f"正在处理 {col} 数据...")
        temp_wide = pd.read_parquet(paths[col])
        temp_long = wide_to_long(temp_wide, col)
        
        main_df = pd.merge(main_df, temp_long, on=['time', 'stock_code'], how='left')
        del temp_wide, temp_long
        gc.collect()
    
    main_df['datetime'] = pd.to_datetime(main_df['time'])
    main_df = main_df.set_index('datetime')
    
    # 删除有缺失值的行
    main_df = main_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    print(f"完成数据合并，形状: {main_df.shape}")
    print(f"数据时间范围: {main_df.index.min()} ~ {main_df.index.max()}")
    
    return main_df


def filter_main_board_and_limit_up(df: pd.DataFrame, target_date: datetime, df_full: pd.DataFrame):
    """
    筛选主板股票并过滤开盘涨停股（与 backtrader.eval_1.1.py 逻辑一致）
    
    Args:
        df: 目标日期的预测数据
        target_date: 目标日期
        df_full: 完整数据（用于获取T+1数据）
    
    Returns:
        filtered_df: 过滤后的数据
    """
    # 1. 只保留主板股票（60/00开头）
    df = df[df['stock_code'].str.match(r'^(60|00)\d{4}\.(SH|SZ)$', na=False)]
    
    if df.empty:
        return df
    
    # 2. 获取T+1日数据，过滤开盘涨停的
    future_dates = df_full[df_full.index > target_date].index.unique()
    if len(future_dates) == 0:
        # 没有T+1数据，不过滤涨停
        return df
    
    next_date = future_dates[0]
    
    try:
        # 获取T+1的数据
        next_day_df = df_full.loc[next_date]
        if isinstance(next_day_df, pd.Series):
            next_day_df = next_day_df.to_frame().T
        
        # 合并T日和T+1日的数据（按stock_code）
        merged = df[['stock_code', 'close']].merge(
            next_day_df[['stock_code', 'open']], 
            on='stock_code', 
            suffixes=('_t', '_t1')
        )
        
        if len(merged) > 0:
            # 计算开盘涨幅
            merged['open_return'] = merged['open_t1'] / merged['close_t'] - 1
            
            # 过滤开盘涨停（>=9.9%）的
            tradable_stocks = merged.loc[merged['open_return'] < 0.099, 'stock_code']
            tradable = df[df['stock_code'].isin(tradable_stocks)]
            
            filtered_count = len(merged) - len(tradable_stocks)
            if filtered_count > 0:
                print(f"  过滤 {filtered_count} 只开盘涨停股")
        else:
            tradable = df
    except Exception as e:
        # 如果出错，就不过滤
        print(f"  过滤涨停时出错: {e}")
        tradable = df
    
    return tradable


def generate_live_signals(main_df: pd.DataFrame, target_date: datetime, top_n: int, total_cash: float):
    """
    生成实时交易信号（与回测逻辑一致）
    
    Args:
        main_df: 合并后的完整数据
        target_date: 目标日期
        top_n: 选股数量
        total_cash: 总资金
    
    Returns:
        signals_df: 交易信号DataFrame
    """
    # 获取目标日期的数据
    try:
        target_df = main_df.loc[target_date]
    except KeyError:
        # 尝试找最近的交易日
        available_dates = sorted(main_df.index.unique())
        closest_date = None
        for d in available_dates:
            if d <= target_date:
                closest_date = d
        if closest_date is None:
            raise ValueError(f"没有 {target_date.strftime('%Y-%m-%d')} 及之前的数据")
        print(f"  警告: {target_date.strftime('%Y-%m-%d')} 无数据，使用最近的交易日 {closest_date.strftime('%Y-%m-%d')}")
        target_date = closest_date
        target_df = main_df.loc[target_date]
    
    if isinstance(target_df, pd.Series):
        target_df = target_df.to_frame().T
    
    print(f"\n--- 步骤2: 筛选主板股票并过滤涨停 ---")
    print(f"目标日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"原始股票数: {len(target_df)}")
    
    # 应用主板+涨停过滤
    tradable = filter_main_board_and_limit_up(target_df, target_date, main_df)
    
    print(f"主板可交易股票数: {len(tradable)}")
    
    if tradable.empty:
        raise ValueError("没有符合条件的可交易股票")
    
    # 按预测分数排序，选出 top_n
    signals = tradable.sort_values('prediction', ascending=False).head(top_n).copy()
    
    # 计算仓位（与回测一致：90%资金等权分配）
    position_pct = 0.90
    invest_cash = total_cash * position_pct
    per_stock_cash = invest_cash / len(signals)
    per_stock_weight = position_pct / len(signals)
    
    signals['目标金额'] = per_stock_cash
    signals['建议权重'] = per_stock_weight
    signals['排序'] = range(1, len(signals) + 1)
    
    # 选择输出列
    output_cols = ['排序', 'stock_code', 'prediction', '目标金额', '建议权重', 'close']
    signals = signals[output_cols].rename(columns={'prediction': 'pred_score', 'close': 'T日收盘价'})
    
    return target_date, signals


def save_signals(signals_df: pd.DataFrame, signal_date: datetime, exp_id: str, report_dir: Path):
    """保存交易信号到CSV"""
    date_str = signal_date.strftime('%Y%m%d')
    filename = f"live_signals_{date_str}.csv"
    filepath = report_dir / filename
    
    signals_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath


def print_signals_table(signals_df: pd.DataFrame, signal_date: datetime, total_cash: float):
    """打印交易信号表格"""
    print("\n" + "="*80)
    print(f"实时交易信号生成报告 (主板股票 + 过滤涨停)")
    print(f"预测日期: {signal_date.strftime('%Y-%m-%d')}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\n资金配置:")
    print(f"  总资金: {total_cash:,.0f} 元")
    print(f"  计划仓位: 90%")
    print(f"  投资金额: {total_cash * 0.90:,.0f} 元")
    print(f"  选股数量: {len(signals_df)} 只")
    print(f"  每只股票目标金额: {total_cash * 0.90 / len(signals_df):,.0f} 元")
    
    print(f"\n推荐买入列表 (Top {len(signals_df)}):")
    print("-"*80)
    print(f"{'排名':<6}{'股票代码':<12}{'预测分数':<12}{'T日收盘价':<12}{'目标金额':<15}{'建议权重':<10}")
    print("-"*80)
    
    for _, row in signals_df.iterrows():
        print(f"{int(row['排序']):<6}{row['stock_code']:<12}{row['pred_score']:<12.6f}{row['T日收盘价']:<12.2f}{row['目标金额']:<15,.0f}{row['建议权重']:<10.2%}")
    
    print("-"*80)
    print(f"\n选股逻辑说明:")
    print(f"  1. 只保留主板股票 (60/00开头)")
    print(f"  2. 过滤T+1开盘涨停股 (涨幅>=9.9%)")
    print(f"  3. 按预测分数排序选Top {len(signals_df)}")
    print(f"  4. 等权分配90%仓位")
    print(f"\n提示:")
    print(f"  - 此信号基于 {signal_date.strftime('%Y-%m-%d')} 的模型预测")
    print(f"  - 建议在下个交易日开盘后按目标金额买入")
    print(f"  - 下次正常调仓日: 下个月第一个交易日")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='生成实时交易信号（与回测逻辑一致）')
    parser.add_argument('--exp-id', type=str, required=True, help='实验ID，如 exp_001')
    parser.add_argument('--top-n', type=int, default=30, help='选股数量，默认30只（与回测一致）')
    parser.add_argument('--total-cash', type=float, default=100000, help='总资金，默认100000')
    parser.add_argument('--date', type=str, default=None, help='指定预测日期(如 2026-03-10)，默认使用最新日期')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录，默认 04回测层/reports/{exp_id}')
    
    args = parser.parse_args()
    
    print(f"\n>>> 正在生成实时交易信号...")
    print(f"实验ID: {args.exp_id}")
    print(f"选股数量: {args.top_n}")
    print(f"总资金: {args.total_cash:,.0f} 元")
    if args.date:
        print(f"指定日期: {args.date}")
    else:
        print(f"使用日期: 最新可用日期")
    
    # 1. 加载数据（与回测一致）
    print("\n" + "="*60)
    main_df = load_predictions_and_market_data(args.exp_id)
    print("="*60)
    
    # 2. 确定目标日期
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = main_df.index.max()
        print(f"\n使用最新日期: {target_date.strftime('%Y-%m-%d')}")
    
    # 3. 生成交易信号
    try:
        signal_date, signals_df = generate_live_signals(main_df, target_date, args.top_n, args.total_cash)
    except ValueError as e:
        print(f"\n错误: {e}")
        return
    
    # 4. 保存并打印
    print("\n--- 步骤3: 保存交易信号 ---")
    if args.output_dir:
        report_dir = Path(args.output_dir)
    else:
        report_dir = get_project_root() / "04回测层" / "reports" / args.exp_id
    
    report_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_signals(signals_df, signal_date, args.exp_id, report_dir)
    print(f"信号文件已保存: {filepath}")
    
    # 打印交易表格
    print_signals_table(signals_df, signal_date, args.total_cash)
    
    print(f"\n>>> 完成！")
    print(f"请将 '{filepath}' 中的股票列表导入交易软件执行买入。")


if __name__ == "__main__":
    main()
