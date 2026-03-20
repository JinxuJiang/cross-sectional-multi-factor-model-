#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型预测融合脚本
动态IC加权融合两个模型的平滑预测

使用方法:
    python fuse_predictions.py --exp-1 test_001_fined_20_v1 --exp-2 test_001_hor5_v1 --output-exp ensemble_5d_20d_v1

逻辑:
    1. 取两个模型的日期交集
    2. 排名标准化（统一量纲）
    3. 计算每个模型与exp-1的actual_return的滚动IC（20天窗口）
    4. IC<0设为0，归一化得到权重
    5. 加权融合排名作为最终pred_score
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_smoothed_predictions(exp_dir):
    """加载平滑预测文件"""
    smooth_file = exp_dir / 'smoothed_predictions.parquet'
    if not smooth_file.exists():
        raise FileNotFoundError(f"找不到平滑预测文件: {smooth_file}")
    return pd.read_parquet(smooth_file)


def calc_daily_ic(df, pred_col, actual_col):
    """计算每日截面IC (Spearman)"""
    def _ic(x):
        if len(x) < 10:  # 股票太少跳过
            return np.nan
        return x[pred_col].corr(x[actual_col], method='spearman')
    return df.groupby('date').apply(_ic)


def fuse_predictions(exp1_dir, exp2_dir, output_dir):
    """融合两个模型的预测"""
    print(f"加载模型1: {exp1_dir.name}")
    df1 = load_smoothed_predictions(exp1_dir)
    print(f"  样本数: {len(df1)}, 日期: {df1['date'].min().date()} ~ {df1['date'].max().date()}")
    
    print(f"加载模型2: {exp2_dir.name}")
    df2 = load_smoothed_predictions(exp2_dir)
    print(f"  样本数: {len(df2)}, 日期: {df2['date'].min().date()} ~ {df2['date'].max().date()}")
    
    # 取交集（共同日期和股票）
    print("\n取日期交集...")
    common_dates = set(df1['date'].unique()) & set(df2['date'].unique())
    print(f"  共同交易日: {len(common_dates)}")
    
    df1 = df1[df1['date'].isin(common_dates)].copy()
    df2 = df2[df2['date'].isin(common_dates)].copy()
    
    # 合并（使用exp1的actual_return作为基准）
    print("\n合并数据...")
    merged = pd.merge(
        df1[['date', 'stock_code', 'pred_score_smooth', 'actual_return', 'fold_id']],
        df2[['date', 'stock_code', 'pred_score_smooth']],
        on=['date', 'stock_code'],
        suffixes=('_1', '_2')
    )
    print(f"  交集样本数: {len(merged)}")
    
    # 每天截面排名标准化（统一量纲）
    print("\n排名标准化...")
    merged['rank_1'] = merged.groupby('date')['pred_score_smooth_1'].rank(pct=True)
    merged['rank_2'] = merged.groupby('date')['pred_score_smooth_2'].rank(pct=True)
    
    # 计算每日截面IC（与exp1的actual_return）
    print("\n计算每日截面IC (Spearman)...")
    daily_ic = pd.DataFrame({
        'date': sorted(common_dates)
    })
    
    ic_1_list = []
    ic_2_list = []
    
    for date in sorted(common_dates):
        day_data = merged[merged['date'] == date]
        if len(day_data) < 10:
            ic_1_list.append(np.nan)
            ic_2_list.append(np.nan)
            continue
        
        # IC vs exp1的actual_return
        # 手动实现spearman（避免scipy依赖）
        def spearman_corr(x, y):
            return x.rank().corr(y.rank())
        
        ic_1 = spearman_corr(day_data['rank_1'], day_data['actual_return'])
        ic_2 = spearman_corr(day_data['rank_2'], day_data['actual_return'])
        ic_1_list.append(ic_1)
        ic_2_list.append(ic_2)
    
    daily_ic['ic_1'] = ic_1_list
    daily_ic['ic_2'] = ic_2_list
    daily_ic = daily_ic.set_index('date')
    
    # 滚动IC（20天窗口，平滑噪声）
    print("计算滚动IC (20天窗口)...")
    daily_ic['ic_1_roll'] = daily_ic['ic_1'].rolling(window=20, min_periods=1).mean()
    daily_ic['ic_2_roll'] = daily_ic['ic_2'].rolling(window=20, min_periods=1).mean()
    
    # 负IC截断为0
    print("IC负值截断...")
    daily_ic['ic_1_clip'] = daily_ic['ic_1_roll'].clip(lower=0)
    daily_ic['ic_2_clip'] = daily_ic['ic_2_roll'].clip(lower=0)
    
    # 归一化权重
    print("计算归一化权重...")
    total = daily_ic['ic_1_clip'] + daily_ic['ic_2_clip']
    # 处理除0：如果都为0，默认等权重
    daily_ic['weight_1'] = daily_ic['ic_1_clip'] / total.where(total > 0, 1)
    daily_ic['weight_2'] = daily_ic['ic_2_clip'] / total.where(total > 0, 1)
    
    # 都为0时用等权重
    mask_zero = total <= 0
    daily_ic.loc[mask_zero, 'weight_1'] = 0.5
    daily_ic.loc[mask_zero, 'weight_2'] = 0.5
    
    print(f"  平均权重1: {daily_ic['weight_1'].mean():.3f}")
    print(f"  平均权重2: {daily_ic['weight_2'].mean():.3f}")
    
    # 合并权重回主表
    merged = merged.merge(
        daily_ic[['weight_1', 'weight_2']].reset_index(),
        on='date',
        how='left'
    )
    
    # 删除前20天（IC滚动窗口不足的数据）
    print(f"\n删除IC窗口不足的数据...")
    before_len = len(merged)
    merged = merged.dropna(subset=['weight_1', 'weight_2'])
    after_len = len(merged)
    print(f"  删除样本: {before_len - after_len} ({(before_len - after_len) / before_len * 100:.1f}%)")
    
    # 加权融合排名
    print("\n加权融合...")
    merged['rank_fused'] = (
        merged['weight_1'] * merged['rank_1'] + 
        merged['weight_2'] * merged['rank_2']
    )
    
    # 转回0-1范围的pred_score（保持与排名一致）
    merged['pred_score'] = merged['rank_fused']
    
    # 准备输出（为了和--use-smooth兼容，列名用pred_score_smooth）
    output = merged[['date', 'stock_code', 'pred_score', 'actual_return', 'fold_id']].copy()
    output = output.rename(columns={'pred_score': 'pred_score_smooth'})
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'smoothed_predictions.parquet'
    output.to_parquet(output_file)
    print(f"\n[OK] 融合预测已保存: {output_file}")
    print(f"  样本数: {len(output)}")
    print(f"  日期范围: {output['date'].min().date()} ~ {output['date'].max().date()}")
    print(f"  pred_score_smooth均值: {output['pred_score_smooth'].mean():.4f}, 标准差: {output['pred_score_smooth'].std():.4f}")
    
    return output, daily_ic


def fuse_live_predictions(exp1_dir, exp2_dir, output_dir):
    """融合live预测（如果存在）"""
    live1 = exp1_dir / 'smoothed_live_predictions.parquet'
    live2 = exp2_dir / 'smoothed_live_predictions.parquet'
    
    if not live1.exists() or not live2.exists():
        print(f"\n[SKIP] Live预测文件缺失，跳过live融合")
        return None
    
    print(f"\n融合Live预测...")
    df1 = pd.read_parquet(live1)
    df2 = pd.read_parquet(live2)
    
    # 取交集
    common_dates = set(df1['date'].unique()) & set(df2['date'].unique())
    print(f"  Live共同交易日: {len(common_dates)}")
    
    if len(common_dates) == 0:
        print("  [SKIP] 无共同日期，跳过")
        return None
    
    df1 = df1[df1['date'].isin(common_dates)].copy()
    df2 = df2[df2['date'].isin(common_dates)].copy()
    
    # 合并（live没有actual_return，设为NaN）
    merged = pd.merge(
        df1[['date', 'stock_code', 'pred_score_smooth', 'fold_id']],
        df2[['date', 'stock_code', 'pred_score_smooth']],
        on=['date', 'stock_code'],
        suffixes=('_1', '_2')
    )
    
    # 排名标准化
    merged['rank_1'] = merged.groupby('date')['pred_score_smooth_1'].rank(pct=True)
    merged['rank_2'] = merged.groupby('date')['pred_score_smooth_2'].rank(pct=True)
    
    # Live用等权重（没有actual_return算不了IC）
    merged['pred_score_smooth'] = 0.5 * merged['rank_1'] + 0.5 * merged['rank_2']
    merged['actual_return'] = np.nan
    
    output = merged[['date', 'stock_code', 'pred_score_smooth', 'actual_return', 'fold_id']].copy()
    output_file = output_dir / 'smoothed_live_predictions.parquet'
    output.to_parquet(output_file)
    
    print(f"[OK] Live融合预测已保存: {output_file}")
    print(f"  样本数: {len(output)}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='多模型预测融合')
    parser.add_argument('--exp-1', required=True, help='第一个模型实验ID（作为基准，使用其actual_return）')
    parser.add_argument('--exp-2', required=True, help='第二个模型实验ID')
    parser.add_argument('--output-exp', required=True, help='输出实验ID')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent / 'experiments'
    exp1_dir = base_dir / args.exp_1
    exp2_dir = base_dir / args.exp_2
    output_dir = base_dir / args.output_exp
    
    print("="*70)
    print("多模型预测融合")
    print("="*70)
    print(f"模型1 (基准): {args.exp_1}")
    print(f"模型2: {args.exp_2}")
    print(f"输出: {args.output_exp}")
    print("="*70)
    
    # 融合主预测
    output, daily_ic = fuse_predictions(exp1_dir, exp2_dir, output_dir)
    
    # 融合live（如果存在）
    fuse_live_predictions(exp1_dir, exp2_dir, output_dir)
    
    print("\n" + "="*70)
    print("完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
