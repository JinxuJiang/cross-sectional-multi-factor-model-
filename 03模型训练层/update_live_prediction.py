#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新实盘预测脚本
使用最新Fold的模型，预测新数据（从该Fold测试期结束到最新日期）
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob

# 路径设置
BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / 'experiments' / 'test_001_fined_v1'
FACTOR_DIR = BASE_DIR.parent / '02因子库' / 'processed_data' / 'factors'
MARKET_DATA_DIR = BASE_DIR.parent / '02因子库' / 'processed_data' / 'market_data'
MODEL_PATH = EXPERIMENT_DIR / 'models' / 'model_fold_047.pkl'
OUTPUT_PATH = EXPERIMENT_DIR / 'live_predictions.parquet'


def discover_factors():
    """自动发现所有因子文件"""
    factor_files = {}
    
    # 技术因子
    tech_path = FACTOR_DIR / 'technical'
    if tech_path.exists():
        for f in sorted(tech_path.glob('*.parquet')):
            factor_files[f.stem] = f
    
    # 财务因子
    fin_path = FACTOR_DIR / 'financial'
    if fin_path.exists():
        for f in sorted(fin_path.glob('*.parquet')):
            factor_files[f.stem] = f
    
    print(f"发现 {len(factor_files)} 个因子: {list(factor_files.keys())}")
    return factor_files


def load_factor(factor_name, factor_path):
    """加载单个因子数据"""
    df = pd.read_parquet(factor_path)
    if 'time' in df.columns:
        df = df.set_index('time')
    df.index = pd.to_datetime(df.index)
    return df


def load_close_data():
    """加载收盘价"""
    close_path = MARKET_DATA_DIR / 'close.parquet'
    df = pd.read_parquet(close_path)
    if 'time' in df.columns:
        df = df.set_index('time')
    df.index = pd.to_datetime(df.index)
    return df


def construct_features(factor_files, start_date, end_date):
    """
    构造特征矩阵
    自动扫描所有因子，对齐日期和股票
    """
    print(f"\n构造特征矩阵: {start_date} ~ {end_date}")
    
    # 加载close确定股票池和日期
    close_df = load_close_data()
    date_mask = (close_df.index >= start_date) & (close_df.index <= end_date)
    close_df = close_df[date_mask]
    
    print(f"日期范围: {close_df.index.min().date()} ~ {close_df.index.max().date()}")
    print(f"交易日数量: {len(close_df)}")
    
    # 加载所有因子并对齐
    factor_data = {}
    for factor_name, factor_path in factor_files.items():
        print(f"  加载因子: {factor_name}")
        factor_df = load_factor(factor_name, factor_path)
        # 对齐到close的日期和股票
        factor_df = factor_df.reindex(index=close_df.index, columns=close_df.columns)
        factor_data[factor_name] = factor_df
    
    # 构建截面数据
    all_data = []
    
    for date in close_df.index:
        # 获取当天有价格的股票
        stocks_today = close_df.loc[date].dropna().index.tolist()
        
        if len(stocks_today) == 0:
            continue
        
        # 构造当天所有股票的特征
        day_features = {}
        for factor_name in factor_files.keys():
            factor_values = factor_data[factor_name].loc[date, stocks_today]
            day_features[factor_name] = factor_values.values
        
        # 创建DataFrame
        day_df = pd.DataFrame(day_features, index=stocks_today)
        day_df.index.name = 'stock_code'
        day_df['date'] = date
        
        # 删除有缺失值的样本
        day_df = day_df.dropna()
        
        if len(day_df) > 0:
            all_data.append(day_df)
    
    if not all_data:
        raise ValueError("没有有效的特征数据")
    
    # 合并所有日期
    X = pd.concat(all_data, ignore_index=False)
    X = X.reset_index().set_index(['date', 'stock_code'])
    
    # 如果特征数不足24个，添加虚拟列（保持与训练时一致）
    n_features = len(X.columns)
    if n_features < 24:
        for i in range(24 - n_features):
            X[f'dummy_{i}'] = 0.0
        print(f"  添加 {24 - n_features} 个虚拟列保持兼容")
    
    print(f"\n特征矩阵构造完成:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(X.columns)}")
    print(f"  日期数: {X.index.get_level_values(0).nunique()}")
    
    return X


def predict_and_save(model_wrapper, X):
    """预测并保存结果"""
    print("\n开始预测...")
    
    # 使用底层LightGBM模型预测
    lgb_model = model_wrapper.model
    pred_scores = lgb_model.predict(X.values, num_iteration=lgb_model.best_iteration)
    
    # 构造输出DataFrame
    result = pd.DataFrame({
        'date': X.index.get_level_values(0),
        'stock_code': X.index.get_level_values(1),
        'pred_score': pred_scores,
        'actual_return': np.nan,
        'fold_id': 47
    })
    
    # 保存
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n预测结果已保存: {OUTPUT_PATH}")
    print(f"  总样本数: {len(result)}")
    print(f"  日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"  股票数: {result['stock_code'].nunique()}")
    
    # 显示最新一天的统计
    latest_date = result['date'].max()
    latest = result[result['date'] == latest_date]
    print(f"\n最新日期 ({latest_date.date()}) 统计:")
    print(f"  预测股票数: {len(latest)}")
    print(f"  预测分数均值: {latest['pred_score'].mean():.4f}")
    print(f"  预测分数标准差: {latest['pred_score'].std():.4f}")
    print(f"  前5名股票:")
    top5 = latest.nlargest(5, 'pred_score')[['stock_code', 'pred_score']]
    for _, row in top5.iterrows():
        print(f"    {row['stock_code']}: {row['pred_score']:.4f}")


def main():
    print("=" * 60)
    print("更新实盘预测")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 发现所有因子
    factor_files = discover_factors()
    
    # 2. 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("模型加载成功")
    
    # 3. 确定日期范围
    # Fold 47测试期结束: 2026-01-19，新预测从 2026-01-20开始
    start_date = '2026-01-20'
    
    # 获取最新日期
    close_df = load_close_data()
    end_date = close_df.index.max().strftime('%Y-%m-%d')
    
    print(f"\n预测日期范围: {start_date} ~ {end_date}")
    
    # 4. 构造特征
    X = construct_features(factor_files, start_date, end_date)
    
    # 5. 预测并保存
    predict_and_save(model, X)
    
    # 生成平滑版本
    try:
        print("\n生成平滑版本...")
        df = pd.read_parquet(OUTPUT_PATH)
        
        # 计算rank autocorrelation
        df['rank'] = df.groupby('date')['pred_score'].rank()
        rank_wide = df.pivot(index='date', columns='stock_code', values='rank')
        autocorrs = []
        for i in range(1, len(rank_wide)):
            yest, today = rank_wide.iloc[i-1], rank_wide.iloc[i]
            mask = yest.notna() & today.notna()
            if mask.sum() >= 10:
                corr = yest[mask].corr(today[mask])
                if not np.isnan(corr):
                    autocorrs.append(corr)
        autocorr = np.mean(autocorrs) if autocorrs else 0.94
        halflife = np.log(0.5) / np.log(autocorr)
        window = int(halflife)
        
        # 指数平滑
        weights = (0.5 ** (np.arange(window) / halflife))[::-1]
        weights = weights / weights.sum()
        
        def ewma(x):
            n = len(x)
            result = np.empty(n)
            for i in range(n):
                start = max(0, i - window + 1)
                w = weights[-(i - start + 1):]
                w = w / w.sum()
                result[i] = np.average(x[start:i+1], weights=w)
            return result
        
        df = df.sort_values(['stock_code', 'date'])
        df['pred_score_smooth'] = df.groupby('stock_code')['pred_score'].transform(ewma)
        
        smooth_path = EXPERIMENT_DIR / 'smoothed_live_predictions.parquet'
        df[['date', 'stock_code', 'pred_score', 'pred_score_smooth', 'actual_return', 'fold_id']].to_parquet(smooth_path)
        print(f"[OK] 平滑版本: {smooth_path}")
    except Exception as e:
        print(f"[SKIP] 平滑失败: {e}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
