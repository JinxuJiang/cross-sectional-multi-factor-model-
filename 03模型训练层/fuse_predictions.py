#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型预测融合脚本 (V2 - 支持任意数量模型)
动态IC加权融合多个模型的平滑预测

使用方法:
    python fuse_predictions.py --exps test_001_hor5_v1 test_001_fined_20_v1 --base-idx 1 --output-exp ensemble_5d_20d_v1
    python fuse_predictions.py --exps test_001_hor5_v1 test_001_fined_20_v1 test_001_hor60_v1 --base-idx 1 --output-exp ensemble_5d_20d_60d_v1

参数说明:
    --exps:      输入模型实验ID列表（按顺序排列，索引从0开始）
    --base-idx:  基准模型索引（从0开始），决定IC权重计算的滞后天数(lag)
                 lag = 基准模型的horizon，用于避免使用未来信息计算权重
                 示例: --exps hor5 hor20 hor60 --base-idx 1 表示以hor20为基准，lag=20
    --output-exp: 融合后的输出实验ID

逻辑:
    1. 读取各模型的 smoothed_predictions.parquet 和 smoothed_live_predictions.parquet
    2. 确定分界日期：最长horizon模型的test结束日期
    3. 每天取股票交集（所有模型都有的股票）
    4. 每天截面排名标准化（pct_rank）
    5. 计算滞后IC权重（lag=基准模型horizon）
    6. 融合test（<=分界日期）：每天加权平均排名
    7. 融合live（>分界日期）：固定使用test最后一天权重
    
输出:
    - smoothed_predictions.parquet: 融合test预测
    - smoothed_live_predictions.parquet: 融合live预测  
    - fusion_config.yaml: 融合配置记录

--base-idx 选择建议:
    - 0 (horizon=5):   权重变化快，适合震荡市场
    - 1 (horizon=20):  权重变化适中，推荐选择 ✅
    - 2 (horizon=60):  权重变化慢，适合趋势市场
"""

import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模型预测融合')
    parser.add_argument('--exps', nargs='+', required=True,
                        help='多个模型实验ID（如：test_001_hor5_v1 test_001_fined_20_v1）')
    parser.add_argument('--base-idx', type=int, default=0,
                        help=('基准模型索引（从0开始），决定IC权重计算的滞后天数(lag)。'
                              'lag = 基准模型的horizon，用于避免使用未来信息。'
                              '例如：--exps hor5 hor20 hor60 --base-idx 1 表示lag=20。'
                              '建议选择horizon适中的模型（如20d）作为基准。'))
    parser.add_argument('--output-exp', required=True,
                        help='输出实验ID（如：ensemble_5d_20d_v1）')
    return parser.parse_args()


def load_model_config(exp_dir: Path) -> Dict:
    """加载模型配置，获取horizon信息"""
    config_path = exp_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    horizon = config.get('data', {}).get('label', {}).get('horizon', 20)
    return {'horizon': horizon}


def load_model_data(exp_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载模型数据
    返回: (test_df, live_df)
    """
    # 读取test数据
    test_file = exp_dir / 'smoothed_predictions.parquet'
    if not test_file.exists():
        raise FileNotFoundError(f"Test预测不存在: {test_file}")
    
    test_df = pd.read_parquet(test_file)
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 读取live数据（如果存在）
    live_file = exp_dir / 'smoothed_live_predictions.parquet'
    if live_file.exists():
        live_df = pd.read_parquet(live_file)
        live_df['date'] = pd.to_datetime(live_df['date'])
    else:
        live_df = pd.DataFrame(columns=test_df.columns)
    
    return test_df, live_df


def determine_split_date(models_data: List[Dict]) -> pd.Timestamp:
    """
    确定分界日期：最长horizon模型的test结束日期
    """
    # 找到horizon最长的模型
    longest_model = max(models_data, key=lambda x: x['horizon'])
    
    # 该模型的test结束日期
    split_date = longest_model['test_df']['date'].max()
    
    print(f"\n[分界日期确定]")
    print(f"  最长horizon模型: {longest_model['exp_id']} (horizon={longest_model['horizon']})")
    print(f"  分界日期: {split_date.date()}")
    
    return split_date


def merge_with_intersection(dfs: List[pd.DataFrame], value_col: str = 'pred_score_smooth') -> pd.DataFrame:
    """
    合并多个DataFrame，按日期取股票交集
    每天只有所有模型都有的股票才保留
    """
    n_models = len(dfs)
    
    # 准备合并用的列名
    prepared_dfs = []
    for i, df in enumerate(dfs):
        df = df.copy()
        if value_col not in df.columns:
            raise ValueError(f"DF {i} 缺少列 {value_col}, 现有列: {df.columns.tolist()}")
        df[f'pred_{i}'] = df[value_col]
        prepared_dfs.append(df[['date', 'stock_code', f'pred_{i}', 'actual_return']])
    
    # 从第一个模型开始
    merged = prepared_dfs[0].copy()
    
    # 逐个合并其他模型，取交集
    for i in range(1, n_models):
        merged = pd.merge(
            merged, 
            prepared_dfs[i][['date', 'stock_code', f'pred_{i}']], 
            on=['date', 'stock_code'],
            how='inner'  # inner join取交集
        )
    
    return merged


def calc_daily_ic(df: pd.DataFrame, n_models: int, base_actual_col: str = 'actual_return') -> pd.DataFrame:
    """
    计算每日截面IC（用基准模型的actual_return作为标签）
    """
    dates = sorted(df['date'].unique())
    ic_records = []
    
    for date in dates:
        day_data = df[df['date'] == date]
        
        if len(day_data) < 10:  # 股票太少跳过
            ic_record = {'date': date}
            for i in range(n_models):
                ic_record[f'ic_{i}'] = np.nan
            ic_records.append(ic_record)
            continue
        
        # 获取基准actual_return
        actual = day_data[base_actual_col].values
        
        # 计算各模型的Spearman IC
        ic_record = {'date': date}
        for i in range(n_models):
            pred = day_data[f'pred_{i}'].values
            # Spearman IC: rank correlation
            pred_rank = pd.Series(pred).rank().values
            actual_rank = pd.Series(actual).rank().values
            
            # 计算相关系数
            if len(pred_rank) > 1:
                ic = np.corrcoef(pred_rank, actual_rank)[0, 1]
            else:
                ic = np.nan
            
            ic_record[f'ic_{i}'] = ic
        
        ic_records.append(ic_record)
    
    return pd.DataFrame(ic_records)


def calc_lagged_weights(daily_ic: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    计算滞后IC权重
    第t天的权重 = 第0天到第t-lag天的IC累积均值（不包含最近lag天）
    前lag天（t < lag）：等权重
    """
    n_models = len([c for c in daily_ic.columns if c.startswith('ic_')])
    n_days = len(daily_ic)
    
    weights_records = []
    
    for t in range(n_days):
        date = daily_ic.iloc[t]['date']
        
        if t < lag:
            # 前lag天使用等权重
            w = [1.0 / n_models] * n_models
        else:
            # 历史IC累积均值 [0, t-lag]
            hist_ic = daily_ic.iloc[:t-lag+1][[f'ic_{i}' for i in range(n_models)]].mean()
            
            # 负IC截断为0
            hist_ic = hist_ic.clip(lower=0)
            
            # 归一化权重
            if hist_ic.sum() > 0:
                w = (hist_ic / hist_ic.sum()).values
            else:
                w = [1.0 / n_models] * n_models
        
        record = {'date': date}
        for i in range(n_models):
            record[f'weight_{i}'] = w[i]
        weights_records.append(record)
    
    return pd.DataFrame(weights_records)


def rank_standardize(df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    """
    每天截面排名标准化（pct_rank）
    """
    df = df.copy()
    
    for i in range(n_models):
        df[f'rank_{i}'] = df.groupby('date')[f'pred_{i}'].rank(pct=True)
    
    return df


def fuse_with_weights(df: pd.DataFrame, weights_df: pd.DataFrame, n_models: int) -> pd.DataFrame:
    """
    使用权重融合排名
    """
    df = df.copy()
    
    # 合并权重
    df = df.merge(weights_df, on='date', how='left')
    
    # 计算加权融合排名
    df['rank_fused'] = 0
    for i in range(n_models):
        df['rank_fused'] += df[f'weight_{i}'] * df[f'rank_{i}']
    
    # 构造输出（与原fuse_predictions.py格式一致）
    output = df[['date', 'stock_code', 'rank_fused', 'actual_return']].copy()
    output = output.rename(columns={'rank_fused': 'pred_score_smooth'})
    
    # 添加fold_id列（test期间标记为0）
    output['fold_id'] = 0
    
    return output


def fuse_with_fixed_weights(df: pd.DataFrame, fixed_weights: pd.Series, n_models: int) -> pd.DataFrame:
    """
    使用固定权重融合排名（用于live期间）
    """
    df = df.copy()
    
    # 计算加权融合排名
    df['rank_fused'] = 0
    for i in range(n_models):
        df['rank_fused'] += fixed_weights.iloc[i] * df[f'rank_{i}']
    
    # 构造输出
    output = df[['date', 'stock_code', 'rank_fused']].copy()
    output['actual_return'] = np.nan  # live无标签
    output = output.rename(columns={'rank_fused': 'pred_score_smooth'})
    output['fold_id'] = -1  # live标记为-1
    
    return output


def save_fusion_config(output_dir: Path, models_data: List[Dict], split_date: pd.Timestamp,
                       lag: int, last_weights: pd.Series, base_idx: int, base_model: str):
    """保存融合配置"""
    config = {
        'fusion_info': {
            'n_models': len(models_data),
            'base_model': base_model,
            'base_model_index': base_idx,
            'split_date': split_date.strftime('%Y-%m-%d'),
            'ic_lag': lag,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'models': [],
        'weights': {}
    }
    
    # 记录各模型信息
    for i, model in enumerate(models_data):
        config['models'].append({
            'index': i,
            'exp_id': model['exp_id'],
            'horizon': model['horizon'],
            'final_weight': float(last_weights.iloc[i])
        })
        config['weights'][f'model_{i}'] = float(last_weights.iloc[i])
    
    # 保存YAML
    config_path = output_dir / 'fusion_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n[配置已保存] {config_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("多模型预测融合")
    print("="*70)
    print(f"输入模型: {args.exps}")
    print(f"基准模型索引: {args.base_idx}")
    print(f"输出实验ID: {args.output_exp}")
    print("="*70)
    
    # 路径设置
    base_dir = Path(__file__).parent / 'experiments'
    output_dir = base_dir / args.output_exp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载各模型配置和数据
    print("\n[1/8] 加载模型数据...")
    models_data = []
    
    for exp_id in args.exps:
        exp_path = base_dir / exp_id
        print(f"  加载: {exp_id}")
        
        # 加载配置
        config = load_model_config(exp_path)
        
        # 加载数据
        test_df, live_df = load_model_data(exp_path)
        
        print(f"    horizon: {config['horizon']}")
        print(f"    test: {test_df['date'].min().date()} ~ {test_df['date'].max().date()} ({len(test_df)}行)")
        if len(live_df) > 0:
            print(f"    live: {live_df['date'].min().date()} ~ {live_df['date'].max().date()} ({len(live_df)}行)")
        
        models_data.append({
            'exp_id': exp_id,
            'horizon': config['horizon'],
            'test_df': test_df,
            'live_df': live_df
        })
    
    n_models = len(models_data)
    
    # 2. 确定分界日期
    print("\n[2/8] 确定分界日期...")
    split_date = determine_split_date(models_data)
    
    # 3. 准备数据：test和live分别处理
    print("\n[3/8] 合并数据（取股票交集）...")
    
    # 准备test数据（<=分界日期）
    test_dfs = []
    for model in models_data:
        df = model['test_df'][model['test_df']['date'] <= split_date].copy()
        test_dfs.append(df)
    
    test_merged = merge_with_intersection(test_dfs)
    print(f"  test交集后: {len(test_merged)}行, {test_merged['date'].nunique()}交易日")
    
    # 准备live数据（>分界日期，所有模型的test剩余部分+live）
    live_dfs = []
    for model in models_data:
        # test剩余部分
        test_remain = model['test_df'][model['test_df']['date'] > split_date].copy()
        # live部分
        live_part = model['live_df'].copy()
        # 合并
        combined = pd.concat([test_remain, live_part], ignore_index=True)
        if len(combined) > 0:
            live_dfs.append(combined)
    
    if len(live_dfs) > 0:
        live_merged = merge_with_intersection(live_dfs)
        print(f"  live交集后: {len(live_merged)}行, {live_merged['date'].nunique()}交易日")
    else:
        live_merged = pd.DataFrame()
        print("  live数据为空")
    
    # 4. 排名标准化
    print("\n[4/8] 排名标准化...")
    test_merged = rank_standardize(test_merged, n_models)
    if len(live_merged) > 0:
        live_merged = rank_standardize(live_merged, n_models)
    
    # 5. 计算每日IC
    print("\n[5/8] 计算每日IC...")
    daily_ic = calc_daily_ic(test_merged, n_models)
    print(f"  共{daily_ic['date'].nunique()}个交易日")
    
    # 打印平均IC
    for i in range(n_models):
        avg_ic = daily_ic[f'ic_{i}'].mean()
        print(f"  Model {i} ({models_data[i]['exp_id']}): 平均IC={avg_ic:.4f}")
    
    # 6. 计算滞后权重
    print("\n[6/8] 计算滞后权重...")
    lag = models_data[args.base_idx]['horizon']
    base_model = models_data[args.base_idx]['exp_id']
    print(f"  基准模型: {base_model} (索引={args.base_idx})")
    print(f"  IC滞后lag={lag}天（等于基准模型的horizon）")
    print(f"  说明: 前{lag}天使用等权重，第{lag+1}天起使用历史IC均值")
    test_weights = calc_lagged_weights(daily_ic, lag)
    
    # 7. 融合test
    print("\n[7/8] 融合test预测...")
    test_fused = fuse_with_weights(test_merged, test_weights, n_models)
    
    # 保存
    test_file = output_dir / 'smoothed_predictions.parquet'
    test_fused.to_parquet(test_file, index=False)
    print(f"  已保存: {test_file} ({len(test_fused)}行)")
    
    # 8. 融合live
    print("\n[8/8] 融合live预测...")
    if len(live_merged) > 0:
        # 获取test最后一天权重
        last_date = test_weights['date'].max()
        last_weights = test_weights[test_weights['date'] == last_date][[f'weight_{i}' for i in range(n_models)]].iloc[0]
        
        print(f"  使用{last_date.date()}的固定权重:")
        for i in range(n_models):
            print(f"    Model {i}: {last_weights.iloc[i]:.4f}")
        
        live_fused = fuse_with_fixed_weights(live_merged, last_weights, n_models)
        
        live_file = output_dir / 'smoothed_live_predictions.parquet'
        live_fused.to_parquet(live_file, index=False)
        print(f"  已保存: {live_file} ({len(live_fused)}行)")
    else:
        last_weights = pd.Series([1.0/n_models]*n_models, index=[f'weight_{i}' for i in range(n_models)])
        print("  无live数据，跳过")
    
    # 9. 保存融合配置
    save_fusion_config(output_dir, models_data, split_date, lag, last_weights, 
                       args.base_idx, models_data[args.base_idx]['exp_id'])
    
    print("\n" + "="*70)
    print("融合完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
