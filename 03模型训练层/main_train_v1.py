# -*- coding: utf-8 -*-
"""
主训练脚本 V1 (修复版)
===================

修复内容：
    1. 使用 WalkForwardSplitterV1：双重Gap防止数据泄露
    2. 使用 DataConstructorV1：使用真实交易时点价格计算标签

使用方法：
    cd <项目根目录>
    conda activate AKTool
    python 03模型训练层/main_train_v1.py
    python 03模型训练层/main_train_v1.py --config configs/fined_lgbm_config.yaml --exp-id test_001_fined --start-date 2020-01-01 -y  

命令行参数：
    --exp-id, -e        指定实验ID（如：test_001）
    --config, -c        指定配置文件路径(注意！--config configs/fined_lgbm_config.yaml)
    --start-date        训练开始日期（YYYY-MM-DD）
    --end-date          训练结束日期（YYYY-MM-DD）
    --gap               Gap大小（天，默认等于horizon）
    --horizon           Label horizon（天）
    --use-open-price    使用开盘价计算标签
    --no-open-price     使用收盘价计算标签（兼容旧版）
    -y, --yes           跳过确认提示

使用示例：
    # 基础用法
    python 03模型训练层/main_train_v1.py
    
    # 指定实验ID
    python 03模型训练层/main_train_v1.py --exp-id test_001
    
    # 指定日期范围
    python 03模型训练层/main_train_v1.py --start-date 2020-01-01 --end-date 2023-12-31
    
    # 指定gap和horizon
    python 03模型训练层/main_train_v1.py --gap 21 --horizon 20
    
    # 跳过确认直接运行
    python 03模型训练层/main_train_v1.py --exp-id my_test -y

配置文件：
    03模型训练层/configs/default_config.yaml
    
    需要确保配置中包含：
    - data.label.horizon: 20
    - data.label.use_open_price: true
    - data.open_column: 'open'
    - walk_forward.gap_train_valid: 21 (可选，默认等于label_horizon + 1，使用T+1开盘买入时需要)
    - walk_forward.gap_valid_test: 21 (可选，默认等于label_horizon + 1)

输出目录：
    03模型训练层/experiments/{exp_id}_v1/
"""

import sys
import os
from pathlib import Path
import yaml
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.walk_forward_trainer_v1 import WalkForwardTrainerV1


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='截面多因子模型 - Walk-forward训练 V1 (修复版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 基础用法（自动生成实验ID）
    python main_train_v1.py
    
    # 指定实验ID
    python main_train_v1.py --exp-id test_001
    
    # 指定配置文件
    python main_train_v1.py --config configs/my_config.yaml
    
    # 指定训练日期范围
    python main_train_v1.py --start-date 2020-01-01 --end-date 2023-12-31
    
    # 指定gap大小
    python main_train_v1.py --gap 20 --horizon 20
    
    # 跳过确认提示
    python main_train_v1.py -y
        '''
    )
    
    parser.add_argument(
        '--exp-id', '-e',
        type=str,
        default=None,
        help='实验ID（如：test_001）。如果不指定，自动生成时间戳格式'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径（默认：configs/default_config.yaml）'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='训练开始日期（格式：YYYY-MM-DD，覆盖配置文件）'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='训练结束日期（格式：YYYY-MM-DD，覆盖配置文件）'
    )
    
    parser.add_argument(
        '--gap',
        type=int,
        default=None,
        help='Gap大小（天）。如果不指定，默认等于horizon'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='Label horizon（天）。覆盖配置文件中的设置'
    )
    
    parser.add_argument(
        '--use-open-price',
        action='store_true',
        default=None,
        help='使用开盘价计算标签（真实交易时点）'
    )
    
    parser.add_argument(
        '--no-open-price',
        action='store_true',
        help='不使用开盘价，使用收盘价（兼容旧版）'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='跳过确认提示，直接开始训练'
    )
    
    return parser.parse_args()


def setup_logging():
    """设置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: Path) -> dict:
    """
    加载配置文件
    
    如果配置文件不存在，尝试在 configs/ 目录下查找同名文件
    如果还找不到，使用默认配置
    """
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功: {config_path}")
        return config
    else:
        # 尝试在 configs/ 目录下查找同名文件
        script_dir = Path(__file__).parent
        configs_dir = script_dir / "configs"
        fallback_path = configs_dir / config_path.name
        
        if fallback_path.exists():
            with open(fallback_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ 配置文件加载成功: {fallback_path}")
            return config
        else:
            print(f"⚠ 配置文件不存在: {config_path}")
            print("使用默认配置...")
            return get_default_config()


def get_default_config() -> dict:
    """
    默认配置（当配置文件不存在时使用）
    """
    config = {
        'data': {
            'factor_paths': {
                'technical': '02因子库/processed_data/technical_factors',
                'financial': '02因子库/processed_data/financial_factors'
            },
            'market_data_path': '02因子库/processed_data/market_data',
            'price_column': 'close',
            'open_column': 'open',  # V1新增：开盘价字段
            'label': {
                'horizon': 20,
                'use_open_price': True  # V1新增：使用开盘价计算标签
            }
        },
        'walk_forward': {
            'train_window': '3Y',
            'valid_window': '6M',
            'test_window': '3M',
            'step': '3M',
            'gap_train_valid': 21,  # V1新增：训练-验证gap（默认label_horizon + 1，使用T+1开盘买入、T+(horizon+1)开盘卖出）
            'gap_valid_test': 21,   # V1新增：验证-测试gap（默认label_horizon + 1）
            # 'start_date': '2015-01-01',  # 可选：训练开始日期
            # 'end_date': '2024-12-31',    # 可选：训练结束日期
        },
        'model': {
            'name': 'lightgbm',
            'params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 1000,
                'early_stopping_rounds': 50
            }
        },
        'training': {
            'save_models': True,
            'save_feature_importance': True
        },
        'output': {
            'experiments_dir': '03模型训练层/experiments',
            'model_filename': 'model_fold_{fold_id}.pkl',
            'importance_filename': 'importance_fold_{fold_id}.csv',
            'predictions_filename': 'predictions.parquet'
        }
    }
    return config


def check_data_files(config: dict) -> bool:
    """
    检查必要的文件是否存在
    
    返回：
        bool: 检查是否通过
    """
    print("\n" + "="*60)
    print("检查数据文件...")
    print("="*60)
    
    all_ok = True
    
    # 1. 检查收盘价数据
    market_data_path = Path(config['data']['market_data_path'])
    close_file = market_data_path / f"{config['data']['price_column']}.parquet"
    if close_file.exists():
        print(f"✓ 收盘价数据: {close_file}")
    else:
        print(f"✗ 收盘价数据不存在: {close_file}")
        all_ok = False
    
    # 2. 检查开盘价数据（V1版本需要）
    if config['data']['label'].get('use_open_price', True):
        # 获取open_column配置，默认为'open'
        open_column = config['data'].get('open_column', 'open')
        open_file = market_data_path / f"{open_column}.parquet"
        if open_file.exists():
            print(f"✓ 开盘价数据: {open_file}")
        else:
            print(f"⚠ 开盘价数据不存在: {open_file}")
            print("  将使用收盘价替代（标签计算可能不够精确）")
    
    # 3. 检查因子数据目录
    tech_path = Path(config['data']['factor_paths']['technical'])
    fin_path = Path(config['data']['factor_paths']['financial'])
    
    if tech_path.exists():
        tech_files = list(tech_path.glob('*.parquet'))
        print(f"✓ 技术因子目录: {tech_path} ({len(tech_files)}个文件)")
    else:
        print(f"✗ 技术因子目录不存在: {tech_path}")
        all_ok = False
    
    if fin_path.exists():
        fin_files = list(fin_path.glob('*.parquet'))
        print(f"✓ 财务因子目录: {fin_path} ({len(fin_files)}个文件)")
    else:
        print(f"✗ 财务因子目录不存在: {fin_path}")
        all_ok = False
    
    print("="*60)
    
    return all_ok


def print_config_summary(config: dict):
    """
    打印配置摘要
    """
    print("\n" + "="*60)
    print("配置摘要")
    print("="*60)
    
    # 数据配置
    print("\n【数据配置】")
    print(f"  技术因子: {config['data']['factor_paths']['technical']}")
    print(f"  财务因子: {config['data']['factor_paths']['financial']}")
    print(f"  行情数据: {config['data']['market_data_path']}")
    print(f"  收盘价列: {config['data']['price_column']}")
    print(f"  开盘价列: {config['data'].get('open_column', 'open')}")
    
    # 标签配置（V1重点）
    print("\n【标签配置】V1修复版")
    label_config = config['data']['label']
    print(f"  预测周期: {label_config['horizon']}天")
    print(f"  使用开盘价: {label_config.get('use_open_price', True)}")
    if label_config.get('use_open_price', True):
        horizon = label_config['horizon']
        print(f"  标签计算: T+1开盘买入 → T+{horizon+1}开盘卖出（真实交易时点）")
    else:
        print("  标签计算: T日收盘买入 → T+20日收盘卖出（传统方法）")
    
    # Walk-forward配置（V1重点）
    print("\n【Walk-forward配置】V1修复版")
    wf_config = config['walk_forward']
    print(f"  训练窗口: {wf_config['train_window']}")
    print(f"  验证窗口: {wf_config['valid_window']}")
    print(f"  测试窗口: {wf_config['test_window']}")
    print(f"  滚动步长: {wf_config['step']}")
    
    # Gap配置（V1新增）
    label_horizon = label_config['horizon']
    gap_train_valid = wf_config.get('gap_train_valid', label_horizon)
    gap_valid_test = wf_config.get('gap_valid_test', label_horizon)
    print(f"\n  【双重Gap - 防止数据泄露】")
    print(f"  训练-验证gap: {gap_train_valid}天")
    print(f"  验证-测试gap: {gap_valid_test}天")
    print(f"  说明: gap确保标签计算不越界，消除数据泄露")
    
    # 模型配置
    print("\n【模型配置】")
    print(f"  模型类型: {config['model']['name']}")
    print(f"  学习率: {config['model']['params']['learning_rate']}")
    print(f"  树数量: {config['model']['params']['n_estimators']}")
    print(f"  早停轮数: {config['model']['params']['early_stopping_rounds']}")
    
    # 输出配置
    print("\n【输出配置】")
    print(f"  实验目录: {config['output']['experiments_dir']}")
    
    print("="*60)


def update_config_from_args(config: dict, args) -> dict:
    """
    根据命令行参数更新配置
    """
    # 更新日期范围
    if args.start_date:
        config['walk_forward']['start_date'] = args.start_date
        print(f"  命令行覆盖: start_date = {args.start_date}")
    
    if args.end_date:
        config['walk_forward']['end_date'] = args.end_date
        print(f"  命令行覆盖: end_date = {args.end_date}")
    
    # 更新horizon
    if args.horizon is not None:
        config['data']['label']['horizon'] = args.horizon
        print(f"  命令行覆盖: horizon = {args.horizon}")
    
    # 更新gap（交易日数量）
    if args.gap is not None:
        horizon = config['data']['label'].get('horizon', 20)
        config['walk_forward']['gap_train_valid'] = args.gap
        config['walk_forward']['gap_valid_test'] = args.gap
        if args.gap < horizon + 1:
            print(f"  [警告] gap={args.gap} < horizon+1={horizon+1}，可能导致数据泄露！")
        print(f"  命令行覆盖: gap_train_valid = {args.gap}个交易日, gap_valid_test = {args.gap}个交易日")
    
    # 更新是否使用开盘价
    if args.use_open_price:
        config['data']['label']['use_open_price'] = True
        print(f"  命令行覆盖: use_open_price = True")
    elif args.no_open_price:
        config['data']['label']['use_open_price'] = False
        print(f"  命令行覆盖: use_open_price = False")
    
    return config


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging()
    
    print("\n" + "="*80)
    print("截面多因子模型 - Walk-forward训练 V1 (修复版)")
    print("="*80)
    print("\n修复内容：")
    print("  1. 使用双重Gap隔离训练/验证/测试集（消除数据泄露）")
    print("  2. 使用真实交易时点价格计算标签（T+1开盘买入）")
    print("\n预期效果：")
    print("  - 回测收益会下降（但更真实）")
    print("  - 验证集IC会更低（但无泄露）")
    print("  - 实盘可信度大幅提升")
    print("="*80)
    
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 配置文件路径
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = script_dir / config_path
    else:
        config_path = script_dir / "configs" / "default_config.yaml"
    
    # 加载配置
    config = load_config(config_path)
    
    # 补充默认配置（如果配置文件缺少某些字段）
    if 'open_column' not in config['data']:
        config['data']['open_column'] = 'open'
        print("  补充默认配置: open_column = 'open'")
    
    # 根据命令行参数更新配置
    print("\n" + "-"*60)
    print("应用命令行参数覆盖...")
    config = update_config_from_args(config, args)
    print("-"*60)
    
    # 检查数据文件
    if not check_data_files(config):
        print("\n✗ 数据文件检查失败，请确保：")
        print("  1. 行情数据已下载（close.parquet）")
        print("  2. 因子数据已计算（technical_factors/和financial_factors/）")
        print("  3. 配置文件路径正确")
        sys.exit(1)
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 确认继续
    print("\n" + "="*60)
    if args.yes:
        print("跳过确认（-y模式）")
        response = 'y'
    else:
        response = input("确认开始训练？(y/n): ").strip().lower()
    
    if response != 'y':
        print("训练已取消")
        sys.exit(0)
    print("="*60)
    
    # 生成或使用指定的实验ID
    if args.exp_id:
        exp_id = args.exp_id
        # 自动添加_v1后缀（如果没有）
        if not exp_id.endswith('_v1'):
            exp_id = f"{exp_id}_v1"
    else:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v1"
    
    try:
        # 创建训练器（V1版本）
        print(f"\n创建训练器，实验ID: {exp_id}")
        trainer = WalkForwardTrainerV1(config, exp_id=exp_id)
        
        # 运行训练
        print("\n开始训练...")
        trainer.run()
        
        print("\n" + "="*80)
        print("✓ 训练完成！")
        print(f"✓ 实验目录: {trainer.exp_dir}")
        print("\n输出文件：")
        print(f"  - 预测结果: {trainer.exp_dir / config['output']['predictions_filename']}")
        print(f"  - 汇总报告: {trainer.exp_dir / 'summary.parquet'}")
        print(f"  - IC趋势图: {trainer.exp_dir / 'ic_trend_v1.png'}")
        print(f"  - 特征重要性: {trainer.exp_dir / 'feature_importance_v1.png'}")
        print("="*80)
        
        # 生成平滑预测
        try:
            def smooth_df(df, halflife, window):
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
                return df
            
            def calc_halflife(df):
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
                return np.log(0.5) / np.log(autocorr)
            
            # 1. 平滑 predictions
            pred_file = trainer.exp_dir / config['output']['predictions_filename']
            df_pred = pd.read_parquet(pred_file)
            halflife = calc_halflife(df_pred)
            window = int(halflife)
            df_pred = smooth_df(df_pred, halflife, window)
            smooth_file = trainer.exp_dir / 'smoothed_predictions.parquet'
            # 删除已存在的旧平滑文件（确保生成最新版本）
            if smooth_file.exists():
                smooth_file.unlink()
                print(f"\n[注意] 删除旧的平滑文件: {smooth_file.name}")
            df_pred[['date', 'stock_code', 'pred_score', 'pred_score_smooth', 'actual_return', 'fold_id']].to_parquet(smooth_file)
            
            # 2. 平滑 live_predictions
            live_file = trainer.exp_dir / 'live_predictions.parquet'
            if live_file.exists():
                df_live = pd.read_parquet(live_file)
                df_live = smooth_df(df_live, halflife, window)
                smooth_live_file = trainer.exp_dir / 'smoothed_live_predictions.parquet'
                # 删除已存在的旧平滑文件（确保生成最新版本）
                if smooth_live_file.exists():
                    smooth_live_file.unlink()
                    print(f"[注意] 删除旧的平滑文件: {smooth_live_file.name}")
                df_live[['date', 'stock_code', 'pred_score', 'pred_score_smooth', 'actual_return', 'fold_id']].to_parquet(smooth_live_file)
                print(f"\n[OK] 平滑预测: predictions & live (halflife={halflife:.1f}天)")
            else:
                print(f"\n[OK] 平滑预测: predictions (halflife={halflife:.1f}天)")
        except Exception as e:
            print(f"\n[SKIP] 平滑预测失败: {e}")
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
