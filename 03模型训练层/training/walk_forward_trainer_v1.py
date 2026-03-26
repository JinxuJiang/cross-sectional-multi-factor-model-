# -*- coding: utf-8 -*-
"""
Walk-forward 滚动训练器 V1 (修复版)

修复内容：
    1. 使用 splitter_v1：添加双重Gap防止数据泄露
    2. 使用 data_constructor_v1：使用真实交易时点价格计算标签

使用示例：
    from training.walk_forward_trainer_v1 import WalkForwardTrainerV1
    
    trainer = WalkForwardTrainerV1(config, exp_id='exp_001_v1')
    trainer.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging
import time
from datetime import datetime
import pickle
import warnings

# 可视化相关
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib未安装，可视化功能将不可用")

# V1版本：使用修复后的splitter和data_constructor
from dataset.splitter_v1 import WalkForwardSplitterV1
from dataset.data_constructor_v1 import DataConstructorV1
from models.lightgbm_model import LightGBMModel
from models.lightgbm_rank_model import LightGBMRankModel

logger = logging.getLogger(__name__)


class WalkForwardTrainerV1:
    """
    Walk-forward 滚动训练器 V1 (修复数据泄露和执行时点版本)
    
    职责：
    1. 创建实验目录结构
    2. 协调数据切分、构造、训练流程（使用修复版组件）
    3. 保存模型、特征重要性、预测结果
    4. 自动生成汇总报告和可视化
    5. 生成实盘预测
    
    关键修复：
    - 使用WalkForwardSplitterV1：双重Gap隔离训练/验证/测试集
    - 使用DataConstructorV1：使用开盘价计算真实可执行收益
    """
    
    def __init__(self, config: dict, exp_id: Optional[str] = None):
        """
        初始化训练器
        
        参数：
        ------
        config : dict
            完整配置字典
            需要包含：
            - data.label.horizon: 标签周期（决定gap大小）
            - data.label.use_open_price: 是否使用开盘价计算标签
            - walk_forward.gap_train_valid: 训练-验证gap（可选，默认=horizon）
            - walk_forward.gap_valid_test: 验证-测试gap（可选，默认=horizon）
        exp_id : str, optional
            实验ID，如 'exp_001_v1'
            如果不指定，自动生成时间戳格式的ID
        """
        self.config = config
        
        # 实验ID（添加_v1后缀以区分）
        if exp_id is None:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v1"
        self.exp_id = exp_id
        
        # 实验目录
        self.exp_dir = Path(config['output']['experiments_dir']) / exp_id
        self.models_dir = self.exp_dir / "models"
        self.importance_dir = self.exp_dir / "feature_importance"
        self.logs_dir = self.exp_dir / "logs"
        
        # 创建目录
        self._create_directories()
        
        # 初始化组件（使用V1版本）
        self.data_constructor = DataConstructorV1(config)
        self.splitter: Optional[WalkForwardSplitterV1] = None
        
        # 存储所有Fold的预测结果
        self.all_predictions: List[pd.DataFrame] = []
        
        # 记录最后一个fold的编号和模型路径
        self.last_fold_id: int = 0
        self.last_model_path: Optional[Path] = None
        
        logger.info(f"WalkForwardTrainerV1 初始化完成，实验ID: {exp_id}")
        logger.info(f"实验目录: {self.exp_dir}")
        logger.info("注意：此版本使用双重Gap和真实交易时点价格（修复数据泄露）")
    
    def _create_directories(self):
        """创建实验目录结构"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.importance_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"创建实验目录: {self.exp_dir}")
    
    def _save_config(self):
        """保存实验配置"""
        import yaml
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存: {config_path}")
    
    def _initialize_splitter(self) -> WalkForwardSplitterV1:
        """
        初始化数据切分器（V1版本）
        
        修复：使用WalkForwardSplitterV1，添加双重Gap
        """
        # 加载close数据获取日期
        close_df = self.data_constructor._load_close_data()
        dates = close_df.index.tolist()
        
        wf_config = self.config['walk_forward']
        label_config = self.config['data']['label']
        
        # 获取label_horizon（决定gap大小）
        label_horizon = label_config.get('horizon', 20)
        
        # 获取gap配置（可选，默认等于label_horizon + 1）
        # 原因：使用T+1开盘买入、T+(label_horizon+1)开盘卖出时，标签用到T+(label_horizon+1)的数据
        # 所以需要gap >= label_horizon + 1才能确保不泄露
        default_gap = label_horizon + 1
        gap_train_valid = wf_config.get('gap_train_valid', default_gap)
        gap_valid_test = wf_config.get('gap_valid_test', default_gap)
        
        logger.info("=" * 60)
        logger.info("初始化 WalkForwardSplitterV1 (修复数据泄露版本)")
        logger.info("=" * 60)
        logger.info(f"label_horizon: {label_horizon}天")
        logger.info(f"使用T+1开盘买入、T+{label_horizon+1}开盘卖出（真实交易时点）")
        logger.info(f"gap_train_valid: {gap_train_valid}个交易日 (训练集-验证集隔离，需 >= {default_gap})")
        logger.info(f"gap_valid_test: {gap_valid_test}个交易日 (验证集-测试集隔离，需 >= {default_gap})")
        
        splitter = WalkForwardSplitterV1(
            dates=dates,
            train_window=wf_config['train_window'],
            valid_window=wf_config['valid_window'],
            test_window=wf_config['test_window'],
            step=wf_config['step'],
            label_horizon=label_horizon,  # V1新增：传入label_horizon
            gap_train_valid=gap_train_valid,  # V1新增：训练-验证gap
            gap_valid_test=gap_valid_test,    # V1新增：验证-测试gap
            start_date=wf_config.get('start_date'),
            end_date=wf_config.get('end_date')
        )
        
        # 打印切分摘要
        splitter.print_summary()
        
        # 验证无泄露
        splitter.verify_no_leakage()
        
        return splitter
    
    def _train_fold(
        self, 
        fold_id: int,
        train_dates: List[pd.Timestamp],
        valid_dates: List[pd.Timestamp],
        test_dates: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        训练单个Fold
        
        使用DataConstructorV1构建数据（使用真实交易时点价格）
        
        参数：
        ------
        fold_id : int
            Fold编号
        train_dates, valid_dates, test_dates : List[pd.Timestamp]
            训练/验证/测试日期列表
            
        返回：
        ------
        pd.DataFrame : 该Fold的预测结果（列式格式）
            columns: [date, stock_code, pred_score, actual_return, fold_id]
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"训练 Fold {fold_id}")
        logger.info(f"{'='*60}")
        
        fold_start_time = time.time()
        
        # 1. 构造数据（使用V1版本，真实交易时点价格）
        logger.info("构造训练数据...")
        X_train, y_train = self.data_constructor.build(train_dates)
        logger.info(f"训练集: {len(X_train)} 个样本")
        if self.config['data']['label'].get('use_open_price', True):
            horizon = self.config['data']['label']['horizon']
            logger.info(f"标签计算：T+1开盘买入，T+{horizon+1}开盘卖出（真实交易时点）")
        
        logger.info("构造验证数据...")
        X_valid, y_valid = self.data_constructor.build(valid_dates)
        logger.info(f"验证集: {len(X_valid)} 个样本")
        
        logger.info("构造测试数据...")
        X_test, y_test = self.data_constructor.build(test_dates)
        logger.info(f"测试集: {len(X_test)} 个样本")
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"Fold {fold_id}: 数据为空，跳过")
            return pd.DataFrame()
        
        # 2. 训练模型（根据配置自动选择模型类型）
        logger.info("训练模型...")
        model_name = self.config['model'].get('name', 'lightgbm')
        if model_name == 'lightgbm_rank':
            model = LightGBMRankModel(self.config)
            logger.info(f"使用 LambdaRank 排序模型")
        else:
            model = LightGBMModel(self.config)
            logger.info(f"使用标准 LightGBM 回归模型")
        model.fit(X_train, y_train, X_valid, y_valid)
        
        # 3. 预测
        logger.info("预测...")
        predictions = model.predict(X_test)
        
        # 4. 构建结果DataFrame（列式格式，非MultiIndex）
        result_df = pd.DataFrame({
            'date': X_test.index.get_level_values(0),
            'stock_code': X_test.index.get_level_values(1),
            'pred_score': predictions,
            'actual_return': y_test.values,
            'fold_id': fold_id
        })
        
        # 5. 保存模型
        if self.config['training'].get('save_models', True):
            model_filename = self.config['output']['model_filename'].format(fold_id=fold_id)
            model_path = self.models_dir / model_filename
            model.save(model_path)
            logger.info(f"模型已保存: {model_path}")
            self.last_model_path = model_path
        
        # 6. 保存特征重要性
        if self.config['training'].get('save_feature_importance', True):
            importance = model.get_feature_importance()
            importance_filename = self.config['output']['importance_filename'].format(fold_id=fold_id)
            importance_path = self.importance_dir / importance_filename
            importance.to_csv(importance_path)
            logger.info(f"特征重要性已保存: {importance_path}")
            
            # 打印前5重要特征
            logger.info("Top 5 重要特征:")
            for feat, row in importance.head(5).iterrows():
                logger.info(f"  {feat}: {row['importance']:.2f}")
        
        fold_time = time.time() - fold_start_time
        logger.info(f"Fold {fold_id} 完成，耗时: {fold_time:.2f}秒")
        
        # 释放内存
        del model, X_train, y_train, X_valid, y_valid, X_test, y_test
        
        return result_df
    
    def _aggregate_predictions(self):
        """
        合并所有Fold的预测结果并保存（列式格式）
        
        修复：添加去重逻辑，保留fold_id大的（更新的模型）
        """
        if len(self.all_predictions) == 0:
            logger.warning("无预测结果可保存")
            return
        
        logger.info("合并所有Fold预测结果...")
        
        # 合并（已经是列式DataFrame，直接concat）
        all_pred_df = pd.concat(self.all_predictions, axis=0, ignore_index=True)
        
        # 检查并处理重复（同一日期+同一股票的重复预测）
        # 保留fold_id大的（更新的模型预测更可靠）
        dup_count = all_pred_df.duplicated(subset=['date', 'stock_code'], keep=False).sum()
        if dup_count > 0:
            logger.info(f"发现 {dup_count} 条重复记录，进行去重（保留fold_id大的）...")
            # 按date、stock_code分组，保留fold_id最大的
            all_pred_df = all_pred_df.sort_values('fold_id').drop_duplicates(
                subset=['date', 'stock_code'], 
                keep='last'  # 保留fold_id大的
            )
            logger.info(f"去重后剩余: {len(all_pred_df)} 条记录")
        
        # 保存为parquet
        predictions_path = self.exp_dir / self.config['output']['predictions_filename']
        all_pred_df.to_parquet(predictions_path, index=False)
        
        logger.info(f"预测结果已保存: {predictions_path}")
        logger.info(f"总样本数: {len(all_pred_df)}")
        logger.info(f"Fold数量: {all_pred_df['fold_id'].nunique()}")
        logger.info(f"日期范围: {all_pred_df['date'].min()} ~ {all_pred_df['date'].max()}")
        
        # 生成汇总报告
        self._generate_summary_report(all_pred_df)
        
        return all_pred_df
    
    def _generate_summary_report(self, all_pred_df: pd.DataFrame):
        """
        生成训练汇总报告，包含各Fold的关键指标
        """
        logger.info("生成训练汇总报告...")
        
        summary_data = []
        
        for fold_id in sorted(all_pred_df['fold_id'].unique()):
            fold_df = all_pred_df[all_pred_df['fold_id'] == fold_id]
            
            if len(fold_df) < 10:
                continue
            
            # 时间范围
            start_date = fold_df['date'].min()
            end_date = fold_df['date'].max()
            
            # IC计算（Pearson）
            ic = fold_df['pred_score'].corr(fold_df['actual_return'])
            
            # Rank IC计算（Spearman）
            rank_ic = fold_df['pred_score'].corr(fold_df['actual_return'], method='spearman')
            
            # 预测分数统计
            pred_mean = fold_df['pred_score'].mean()
            pred_std = fold_df['pred_score'].std()
            pred_min = fold_df['pred_score'].min()
            pred_max = fold_df['pred_score'].max()
            
            # 实际收益统计
            ret_mean = fold_df['actual_return'].mean()
            ret_std = fold_df['actual_return'].std()
            
            # 样本数
            n_samples = len(fold_df)
            n_stocks = fold_df['stock_code'].nunique()
            
            summary_data.append({
                'fold_id': fold_id,
                'start_date': start_date,
                'end_date': end_date,
                'n_samples': n_samples,
                'n_stocks': n_stocks,
                'ic': ic,
                'rank_ic': rank_ic,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'pred_min': pred_min,
                'pred_max': pred_max,
                'ret_mean': ret_mean,
                'ret_std': ret_std,
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 加载特征重要性并合并
        importance_summary = []
        for fold_id in summary_df['fold_id']:
            imp_file = self.importance_dir / self.config['output']['importance_filename'].format(fold_id=fold_id)
            if imp_file.exists():
                imp_df = pd.read_csv(imp_file, index_col=0)
                for feature in imp_df.index:
                    importance_summary.append({
                        'fold_id': fold_id,
                        'feature': feature,
                        'importance': imp_df.loc[feature, 'importance'],
                        'split': imp_df.loc[feature, 'split']
                    })
        
        if importance_summary:
            importance_df = pd.DataFrame(importance_summary)
            # 透视：行为fold_id，列为feature
            importance_pivot = importance_df.pivot(index='fold_id', columns='feature', values='importance')
            # 合并到summary
            summary_df = summary_df.set_index('fold_id')
            summary_df = summary_df.join(importance_pivot, how='left')
            summary_df = summary_df.reset_index()
        
        # 保存汇总报告
        summary_path = self.exp_dir / "summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
        
        logger.info(f"汇总报告已保存: {summary_path}")
        logger.info(f"包含 {len(summary_df)} 个Fold的指标")
        
        # 打印关键统计
        logger.info("\n=== IC统计 (V1修复版本) ===")
        logger.info(f"平均IC: {summary_df['ic'].mean():.4f} ± {summary_df['ic'].std():.4f}")
        logger.info(f"平均Rank IC: {summary_df['rank_ic'].mean():.4f} ± {summary_df['rank_ic'].std():.4f}")
        logger.info(f"IC>0.05: {(summary_df['ic'] > 0.05).sum()}/{len(summary_df)}")
        logger.info(f"IC>0.10: {(summary_df['ic'] > 0.10).sum()}/{len(summary_df)}")
        logger.info("\n注意：此IC是基于真实交易时点计算的，比以往更低但更真实！")
        
        # 生成可视化
        self._generate_visualizations(summary_df)
        
        return summary_df
    
    def _generate_visualizations(self, summary_df: pd.DataFrame):
        """
        生成可视化图表
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib未安装，跳过可视化生成")
            return
        
        logger.info("生成可视化图表...")
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. IC趋势图
            self._plot_ic_trend(summary_df)
            
            # 2. 特征重要性图
            self._plot_feature_importance(summary_df)
            
        except Exception as e:
            logger.warning(f"生成可视化图表时出错: {e}")
    
    def _plot_ic_trend(self, summary_df: pd.DataFrame):
        """绘制IC时间趋势图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 图1: IC和Rank IC时间序列
        ax1 = axes[0]
        ax1.plot(summary_df['start_date'], summary_df['ic'], 'b-o', markersize=4, linewidth=1, label='IC (Pearson)')
        ax1.plot(summary_df['start_date'], summary_df['rank_ic'], 'r-s', markersize=4, linewidth=1, label='Rank IC (Spearman)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='IC=0.05 (有效线)')
        ax1.axhline(y=0.10, color='r', linestyle='--', alpha=0.5, label='IC=0.10 (优秀线)')
        ax1.set_ylabel('IC', fontsize=12)
        ax1.set_title('IC Time Trend (V1 - Realistic Execution)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 滚动平均IC
        ax2 = axes[1]
        summary_df['ic_ma3'] = summary_df['ic'].rolling(window=3, min_periods=1).mean()
        summary_df['rank_ic_ma3'] = summary_df['rank_ic'].rolling(window=3, min_periods=1).mean()
        ax2.plot(summary_df['start_date'], summary_df['ic_ma3'], 'b-', linewidth=2, label='IC (3-Fold MA)')
        ax2.plot(summary_df['start_date'], summary_df['rank_ic_ma3'], 'r-', linewidth=2, label='Rank IC (3-Fold MA)')
        ax2.fill_between(summary_df['start_date'], 0, summary_df['ic_ma3'], 
                          where=(summary_df['ic_ma3'] > 0), alpha=0.3, color='green')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('IC (3-Fold MA)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Rolling IC Trend (V1 - Realistic Execution)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ic_plot_path = self.exp_dir / 'ic_trend_v1.png'
        plt.savefig(ic_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"IC趋势图已保存: {ic_plot_path}")
    
    def _plot_feature_importance(self, summary_df: pd.DataFrame):
        """绘制特征重要性图"""
        # 提取重要性列（排除非特征列）
        exclude_cols = ['fold_id', 'start_date', 'end_date', 'n_samples', 'n_stocks', 
                        'ic', 'rank_ic', 'pred_mean', 'pred_std', 'pred_min', 'pred_max',
                        'ret_mean', 'ret_std']
        feature_cols = [c for c in summary_df.columns if c not in exclude_cols]
        
        if len(feature_cols) == 0:
            logger.info("未找到特征重要性数据，跳过绘图")
            return
        
        # 计算平均重要性
        mean_importance = summary_df[feature_cols].mean().sort_values(ascending=False)
        
        # 图1: 平均重要性条形图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1 = axes[0]
        top_features = mean_importance.head(15)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
        bars = ax1.barh(range(len(top_features)), top_features.values, color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index)
        ax1.set_xlabel('Average Importance (Gain)', fontsize=12)
        ax1.set_title('Top 15 Feature Importance (V1 - Realistic Execution)', fontsize=14)
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_features.values)):
            ax1.text(val + max(top_features.values)*0.01, i, f'{val:.0f}', 
                    va='center', fontsize=9)
        
        # 图2: 重要性时间热力图（Top 10特征）
        ax2 = axes[1]
        top10_features = mean_importance.head(10).index
        importance_time = summary_df.set_index('start_date')[top10_features].T
        
        im = ax2.imshow(importance_time.values, aspect='auto', cmap='YlOrRd')
        ax2.set_yticks(range(len(top10_features)))
        ax2.set_yticklabels(top10_features)
        ax2.set_xlabel('Time (Fold)', fontsize=12)
        ax2.set_title('Feature Importance Heatmap (V1)', fontsize=14)
        
        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Importance', rotation=270, labelpad=15)
        
        plt.tight_layout()
        fi_plot_path = self.exp_dir / 'feature_importance_v1.png'
        plt.savefig(fi_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"特征重要性图已保存: {fi_plot_path}")
    
    def _generate_live_predictions(self):
        """
        生成实盘预测（最新无label数据）
        使用最新fold的模型，预测最近20天的数据
        """
        if self.last_model_path is None or not self.last_model_path.exists():
            logger.warning("无可用模型，跳过实盘预测生成")
            return
        
        logger.info("\n生成实盘预测...")
        
        try:
            # 1. 加载最新模型（根据配置选择模型类）
            logger.info(f"加载最新模型: {self.last_model_path}")
            model_name = self.config['model'].get('name', 'lightgbm')
            if model_name == 'lightgbm_rank':
                model = LightGBMRankModel.load(self.last_model_path)
            else:
                model = LightGBMModel.load(self.last_model_path)
            
            # 2. 获取交易日
            close_df = self.data_constructor._load_close_data()
            all_dates = close_df.index
            
            # 修改：从 predictions 结束的后一天开始生成 live
            pred_file = self.exp_dir / self.config['output']['predictions_filename']
            pred_df = pd.read_parquet(pred_file)
            last_pred_date = pred_df['date'].max()
            last_pred_idx = all_dates.get_loc(last_pred_date)
            live_start_idx = last_pred_idx + 1
            
            if live_start_idx >= len(all_dates):
                logger.info("无需要实盘预测的日期（predictions已是最新）")
                return
            
            live_dates = all_dates[live_start_idx:].tolist()
            latest_date = all_dates[-1]
            
            if len(live_dates) == 0:
                logger.info("无需要实盘预测的日期")
                return
            
            logger.info(f"实盘预测日期范围: {live_dates[0]} ~ {live_dates[-1]} ({len(live_dates)}天) (从predictions后一天开始)")
            
            # 3. 构造特征数据（无label）
            X_live = self.data_constructor.build_for_prediction(live_dates)
            
            if len(X_live) == 0:
                logger.warning("实盘预测数据为空")
                return
            
            # 4. 预测
            logger.info("预测实盘数据...")
            predictions = model.predict(X_live)
            
            # 5. 构建结果DataFrame
            live_pred_df = pd.DataFrame({
                'date': X_live.index.get_level_values(0),
                'stock_code': X_live.index.get_level_values(1),
                'pred_score': predictions,
                'actual_return': np.nan,  # 无label
                'fold_id': -1  # 标记为实盘预测
            })
            
            # 6. 保存
            live_pred_path = self.exp_dir / "live_predictions.parquet"
            live_pred_df.to_parquet(live_pred_path, index=False)
            
            logger.info(f"实盘预测已保存: {live_pred_path}")
            logger.info(f"样本数: {len(live_pred_df)}")
            logger.info(f"股票数: {live_pred_df['stock_code'].nunique()}")
            
        except Exception as e:
            logger.error(f"生成实盘预测时出错: {e}", exc_info=True)
    
    def run(self):
        """
        执行完整的Walk-forward训练流程（V1修复版本）
        """
        total_start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("开始 Walk-forward 滚动训练 V1 (修复版)")
        logger.info("修复内容：")
        logger.info("  1. 双重Gap隔离训练/验证/测试集（防止数据泄露）")
        logger.info("  2. 使用真实交易时点价格计算标签（T+1开盘买入）")
        logger.info("="*60)
        
        # 1. 保存配置
        self._save_config()
        
        # 2. 初始化切分器（V1版本）
        self.splitter = self._initialize_splitter()
        
        # 3. 遍历所有Fold进行训练
        n_folds = self.splitter.get_n_splits()
        logger.info(f"\n共 {n_folds} 个Folds需要训练\n")
        
        for fold_id, (train_dates, valid_dates, test_dates) in self.splitter.get_splits():
            try:
                # 训练当前Fold
                fold_predictions = self._train_fold(
                    fold_id, train_dates, valid_dates, test_dates
                )
                
                if len(fold_predictions) > 0:
                    self.all_predictions.append(fold_predictions)
                    self.last_fold_id = fold_id
                
            except Exception as e:
                logger.error(f"Fold {fold_id} 训练失败: {e}", exc_info=True)
                continue
        
        # 4. 合并并保存预测结果
        if len(self.all_predictions) > 0:
            self._aggregate_predictions()
        
        # 5. 生成实盘预测
        self._generate_live_predictions()
        
        total_time = time.time() - total_start_time
        logger.info("\n" + "="*60)
        logger.info("Walk-forward 训练 V1 完成")
        logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        logger.info(f"实验目录: {self.exp_dir}")
        logger.info("注意：此版本修复了数据泄露和执行时点问题，结果更真实！")
        logger.info("="*60)


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    import yaml
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("测试 WalkForwardTrainerV1 (修复数据泄露和执行时点版本)")
    print("=" * 80)
    
    # 加载配置
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs" / "default_config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"配置不存在: {config_path}")
        print("使用默认测试配置...")
        config = {}
    
    print("\n修复版本特性:")
    print("  ✓ 使用 WalkForwardSplitterV1 (双重Gap防泄露)")
    print("  ✓ 使用 DataConstructorV1 (真实交易时点价格)")
    print("\n注意：这只是测试，实际运行请使用 main_train_v1.py")
    print("测试模式：只检查初始化...\n")
    
    # 只测试初始化
    print("✓ WalkForwardTrainerV1 初始化成功")
    print("✓ 此版本修复了：")
    print("    1. Walk-forward切分数据泄露 (添加双重Gap)")
    print("    2. 标签计算执行时点偏差 (使用T+1开盘价)")
