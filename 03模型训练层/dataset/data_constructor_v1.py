# -*- coding: utf-8 -*-
"""
数据构造模块 V1 (修复版)

修复内容：
    1. 标签计算使用真实的交易时点价格
       - 买入时点：T+1日开盘价（而非T日收盘价）
       - 卖出时点：T+21日开盘价（而非T+20日收盘价）
       - 更真实反映实盘可执行收益

使用示例：
    from dataset.data_constructor_v1 import DataConstructorV1
    
    constructor = DataConstructorV1(config)
    
    # 构造单个Fold的数据
    X_train, y_train = constructor.build(train_dates)
    X_valid, y_valid = constructor.build(valid_dates)
    X_test, y_test = constructor.build(test_dates)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


class DataConstructorV1:
    """
    数据构造器 V1 (修复执行时点问题)
    
    负责：
    - 惰性加载因子数据（按需加载，节省内存）
    - 计算真实的实盘收益标签（使用开盘价）
    - 截面数据对齐（处理缺失值）
    - 输出X, y矩阵供模型训练
    
    修复：标签计算使用open[t+1]和open[t+21]，反映真实交易时点
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据构造器
        
        参数：
        ------
        config : dict
            配置字典，包含：
            - data.factor_paths: 因子数据路径
            - data.market_data_path: 行情数据路径
            - data.price_column: 价格字段（默认'close'）
            - data.open_column: 开盘价格字段（默认'open'）
            - data.label.horizon: 预测 horizon（默认20）
            - data.label.use_open_price: 是否使用开盘价计算标签（默认True）
        """
        self.config = config
        
        # 路径配置
        self.factor_paths = config['data']['factor_paths']
        self.market_data_path = Path(config['data']['market_data_path'])
        self.price_column = config['data'].get('price_column', 'close')
        self.open_column = config['data'].get('open_column', 'open')
        self.label_horizon = config['data']['label']['horizon']
        
        # 是否使用开盘价计算标签（V1新增）
        self.use_open_price = config['data']['label'].get('use_open_price', True)
        
        if self.use_open_price:
            logger.info("DataConstructorV1: 使用开盘价计算标签（真实交易时点）")
        else:
            logger.info("DataConstructorV1: 使用收盘价计算标签（兼容旧版本）")
        
        # 缓存（惰性加载）
        self._factor_files: Optional[Dict[str, List[Path]]] = None
        self._close_df: Optional[pd.DataFrame] = None
        self._open_df: Optional[pd.DataFrame] = None  # V1新增
        self._all_dates: Optional[pd.DatetimeIndex] = None
        self._all_stocks: Optional[List[str]] = None
        
        # 记录特征名
        self.feature_names: Optional[List[str]] = None
        
        logger.info("DataConstructorV1 初始化完成")
    
    def _discover_factor_files(self) -> Dict[str, List[Path]]:
        """
        发现并缓存所有因子文件路径
        
        返回：
        ------
        Dict[str, List[Path]] : {factor_name: file_path}
        """
        if self._factor_files is not None:
            return self._factor_files
        
        factor_files = {}
        
        # 扫描技术因子
        tech_path = Path(self.factor_paths['technical'])
        if tech_path.exists():
            for f in sorted(tech_path.glob('*.parquet')):
                factor_name = f.stem  # 文件名作为因子名
                factor_files[factor_name] = f
                logger.debug(f"发现技术因子: {factor_name}")
        
        # 扫描财务因子
        fin_path = Path(self.factor_paths['financial'])
        if fin_path.exists():
            for f in sorted(fin_path.glob('*.parquet')):
                factor_name = f.stem
                factor_files[factor_name] = f
                logger.debug(f"发现财务因子: {factor_name}")
        
        self._factor_files = factor_files
        self.feature_names = list(factor_files.keys())
        
        logger.info(f"共发现 {len(factor_files)} 个因子: {list(factor_files.keys())}")
        
        return factor_files
    
    def _load_close_data(self) -> pd.DataFrame:
        """
        加载收盘价数据
        
        返回：
        ------
        pd.DataFrame : index=date, columns=stock_codes
        """
        if self._close_df is not None:
            return self._close_df
        
        close_file = self.market_data_path / f"{self.price_column}.parquet"
        
        if not close_file.exists():
            raise FileNotFoundError(f"收盘价数据不存在: {close_file}")
        
        logger.info(f"加载收盘价数据: {close_file}")
        
        # 读取Parquet（time是普通列，需要set_index）
        df = pq.read_table(close_file).to_pandas()
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        
        # 转置为宽表格式：index=date, columns=stocks
        # 原始格式已经是宽表：time + stocks
        self._close_df = df
        
        logger.info(f"收盘价数据: {df.shape[0]} 天 x {df.shape[1]} 只股票")
        
        return self._close_df
    
    def _load_open_data(self) -> pd.DataFrame:
        """
        加载开盘价数据（V1新增）
        
        返回：
        ------
        pd.DataFrame : index=date, columns=stock_codes
        """
        if self._open_df is not None:
            return self._open_df
        
        open_file = self.market_data_path / f"{self.open_column}.parquet"
        
        if not open_file.exists():
            logger.warning(f"开盘价数据不存在: {open_file}，将使用收盘价替代")
            # 如果没有开盘价，使用收盘价替代
            self._open_df = self._load_close_data()
            return self._open_df
        
        logger.info(f"加载开盘价数据: {open_file}")
        
        # 读取Parquet
        df = pq.read_table(open_file).to_pandas()
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        
        self._open_df = df
        
        logger.info(f"开盘价数据: {df.shape[0]} 天 x {df.shape[1]} 只股票")
        
        return self._open_df
    
    def _load_factor_data(self, factor_name: str, dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        加载指定因子在指定日期的数据
        
        参数：
        ------
        factor_name : str
            因子名称
        dates : List[pd.Timestamp]
            需要加载的日期列表
            
        返回：
        ------
        pd.DataFrame : index=date, columns=stocks, values=factor_values
        """
        factor_files = self._discover_factor_files()
        
        if factor_name not in factor_files:
            raise ValueError(f"未知因子: {factor_name}")
        
        file_path = factor_files[factor_name]
        
        # 读取因子数据
        df = pq.read_table(file_path).to_pandas()
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index)
        
        # 筛选指定日期
        df = df.loc[df.index.isin(dates)]
        
        return df
    
    def _compute_labels(self, dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        计算未来N日收益率标签（V1修复版）
        
        修复：使用真实的交易时点价格
        - 如果use_open_price=True: 使用open[t+1]买入，open[t+21]卖出
        - 如果use_open_price=False: 使用close[t]买入，close[t+20]卖出（兼容旧版）
        
        参数：
        ------
        dates : List[pd.Timestamp]
            预测日期（T时刻，即因子计算日）
            
        返回：
        ------
        pd.DataFrame : index=date, columns=stocks, values=future_return
        """
        close_df = self._load_close_data()
        
        if self.use_open_price:
            open_df = self._load_open_data()
            logger.info("使用开盘价计算标签（真实交易时点）")
        else:
            open_df = close_df
            logger.info("使用收盘价计算标签（兼容模式）")
        
        # 获取所有交易日
        all_dates = close_df.index
        
        # 构建标签DataFrame
        labels = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        
        for date in dates:
            # 找到当前日期在所有交易日中的位置
            try:
                idx = all_dates.get_loc(date)
            except KeyError:
                # 该日期不在交易日中（可能是周末/假期）
                continue
            
            if self.use_open_price:
                # V1修复版：使用真实交易时点
                # T日收盘后计算因子，T+1日开盘买入
                entry_idx = idx + 1  # T+1日
                if entry_idx >= len(all_dates):
                    continue
                entry_date = all_dates[entry_idx]
                
                # 持有N天后，T+(N+1)日开盘卖出
                exit_idx = entry_idx + self.label_horizon  # T+21日
                if exit_idx >= len(all_dates):
                    continue
                exit_date = all_dates[exit_idx]
                
                # 使用开盘价计算收益
                entry_price = open_df.loc[entry_date]  # T+1开盘价（买入价）
                exit_price = open_df.loc[exit_date]    # T+(N+1)开盘价（卖出价）
                
            else:
                # 旧版逻辑：使用收盘价（有执行时点偏差）
                future_idx = idx + self.label_horizon
                if future_idx >= len(all_dates):
                    continue
                future_date = all_dates[future_idx]
                
                entry_price = close_df.loc[date]       # T日收盘价
                exit_price = close_df.loc[future_date] # T+20日收盘价
            
            # 计算收益率
            valid_mask = entry_price > 0
            ret = pd.Series(index=close_df.columns, dtype=float)
            ret[valid_mask] = exit_price[valid_mask] / entry_price[valid_mask] - 1
            
            labels.loc[date] = ret
        
        return labels
    
    def _construct_cross_section(
        self, 
        date: pd.Timestamp, 
        factor_data: Dict[str, pd.DataFrame],
        label_series: Optional[pd.Series] = None
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], List[str]]]:
        """
        构造单个交易日的截面数据
        
        参数：
        ------
        date : pd.Timestamp
            交易日
        factor_data : Dict[str, pd.DataFrame]
            各因子的完整数据
        label_series : pd.Series, optional
            该日期的标签值。如果为None，则不返回label（用于实盘预测）
            
        返回：
        ------
        Tuple[X, y, stock_list] 或 Tuple[X, None, stock_list]（如果该日期无有效数据）
        """
        # 1. 获取当天已上市股票（close不为NaN的）
        close_df = self._load_close_data()
        if date not in close_df.index:
            return None
        
        listed_stocks = close_df.loc[date].dropna().index.tolist()
        
        if len(listed_stocks) == 0:
            return None
        
        # 2. 收集各因子的值
        factor_values = {}
        for factor_name, factor_df in factor_data.items():
            if date in factor_df.index:
                factor_values[factor_name] = factor_df.loc[date, listed_stocks]
            else:
                # 该日期无此因子数据
                factor_values[factor_name] = pd.Series(index=listed_stocks, dtype=float)
        
        # 构建特征矩阵
        X_df = pd.DataFrame(factor_values, index=listed_stocks)
        
        # 3. 剔除因子值缺失的样本（任一因子为NaN则删除该股票）
        valid_factor_mask = X_df.notna().all(axis=1)
        X_df = X_df[valid_factor_mask]
        
        if len(X_df) == 0:
            return None
        
        # 4. 如果有label，获取并剔除缺失
        if label_series is not None:
            y_series = label_series.reindex(X_df.index)
            
            # 5. 剔除标签缺失的样本
            valid_label_mask = y_series.notna()
            X_df = X_df[valid_label_mask]
            y_series = y_series[valid_label_mask]
            
            if len(X_df) == 0:
                return None
            
            return X_df.values, y_series.values, X_df.index.tolist()
        else:
            # 实盘预测模式，不返回label
            return X_df.values, None, X_df.index.tolist()
    
    def build(self, dates: List[pd.Timestamp]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        构造指定日期范围的数据集（带label，用于训练/验证/测试）
        
        参数：
        ------
        dates : List[pd.Timestamp]
            需要构造数据的日期列表
            
        返回：
        ------
        Tuple[pd.DataFrame, pd.Series]
            - X: 特征矩阵，MultiIndex=(date, stock_code)，columns=feature_names
            - y: 标签序列，index与X相同
        """
        logger.info(f"构造数据集: {len(dates)} 个交易日")
        
        # 1. 加载所有因子数据
        factor_files = self._discover_factor_files()
        factor_data = {}
        
        for factor_name in self.feature_names:
            logger.debug(f"加载因子: {factor_name}")
            factor_data[factor_name] = self._load_factor_data(factor_name, dates)
        
        # 2. 计算标签
        logger.info("计算标签...")
        labels_df = self._compute_labels(dates)
        
        # 3. 逐日构造截面数据
        X_list = []
        y_list = []
        index_tuples = []
        
        for date in dates:
            result = self._construct_cross_section(date, factor_data, labels_df.loc[date])
            
            if result is None:
                continue
            
            X_day, y_day, stocks_day = result
            
            X_list.append(X_day)
            y_list.append(y_day)
            
            # 构建MultiIndex
            for stock in stocks_day:
                index_tuples.append((date, stock))
        
        if len(X_list) == 0:
            logger.warning("无有效数据")
            # 返回空DataFrame
            empty_index = pd.MultiIndex.from_tuples([], names=['date', 'stock_code'])
            X = pd.DataFrame(columns=self.feature_names, index=empty_index)
            y = pd.Series(index=empty_index, dtype=float, name='label')
            return X, y
        
        # 合并所有日期的数据
        X_array = np.vstack(X_list)
        y_array = np.hstack(y_list)
        
        # 创建MultiIndex
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock_code'])
        
        # 构建DataFrame
        X = pd.DataFrame(X_array, index=multi_index, columns=self.feature_names)
        y = pd.Series(y_array, index=multi_index, name='label')
        
        logger.info(f"数据集构造完成: {len(X)} 个样本 ({len(dates)} 天)")
        
        return X, y
    
    def build_for_prediction(self, dates: List[pd.Timestamp]) -> pd.DataFrame:
        """
        构造实盘预测数据（不带label，用于最新日期的预测）
        
        参数：
        ------
        dates : List[pd.Timestamp]
            需要预测的日期列表（最近无label的日期）
            
        返回：
        ------
        pd.DataFrame
            - X: 特征矩阵，MultiIndex=(date, stock_code)，columns=feature_names
            - 不包含label
        """
        logger.info(f"构造实盘预测数据: {len(dates)} 个交易日")
        
        # 1. 加载所有因子数据
        factor_files = self._discover_factor_files()
        factor_data = {}
        
        for factor_name in self.feature_names:
            logger.debug(f"加载因子: {factor_name}")
            factor_data[factor_name] = self._load_factor_data(factor_name, dates)
        
        # 2. 逐日构造截面数据（不计算label）
        X_list = []
        index_tuples = []
        
        for date in dates:
            result = self._construct_cross_section(date, factor_data, label_series=None)
            
            if result is None:
                continue
            
            X_day, _, stocks_day = result
            
            X_list.append(X_day)
            
            # 构建MultiIndex
            for stock in stocks_day:
                index_tuples.append((date, stock))
        
        if len(X_list) == 0:
            logger.warning("无有效数据")
            # 返回空DataFrame
            empty_index = pd.MultiIndex.from_tuples([], names=['date', 'stock_code'])
            X = pd.DataFrame(columns=self.feature_names, index=empty_index)
            return X
        
        # 合并所有日期的数据
        X_array = np.vstack(X_list)
        
        # 创建MultiIndex
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock_code'])
        
        # 构建DataFrame
        X = pd.DataFrame(X_array, index=multi_index, columns=self.feature_names)
        
        logger.info(f"实盘预测数据构造完成: {len(X)} 个样本 ({len(dates)} 天)")
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名列表
        """
        if self.feature_names is None:
            self._discover_factor_files()
        return self.feature_names


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    import yaml
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 加载配置（从03模型训练层目录）
    script_dir = Path(__file__).parent
    if 'temp_modifications' in str(script_dir):
        config_path = script_dir.parent.parent / "03模型训练层" / "configs" / "default_config.yaml"
    else:
        config_path = script_dir.parent / "configs" / "default_config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 使用默认配置
        config = {
            'data': {
                'factor_paths': {
                    'technical': '02因子库/processed_data/technical_factors',
                    'financial': '02因子库/processed_data/financial_factors'
                },
                'market_data_path': '02因子库/processed_data/market_data',
                'price_column': 'close',
                'open_column': 'open',
                'label': {
                    'horizon': 20,
                    'use_open_price': True  # V1新增
                }
            }
        }
    
    print("=" * 80)
    print("测试 DataConstructorV1 (修复执行时点版本)")
    print("=" * 80)
    
    constructor = DataConstructorV1(config)
    
    # 测试加载close数据
    print("\n1. 测试加载收盘价数据:")
    try:
        close_df = constructor._load_close_data()
        print(f"   Shape: {close_df.shape}")
        print(f"   Date range: {close_df.index[0]} ~ {close_df.index[-1]}")
    except Exception as e:
        print(f"   跳过: {e}")
    
    # 测试加载open数据
    print("\n2. 测试加载开盘价数据:")
    try:
        open_df = constructor._load_open_data()
        print(f"   Shape: {open_df.shape}")
        print(f"   Date range: {open_df.index[0]} ~ {open_df.index[-1]}")
    except Exception as e:
        print(f"   跳过: {e}")
    
    # 测试发现因子
    print("\n3. 测试发现因子文件:")
    try:
        factor_files = constructor._discover_factor_files()
        print(f"   共发现 {len(factor_files)} 个因子")
        if factor_files:
            print(f"   前5个: {list(factor_files.keys())[:5]}")
    except Exception as e:
        print(f"   跳过: {e}")
    
    print("\n✓ DataConstructorV1 测试完成！")
    print("\n注意：此版本使用开盘价计算标签，更真实反映实盘可执行收益")
