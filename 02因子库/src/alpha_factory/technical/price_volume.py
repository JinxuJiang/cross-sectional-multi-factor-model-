# -*- coding: utf-8 -*-
"""
价格-成交量因子家族 (Price-Volume Factors)

包含：close_position, skew20, kurt20
"""

from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
def _skew_manual(x):
    """手动计算偏度 (scipy替代)"""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    skew = np.sum(((x - mean) / std) ** 3) * n / ((n-1) * (n-2))
    return skew

def _kurtosis_manual(x):
    """手动计算峰度 (scipy替代, excess kurtosis)"""
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    # 计算 excess kurtosis
    kurt = np.sum(((x - mean) / std) ** 4) * n * (n+1) / ((n-1) * (n-2) * (n-3)) - 3 * (n-1)**2 / ((n-2) * (n-3))
    return kurt


class PriceVolumeFactors:
    """价格-成交量因子计算类"""
    
    def __init__(self, market_data_path: Optional[str] = None, output_path: Optional[str] = None):
        current_file = Path(__file__).resolve()
        factor_lib_root = current_file.parent.parent.parent.parent
        
        self.market_data_path = Path(market_data_path) if market_data_path else factor_lib_root / 'processed_data' / 'market_data'
        self.output_path = Path(output_path) if output_path else factor_lib_root / 'processed_data' / 'factors' / 'technical'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._cache = {}
        self._dates = None
        self._stocks = None
    
    def _load(self, field: str):
        if field not in self._cache:
            path = self.market_data_path / f'{field}.parquet'
            if not path.exists():
                raise FileNotFoundError(f"{path} not found")
            self._cache[field] = pq.read_table(path)
        return self._cache[field]
    
    def _to_numpy(self, field='close'):
        table = self._load(field)
        columns = table.column_names
        self._dates = columns[0]
        self._stocks = columns[1:]
        data = [table.column(s).to_pylist() for s in self._stocks]
        matrix = np.array(data, dtype=np.float64).T
        dates = table.column(self._dates).to_pylist()
        return dates, self._stocks, matrix
    
    def _save(self, name, matrix, dates, stocks):
        arrays = [pa.array(dates, type=pa.timestamp('ns'))]
        names = ['time']
        for i, s in enumerate(stocks):
            col = [None if (v != v or np.isinf(v)) else float(v) for v in matrix[:, i]]
            arrays.append(pa.array(col, type=pa.float64()))
            names.append(s)
        output_file = self.output_path / f'{name}.parquet'
        pq.write_table(pa.table(arrays, names=names), output_file)
        return output_file
    
    def factor_close_position(self, save=True):
        """
        收盘价位置因子
        
        公式: (close - low) / (high - low)
        逻辑: 收盘价在日内高低点区间中的位置，>0.5表示收盘偏强
        无未来函数: 只用当日high/low/close
        """
        print("计算因子: close_position (收盘价位置)")
        dates, stocks, close = self._to_numpy('close')
        _, _, high = self._to_numpy('high')
        _, _, low = self._to_numpy('low')
        
        result = np.full_like(close, np.nan)
        price_range = high - low
        valid = price_range > 0
        result[valid] = (close[valid] - low[valid]) / price_range[valid]
        
        # 限制在[0,1]区间
        result = np.where((result >= 0) & (result <= 1), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('close_position', result, dates, stocks)
        return result
    
    def factor_skew20(self, save=True):
        """
        收益率偏度因子 (20日)
        
        公式: skew(ret1, 20)
        逻辑: 过去20日收益率分布的偏度，负偏表示左尾风险大
        无未来函数: 只用过去20日收益率
        """
        print("计算因子: skew20 (20日收益偏度)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # 滚动20日偏度
        period = 20
        n_dates, n_stocks = close.shape
        result = np.full_like(close, np.nan)
        
        for i in range(period - 1, n_dates):
            window = ret1[i - period + 1:i + 1, :]
            for j in range(n_stocks):
                w = window[:, j]
                valid = ~np.isnan(w)
                if np.sum(valid) >= 10:  # 至少10个有效值
                    try:
                        result[i, j] = stats.skew(w[valid], bias=False)
                    except:
                        pass
        
        # 极端值截断
        result = np.where((result > -5) & (result < 5), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('skew20', result, dates, stocks)
        return result
    
    def factor_kurt20(self, save=True):
        """
        收益率峰度因子 (20日)
        
        公式: kurtosis(ret1, 20)
        逻辑: 过去20日收益率分布的峰度，高峰度表示极端风险大
        无未来函数: 只用过去20日收益率
        """
        print("计算因子: kurt20 (20日收益峰度)")
        dates, stocks, close = self._to_numpy('close')
        _, _, pre_close = self._to_numpy('preClose')
        
        # 计算 ret1
        ret1 = np.full_like(close, np.nan)
        mask = pre_close > 0
        ret1[mask] = close[mask] / pre_close[mask] - 1
        
        # 滚动20日峰度
        period = 20
        n_dates, n_stocks = close.shape
        result = np.full_like(close, np.nan)
        
        for i in range(period - 1, n_dates):
            window = ret1[i - period + 1:i + 1, :]
            for j in range(n_stocks):
                w = window[:, j]
                valid = ~np.isnan(w)
                if np.sum(valid) >= 10:  # 至少10个有效值
                    try:
                        result[i, j] = stats.kurtosis(w[valid], bias=False)
                    except:
                        pass
        
        # 极端值截断
        result = np.where((result > -10) & (result < 20), result, np.nan)
        
        print(f"非NaN比例: {np.sum(~np.isnan(result)) / result.size:.2%}")
        if save:
            return self._save('kurt20', result, dates, stocks)
        return result
    
    def compute_all(self, factors: Optional[List[str]] = None):
        available = {
            'close_position': self.factor_close_position,
            'skew20': self.factor_skew20,
            'kurt20': self.factor_kurt20,
        }
        
        if factors is None:
            factors = list(available.keys())
        
        output_files = []
        for name in factors:
            if name in available:
                try:
                    result = available[name](save=True)
                    if result is not None:
                        output_files.append(result)
                except Exception as e:
                    print(f"计算因子 {name} 失败: {e}")
        
        print(f"\n完成！共计算 {len(output_files)} 个因子")
        return output_files


if __name__ == "__main__":
    print("=" * 60)
    print("价格-成交量因子计算")
    print("=" * 60)
    
    try:
        pv = PriceVolumeFactors()
        pv.compute_all()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
