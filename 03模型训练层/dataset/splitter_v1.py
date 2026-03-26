# -*- coding: utf-8 -*-
"""
Walk-forward 时间切分模块 V1 (修复版)

修复内容：
    1. 添加双重Gap防止数据泄露
       - gap_train_valid: 训练集和验证集之间的隔离期
       - gap_valid_test: 验证集和测试集之间的隔离期
    2. 两个gap默认等于label_horizon，确保标签计算不越界

使用示例：
    from dataset.splitter_v1 import WalkForwardSplitterV1
    
    splitter = WalkForwardSplitterV1(
        dates=trading_dates,
        train_window='3Y',
        valid_window='6M',
        test_window='3M',
        step='3M',
        label_horizon=20,  # 关键参数，决定gap大小
        gap_train_valid=20,  # 训练-验证gap（默认等于label_horizon）
        gap_valid_test=20    # 验证-测试gap（默认等于label_horizon）
    )
    
    for fold_id, (train_dates, valid_dates, test_dates) in enumerate(splitter.get_splits()):
        print(f"Fold {fold_id}: 训练{len(train_dates)}天, 验证{len(valid_dates)}天, 测试{len(test_dates)}天")
"""

import pandas as pd
from typing import List, Tuple, Optional, Iterator
from datetime import timedelta
import re


class WalkForwardSplitterV1:
    """
    Walk-forward 滚动时间切分器 V1 (修复数据泄露版本)
    
    参数：
    -----
    dates : List[pd.Timestamp] or pd.DatetimeIndex
        所有交易日列表（已排序）
    train_window : str
        训练窗口大小，如 '3Y' (3年), '6M' (6个月), '90D' (90天)
    valid_window : str
        验证窗口大小，如 '6M' (6个月)
    test_window : str
        测试窗口大小，如 '3M' (3个月)
    step : str
        滚动步长，如 '3M' (每3个月滚动一次)
    label_horizon : int
        标签计算的未来天数（决定gap的默认大小）
    gap_train_valid : int or str, optional
        训练集和验证集之间的gap天数，默认等于label_horizon
    gap_valid_test : int or str, optional
        验证集和测试集之间的gap天数，默认等于label_horizon
    start_date : str or pd.Timestamp, optional
        开始日期，早于该日期的数据不参与切分
    end_date : str or pd.Timestamp, optional
        结束日期，晚于该日期的数据不参与切分
    
    返回：
    ------
    Iterator[Tuple[int, Tuple[List, List, List]]]
        (fold_id, (train_dates, valid_dates, test_dates))
    """
    
    def __init__(
        self,
        dates: List[pd.Timestamp],
        train_window: str = '3Y',
        valid_window: str = '6M',
        test_window: str = '3M',
        step: str = '3M',
        label_horizon: int = 20,
        gap_train_valid: Optional[int] = None,
        gap_valid_test: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        初始化切分器
        """
        # 确保日期已排序且唯一
        self.dates = pd.DatetimeIndex(sorted(set(dates)))
        
        # 保存label_horizon
        self.label_horizon = label_horizon
        
        # 解析窗口大小
        self.train_window = self._parse_window(train_window)
        self.valid_window = self._parse_window(valid_window)
        self.test_window = self._parse_window(test_window)
        self.step = self._parse_window(step)
        
        # 解析gap（默认等于label_horizon + 1）
        # 原因：使用T+1开盘买入、T+(label_horizon+1)开盘卖出时，标签用到T+(label_horizon+1)的数据
        # 所以需要gap >= label_horizon + 1才能确保不泄露
        # gap_train_valid: 训练集和验证集之间的gap（交易日数量）
        self.gap_train_valid = gap_train_valid if gap_train_valid is not None else (label_horizon + 1)
        # gap_valid_test: 验证集和测试集之间的gap（交易日数量）
        self.gap_valid_test = gap_valid_test if gap_valid_test is not None else (label_horizon + 1)
        
        # 过滤日期范围
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
            self.dates = self.dates[self.dates >= start_date]
        
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
            self.dates = self.dates[self.dates <= end_date]
        
        if len(self.dates) == 0:
            raise ValueError("过滤后的日期列表为空，请检查start_date和end_date设置")
        
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]
        
        # 预计算所有splits
        self._splits = self._compute_splits()
    
    def _parse_window(self, window_str: str) -> pd.Timedelta:
        """
        解析窗口字符串为Timedelta
        
        支持格式：
        - '3Y' 或 '3y' : 3年
        - '6M' 或 '6m' : 6个月
        - '90D' 或 '90d' : 90天
        """
        pattern = r'(\d+)([YyMmDd])'
        match = re.match(pattern, window_str)
        
        if not match:
            raise ValueError(f"窗口格式错误: {window_str}, 应为如 '3Y', '6M', '90D'")
        
        value, unit = int(match.group(1)), match.group(2).upper()
        
        if unit == 'Y':
            # 按365天计算
            return pd.Timedelta(days=value * 365)
        elif unit == 'M':
            # 按30天估算（实际会在compute splits时按交易日处理）
            return pd.Timedelta(days=value * 30)
        elif unit == 'D':
            return pd.Timedelta(days=value)
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
    
    def _get_dates_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """
        获取指定日期范围内的所有交易日
        """
        mask = (self.dates >= start) & (self.dates <= end)
        return self.dates[mask].tolist()
    
    def _compute_splits(self) -> List[Tuple[List, List, List]]:
        """
        预计算所有Fold的切分（带双重Gap版本）
        
        修复：使用交易日索引计算gap，确保标签计算不越界
        注意：使用T+1开盘买入、T+(label_horizon+1)开盘卖出时，gap = label_horizon + 1
        
        返回：
        ------
        List[Tuple[train_dates, valid_dates, test_dates]]
        """
        splits = []
        
        # 创建日期到索引的映射，方便计算
        date_to_idx = {date: idx for idx, date in enumerate(self.dates)}
        
        # 找到开始和结束日期在dates中的索引
        start_idx = 0
        end_idx = len(self.dates) - 1
        
        if self.start_date > self.dates[0]:
            start_idx = self.dates.searchsorted(self.start_date)
        if self.end_date < self.dates[-1]:
            end_idx = self.dates.searchsorted(self.end_date) - 1
        
        # 估算窗口大小的交易日数量
        # train_window: 3年 ≈ 756交易日, valid_window: 6个月 ≈ 126交易日, test_window: 3个月 ≈ 63交易日
        train_days = int(self.train_window.days * 0.69)  # 日历天转交易日（约69%）
        valid_days = int(self.valid_window.days * 0.69)
        test_days = int(self.test_window.days * 0.69)
        step_days = int(self.step.days * 0.69)
        
        # 计算第一个fold的测试集开始索引
        # 需要预留：训练集 + gap1 + 验证集 + gap2
        min_required = train_days + self.gap_train_valid + valid_days + self.gap_valid_test
        current_test_start_idx = start_idx + min_required
        
        while True:
            # 检查是否超出范围
            if current_test_start_idx + test_days > end_idx:
                break
            
            # 测试集边界（使用交易日索引）
            test_start_idx = current_test_start_idx
            test_end_idx = min(test_start_idx + test_days, end_idx)
            
            # 验证集边界（使用交易日索引，留出gap_valid_test个交易日）
            valid_end_idx = test_start_idx - self.gap_valid_test
            if valid_end_idx < start_idx:
                current_test_start_idx += step_days
                continue
            valid_start_idx = max(valid_end_idx - valid_days, start_idx)
            
            # 训练集边界（使用交易日索引，留出gap_train_valid个交易日）
            train_end_idx = valid_start_idx - self.gap_train_valid
            if train_end_idx < start_idx:
                current_test_start_idx += step_days
                continue
            train_start_idx = max(train_end_idx - train_days, start_idx)
            
            # 获取各集合的交易日（切片不包含end，所以不用+1）
            train_dates = self.dates[train_start_idx:train_end_idx].tolist()
            valid_dates = self.dates[valid_start_idx:valid_end_idx].tolist()
            test_dates = self.dates[test_start_idx:test_end_idx].tolist()
            
            # 确保每个集合都有数据
            if len(train_dates) > 0 and len(valid_dates) > 0 and len(test_dates) > 0:
                splits.append((train_dates, valid_dates, test_dates))
            
            # 移动到下一个Fold
            current_test_start_idx += step_days
        
        return splits
    
    def get_splits(self) -> Iterator[Tuple[int, Tuple[List, List, List]]]:
        """
        获取所有Fold的切分迭代器
        
        返回：
        ------
        Iterator[Tuple[fold_id, (train_dates, valid_dates, test_dates)]]
        """
        for fold_id, split in enumerate(self._splits):
            yield fold_id, split
    
    def get_n_splits(self) -> int:
        """
        获取Fold总数
        """
        return len(self._splits)
    
    def get_split_info(self, fold_id: int) -> dict:
        """
        获取指定Fold的详细信息
        """
        if fold_id >= len(self._splits):
            raise ValueError(f"Fold {fold_id} 不存在，共 {len(self._splits)} 个Folds")
        
        train_dates, valid_dates, test_dates = self._splits[fold_id]
        
        # 计算gap的日期范围（用于显示）
        gap1_start = train_dates[-1] + pd.Timedelta(days=1) if train_dates else None
        gap1_end = valid_dates[0] - pd.Timedelta(days=1) if valid_dates else None
        gap2_start = valid_dates[-1] + pd.Timedelta(days=1) if valid_dates else None
        gap2_end = test_dates[0] - pd.Timedelta(days=1) if test_dates else None
        
        return {
            'fold_id': fold_id,
            'train_start': train_dates[0] if train_dates else None,
            'train_end': train_dates[-1] if train_dates else None,
            'train_size': len(train_dates),
            'gap1_start': gap1_start,
            'gap1_end': gap1_end,
            'gap1_days': (gap1_end - gap1_start).days if gap1_start and gap1_end else 0,
            'valid_start': valid_dates[0] if valid_dates else None,
            'valid_end': valid_dates[-1] if valid_dates else None,
            'valid_size': len(valid_dates),
            'gap2_start': gap2_start,
            'gap2_end': gap2_end,
            'gap2_days': (gap2_end - gap2_start).days if gap2_start and gap2_end else 0,
            'test_start': test_dates[0] if test_dates else None,
            'test_end': test_dates[-1] if test_dates else None,
            'test_size': len(test_dates),
        }
    
    def print_summary(self):
        """
        打印切分摘要信息
        """
        print("=" * 80)
        print("Walk-forward 切分摘要 V1 (修复数据泄露版本)")
        print("=" * 80)
        print(f"数据范围: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"总交易日: {len(self.dates)}")
        print(f"Fold数量: {len(self._splits)}")
        print()
        print("窗口配置:")
        print(f"  训练窗口: {self.train_window.days // 365}年")
        print(f"  验证窗口: {self.valid_window.days // 30}个月")
        print(f"  测试窗口: {self.test_window.days // 30}个月")
        print(f"  滚动步长: {self.step.days // 30}个月")
        print()
        print("Gap配置 (防止数据泄露):")
        print(f"  label_horizon: {self.label_horizon}天")
        print(f"  训练-验证gap: {self.gap_train_valid}个交易日 (使用T+1开盘买入、T+{self.label_horizon+1}开盘卖出，需要gap >= label_horizon + 1)")
        print(f"  验证-测试gap: {self.gap_valid_test}个交易日")
        print()
        
        if len(self._splits) > 0:
            print("前3个Fold示例:")
            for i in range(min(3, len(self._splits))):
                info = self.get_split_info(i)
                print(f"\nFold {i}:")
                print(f"  训练: {info['train_start'].strftime('%Y-%m-%d')} ~ {info['train_end'].strftime('%Y-%m-%d')} ({info['train_size']}天)")
                print(f"  gap1: {info['gap1_start'].strftime('%Y-%m-%d')} ~ {info['gap1_end'].strftime('%Y-%m-%d')} ({info['gap1_days']}天)")
                print(f"  验证: {info['valid_start'].strftime('%Y-%m-%d')} ~ {info['valid_end'].strftime('%Y-%m-%d')} ({info['valid_size']}天)")
                print(f"  gap2: {info['gap2_start'].strftime('%Y-%m-%d')} ~ {info['gap2_end'].strftime('%Y-%m-%d')} ({info['gap2_days']}天)")
                print(f"  测试: {info['test_start'].strftime('%Y-%m-%d')} ~ {info['test_end'].strftime('%Y-%m-%d')} ({info['test_size']}天)")
            
            if len(self._splits) > 3:
                print(f"\n... 共 {len(self._splits)} 个Folds")
        
        print("=" * 80)
    
    def verify_no_leakage(self) -> bool:
        """
        验证切分是否无数据泄露
        
        返回：
        ------
        bool : 验证是否通过
        """
        print("\n验证数据泄露检查...")
        print("-" * 60)
        
        all_passed = True
        
        for fold_id, (train_dates, valid_dates, test_dates) in self.get_splits():
            # 检查1: 训练集在验证集之前（考虑gap）
            train_last = train_dates[-1]
            valid_first = valid_dates[0]
            gap1_calendar_days = (valid_first - train_last).days - 1
            
            # 计算实际gap交易日数量
            train_last_idx = self.dates.get_loc(train_last)
            valid_first_idx = self.dates.get_loc(valid_first)
            gap1_trading_days = valid_first_idx - train_last_idx - 1
            
            if train_last >= valid_first:
                print(f"[FAIL] Fold {fold_id}: 训练集和验证集重叠！")
                all_passed = False
            elif gap1_trading_days < self.gap_train_valid:
                print(f"[WARN] Fold {fold_id}: gap1实际={gap1_trading_days}个交易日 < 设置={self.gap_train_valid}个交易日")
            
            # 检查2: 验证集在测试集之前（考虑gap）
            valid_last = valid_dates[-1]
            test_first = test_dates[0]
            gap2_calendar_days = (test_first - valid_last).days - 1
            
            # 计算实际gap交易日数量
            valid_last_idx = self.dates.get_loc(valid_last)
            test_first_idx = self.dates.get_loc(test_first)
            gap2_trading_days = test_first_idx - valid_last_idx - 1
            
            if valid_last >= test_first:
                print(f"[FAIL] Fold {fold_id}: 验证集和测试集重叠！")
                all_passed = False
            elif gap2_trading_days < self.gap_valid_test:
                print(f"[WARN] Fold {fold_id}: gap2实际={gap2_trading_days}个交易日 < 设置={self.gap_valid_test}个交易日")
            
            # 检查3: 标签计算是否越界
            # 使用T+1开盘买入、T+(label_horizon+1)开盘卖出
            # 训练集最后一天的标签用到: train_last 后的第 label_horizon + 1 个交易日
            train_last_idx = self.dates.get_loc(train_last)
            train_label_end_idx = train_last_idx + self.label_horizon + 1  # T+21开盘
            if train_label_end_idx >= valid_first_idx:
                print(f"[FAIL] Fold {fold_id}: 训练集标签泄露到验证集！标签用到第{train_label_end_idx}天，验证集从第{valid_first_idx}天开始")
                all_passed = False
            
            # 验证集最后一天的标签用到: valid_last 后的第 label_horizon + 1 个交易日
            valid_last_idx = self.dates.get_loc(valid_last)
            valid_label_end_idx = valid_last_idx + self.label_horizon + 1  # T+21开盘
            if valid_label_end_idx >= test_first_idx:
                print(f"[FAIL] Fold {fold_id}: 验证集标签泄露到测试集！标签用到第{valid_label_end_idx}天，测试集从第{test_first_idx}天开始")
                all_passed = False
        
        if all_passed:
            print("[PASS] 所有Fold通过泄露检查！")
        
        return all_passed


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("测试 WalkForwardSplitterV1 (修复数据泄露版本)...")
    
    # 构造测试数据：2010-01-01 到 2025-12-31 的交易日
    test_dates = pd.date_range(start='2010-01-01', end='2025-12-31', freq='B')  # 工作日
    print(f"测试数据: {len(test_dates)} 个交易日 ({test_dates[0].date()} ~ {test_dates[-1].date()})")
    print()
    
    # 创建切分器
    splitter = WalkForwardSplitterV1(
        dates=test_dates,
        train_window='3Y',
        valid_window='6M',
        test_window='3M',
        step='3M',
        label_horizon=20,  # 关键参数
        gap_train_valid=20,  # 训练-验证gap
        gap_valid_test=20    # 验证-测试gap
    )
    
    # 打印摘要
    splitter.print_summary()
    
    # 验证无泄露
    splitter.verify_no_leakage()
    
    print("\n✓ WalkForwardSplitterV1 测试完成！")
