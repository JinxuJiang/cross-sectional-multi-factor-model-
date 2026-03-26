"""
Backtrader Eval V3 - 截面多因子模型回测脚本 (支持双模式)
===========================================================

使用说明:
---------
本脚本支持两种仓位管理模式，用于不同场景的回测验证。

使用模板 (直接复制修改):
--------------------------

# 模式1: 百分比仓位模式 (默认，带单只上限)
python backtrader_eval_v3.py --exp-id exp_001 --position-mode percent

# 模式2: 固定金额模式
python backtrader_eval_v3.py --exp-id exp_001 --position-mode fixed --fixed-amount-per-stock 30000

# 完整参数示例 - 百分比模式
python backtrader_eval_v3.py \
    --exp-id exp_001 \
    --start-date 2020-01-01 \
    --end-date 2021-12-31 \
    --top-n 30 \
    --rebalance-period 20 \
    --initial-cash 100000 \
    --commission 0.002 \
    --position-mode percent \
    --position-pct 0.90 \
    --max-cash-per-stock 50000

# 完整参数示例 - 固定金额模式
python backtrader_eval_v3.py \
    --exp-id exp_001 \
    --start-date 2020-01-01 \
    --end-date 2021-12-31 \
    --top-n 30 \
    --rebalance-period 20 \
    --initial-cash 100000 \
    --commission 0.002 \
    --position-mode fixed \
    --fixed-amount-per-stock 30000

参数说明:
---------
--exp-id                : 实验ID，对应 03模型训练层/experiments/{exp_id}  (必需)
--start-date            : 回测开始日期，格式 YYYY-MM-DD，默认 2010-01-01
--end-date              : 回测结束日期，格式 YYYY-MM-DD，默认 2026-02-27
--top-n                 : 每次调仓选股数量，默认 30
--rebalance-period      : 调仓周期(交易日)，默认 20
--initial-cash          : 初始资金，默认 100000
--commission            : 手续费率，默认 0.002
--position-mode         : 仓位模式，percent(百分比) 或 fixed(固定金额)，默认 percent
--position-pct          : [percent模式] 总仓位比例，默认 0.90
--max-cash-per-stock    : [percent模式] 单只股票金额上限，默认 50000
--fixed-amount-per-stock: [fixed模式] 每只股票固定买入金额，默认 30000

两种模式说明:
-------------

1. percent 模式 (百分比仓位 + 单只上限)
   - 适用场景: 小资金回测，模拟复利增长
   - 计算方式: 每只金额 = min(总资金*仓位比例/股票数, 单只上限)
   - 收益计算: 复利收益率 (final_value / initial_cash - 1)
   - 特点: 资金增长后，每只买入金额会增加，直到触及上限

2. fixed 模式 (固定金额)
   - 适用场景: 大资金容量测试，防止复利爆炸
   - 计算方式: 每只金额 = 固定金额 (不看账户总资金)
   - 收益计算: 绝对收益额 和 基于初始资金的收益率
   - 特点: 线性收益，便于衡量策略在固定资金下的表现

输出文件:
---------
04回测层/reports/{exp_id}/
    ├── performance.json         # 策略表现指标
    ├── equity_curve.png         # 收益曲线图
    ├── trade_signals.csv        # 详细交易记录
    └── backtrader_report.html   # HTML汇总报告

"""

import pandas as pd
import numpy as np
import gc
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import json
import base64
from io import BytesIO

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================

def get_paths(exp_id):
    """根据exp_id生成数据路径"""
    base_path = Path(__file__).parent.parent
    return {
        'open': os.path.join(base_path, '02因子库', 'processed_data', 'market_data', 'open.parquet'),
        'close': os.path.join(base_path, '02因子库', 'processed_data', 'market_data', 'close.parquet'),
        'high': os.path.join(base_path, '02因子库', 'processed_data', 'market_data', 'high.parquet'),
        'low': os.path.join(base_path, '02因子库', 'processed_data', 'market_data', 'low.parquet'),
        'volume': os.path.join(base_path, '02因子库', 'processed_data', 'market_data', 'volume.parquet'),
        'pred': os.path.join(base_path, '03模型训练层', 'experiments', exp_id, 'predictions.parquet'),
        'live_pred': os.path.join(base_path, '03模型训练层', 'experiments', exp_id, 'live_predictions.parquet'),
        'report_dir': os.path.join(base_path, '04回测层', 'reports', exp_id)
    }

# ==========================================
# 2. 数据处理模块
# ==========================================
def wide_to_long(df_wide, value_name, time_col='time'):
    """宽表转长表"""
    if time_col not in df_wide.columns:
        if df_wide.index.name == time_col:
            df_wide = df_wide.reset_index()
        else:
            df_wide.index.name = time_col
            df_wide = df_wide.reset_index()

    df_wide = df_wide.set_index(time_col)
    df_long = df_wide.stack().reset_index()
    df_long.columns = [time_col, 'stock_code', value_name]
    
    df_long = df_long[df_long['stock_code'].str.match(r'^\d{6}\.(SZ|SH|BJ)$', na=False)]
    df_long[value_name] = df_long[value_name].astype('float32')
    df_long = df_long.dropna(subset=[value_name])
    return df_long

def load_and_merge_data(paths):
    """加载并合并数据"""
    print("--- 步骤1: 加载预测数据 ---")
    prediction = pd.read_parquet(paths['pred'], columns=['date', 'stock_code', 'pred_score'])
    live_prediction = pd.read_parquet(paths['live_pred'], columns=['date', 'stock_code', 'pred_score'])
    
    pred_total = pd.concat([prediction, live_prediction], axis=0)
    pred_total = pred_total.rename(columns={'date': 'time', 'pred_score': 'prediction'})
    pred_total['time'] = pd.to_datetime(pred_total['time'])
    pred_total['prediction'] = pred_total['prediction'].astype('float32')
    
    del prediction, live_prediction
    gc.collect()

    main_df = pred_total
    for col in ['open', 'close', 'high', 'low', 'volume']:
        print(f"正在处理 {col} 数据...")
        temp_wide = pd.read_parquet(paths[col])
        temp_long = wide_to_long(temp_wide, col)
        
        main_df = pd.merge(main_df, temp_long, on=['time', 'stock_code'], how='left')
        del temp_wide, temp_long
        gc.collect()

    main_df['openinterest'] = 0
    main_df['datetime'] = pd.to_datetime(main_df['time'])
    main_df = main_df.set_index('datetime')
    main_df['stock_code'] = main_df['stock_code'].astype('category')
    
    # 删除有缺失值的行
    main_df = main_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    print(f"完成数据合并，形状: {main_df.shape}")
    print(f"数据时间范围: {main_df.index.min()} ~ {main_df.index.max()}")
    return main_df

# ==========================================
# 3. 信号生成模块
# ==========================================
def generate_signals(df, period, top_n, start_date, end_date):
    """生成调仓信号"""
    # 过滤到回测区间（稍微扩展）
    mask = (df.index >= pd.to_datetime(start_date) - pd.Timedelta(days=5)) & \
           (df.index <= pd.to_datetime(end_date) + pd.Timedelta(days=5))
    df_period = df.loc[mask].copy()
    
    buy_dict = {}
    sell_dict = {}
    current_position = set()
    all_held_stocks = set()
    
    trading_days = sorted(df_period.index.unique())
    if not trading_days:
        return {}, {}, []
    
    # 找到第一个大于等于start_date的交易日
    start_idx = 0
    for i, d in enumerate(trading_days):
        if d >= pd.to_datetime(start_date):
            start_idx = i
            break
    
    rebalance_dates = trading_days[start_idx::period]
    
    print(f"\n--- 步骤2: 生成调仓信号 ---")
    print(f"回测范围内总交易日: {len(trading_days)}, 计划调仓次数: {len(rebalance_dates)}")
    print(f"第一个调仓日: {rebalance_dates[0] if rebalance_dates else 'None'}")
    print(f"最后一个调仓日: {rebalance_dates[-1] if rebalance_dates else 'None'}")
    
    for date in rebalance_dates:
        if date > pd.to_datetime(end_date):
            break
            
        date_str = date.strftime('%Y-%m-%d')
        try:
            current_slice = df_period.loc[date]
        except KeyError:
            continue
            
        if isinstance(current_slice, pd.Series):
            current_slice = current_slice.to_frame().T
            
        if current_slice.empty:
            continue
            
        selected = current_slice.sort_values(by='prediction', ascending=False).head(top_n)
        buy_list = selected['stock_code'].tolist()
        buy_list_set = set(buy_list)
        
        sell_list = sorted(list(current_position - buy_list_set))
        
        buy_dict[date_str] = buy_list
        sell_dict[date_str] = sell_list
        current_position = buy_list_set
        all_held_stocks.update(current_position)
        
        print(f"调仓日: {date_str} | 买入{len(buy_list)}只 | 卖出{len(sell_list)}只")
        
    return buy_dict, sell_dict, sorted(list(all_held_stocks))

# ==========================================
# 4. Backtrader 策略类 (双模式支持)
# ==========================================
class MyMultiFactorStrategy(bt.Strategy):
    params = (
        ('buy_date', None),
        ('sell_date', None),
        ('trade_signals_path', None),
        ('position_mode', 'percent'),       # 'percent' 或 'fixed'
        ('position_pct', 0.90),             # percent模式: 总仓位比例
        ('max_cash_per_stock', 50000),      # percent模式: 单只上限
        ('fixed_amount_per_stock', 30000),  # fixed模式: 每只固定金额
    )

    def __init__(self):
        self.trade_count = 0
        self.prenext_count = 0
        self.trade_records = []
        self.daily_nav = {}
        
        # 记录fixed模式下的累计投入成本
        self.total_cost_basis = 0
        
        # 如果文件存在，先删除
        if self.p.trade_signals_path and os.path.exists(self.p.trade_signals_path):
            os.remove(self.p.trade_signals_path)

    def prenext(self):
        """在数据未完全准备好时也执行交易逻辑"""
        self.prenext_count += 1
        self.next()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def calculate_position_size(self, price, valid_stocks):
        """
        根据仓位模式计算买入股数
        
        Returns:
            dict: {stock_code: size} 每只股票的买入股数
        """
        if price <= 0:
            return 0
            
        if self.p.position_mode == 'percent':
            # 百分比模式: 等权分配，但有单只上限
            total_cash = self.broker.getvalue() * self.p.position_pct
            cash_per_stock = total_cash / len(valid_stocks)
            # 应用单只上限
            cash_per_stock = min(cash_per_stock, self.p.max_cash_per_stock)
        else:
            # 固定金额模式: 每只固定金额，不看总资金
            cash_per_stock = self.p.fixed_amount_per_stock
        
        # 计算股数（100的整数倍）
        size = int(cash_per_stock / price / 100) * 100
        return size

    def next(self):
        curr_dt = self.datetime.date(0).strftime('%Y-%m-%d')
        
        # 记录每日净值
        self.daily_nav[curr_dt] = self.broker.getvalue()
        
        # 检查是否是调仓日
        is_buy_day = curr_dt in self.p.buy_date
        is_sell_day = curr_dt in self.p.sell_date
        
        if not is_buy_day and not is_sell_day:
            return
        
        # 是调仓日，执行交易
        self.trade_count += 1
        action = "初始建仓" if self.trade_count == 1 else f"调仓 #{self.trade_count}"
        print(f"\n--- {action}: {curr_dt} ---")
        print(f"当前净值: {self.broker.getvalue():.2f}")
        print(f"仓位模式: {self.p.position_mode}")
        
        # 1. 先执行卖出
        if is_sell_day and self.p.sell_date[curr_dt]:
            s_list = self.p.sell_date[curr_dt]
            print(f"  卖出 {len(s_list)} 只")
            for s_code in s_list:
                try:
                    if s_code in self.getdatanames():
                        data = self.getdatabyname(s_code)
                        pos = self.getposition(data)
                        if pos.size > 0:
                            order = self.order_target_percent(data=data, target=0)
                except Exception as e:
                    continue

        # 2. 再执行买入
        if is_buy_day and self.p.buy_date[curr_dt]:
            b_list = self.p.buy_date[curr_dt]
            if len(b_list) > 0:
                valid_stocks = [s for s in b_list if s in self.getdatanames()]
                
                if valid_stocks:
                    print(f"  买入 {len(valid_stocks)} 只")
                    
                    for i, b_code in enumerate(valid_stocks):
                        try:
                            data = self.getdatabyname(b_code)
                            close_price = data.close[0]
                            
                            # 计算买入股数
                            size = self.calculate_position_size(close_price, valid_stocks)
                            
                            if size > 0:
                                order = self.buy(data=data, size=size)
                                
                                # fixed模式下记录成本
                                if self.p.position_mode == 'fixed':
                                    self.total_cost_basis += size * close_price
                            
                            if (i+1) % 10 == 0:
                                print(f"    进度: {i+1}/{len(valid_stocks)}")
                                
                        except Exception as e:
                            continue

    def notify_order(self, order):
        """订单状态通知，成交后记录交易信号"""
        if order.status == order.Completed:
            dt = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
            stock_code = order.data._name
            action = 'BUY' if order.isbuy() else 'SELL'
            shares = order.executed.size
            price = order.executed.price
            value = order.executed.value
            commission = order.executed.comm
            
            record = {
                'date': dt,
                'stock_code': stock_code,
                'action': action,
                'shares': shares,
                'price': round(price, 4),
                'value': round(value, 2),
                'commission': round(commission, 4),
                'portfolio_value_before': round(self.broker.getvalue() - (value if action == 'BUY' else -value), 2),
                'position_mode': self.p.position_mode,
            }
            
            self.trade_records.append(record)
            
            # 立即写入CSV
            if self.p.trade_signals_path:
                df = pd.DataFrame([record])
                header = not os.path.exists(self.p.trade_signals_path)
                df.to_csv(self.p.trade_signals_path, mode='a', index=False, header=header)

    def stop(self):
        """回测结束时打印统计"""
        print(f"\n>>> 回测结束")
        print(f">>> 总调仓次数: {self.trade_count}")
        print(f">>> prenext 调用次数: {self.prenext_count}")
        print(f">>> 仓位模式: {self.p.position_mode}")
        if self.p.position_mode == 'fixed':
            print(f">>> 累计投入成本(估算): {self.total_cost_basis:.2f}")

# ==========================================
# 5. 回测引擎模块
# ==========================================
def run_backtest(full_df, buy_date, sell_date, stock_list, config, report_dir):
    cerebro = bt.Cerebro(runonce=False)
    
    print("\n--- 步骤3: 加载 Backtrader 数据源 ---")
    total_stocks = len(stock_list)
    valid_count = 0
    
    for i, stock_code in enumerate(stock_list):
        data_slice = full_df[full_df['stock_code'] == stock_code][['open', 'high', 'low', 'close', 'volume', 'openinterest']]
        
        if data_slice.empty:
            continue
        
        data_slice = data_slice[(data_slice.index >= config['start_date']) & 
                                (data_slice.index <= config['end_date'])]
        
        if len(data_slice) < 20:
            continue
            
        data = bt.feeds.PandasData(
            dataname=data_slice,
            fromdate=config['start_date'],
            todate=config['end_date']
        )
        cerebro.adddata(data, name=stock_code)
        valid_count += 1
        
        if (i+1) % 100 == 0 or (i+1) == total_stocks:
            print(f"加载进度: {i+1}/{total_stocks} (成功: {valid_count})")

    print(f"\n成功加载 {valid_count} 只股票")
    
    if valid_count == 0:
        print("错误：没有成功加载任何股票数据！")
        return None, None, None

    # 交易信号文件路径
    trade_signals_path = os.path.join(report_dir, 'trade_signals.csv')

    # 设置策略
    cerebro.addstrategy(
        MyMultiFactorStrategy,
        buy_date=buy_date,
        sell_date=sell_date,
        trade_signals_path=trade_signals_path,
        position_mode=config['position_mode'],
        position_pct=config.get('position_pct', 0.90),
        max_cash_per_stock=config.get('max_cash_per_stock', 50000),
        fixed_amount_per_stock=config.get('fixed_amount_per_stock', 30000),
    )
    
    # 设置资金和手续费
    cerebro.broker.setcash(config['initial_cash'])
    cerebro.broker.setcommission(commission=config['commission'])
    
    # 设置滑点 (千分之一)
    cerebro.broker.set_slippage_perc(perc=0.001)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='Sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='Drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')

    print("\n--- 步骤4: 启动回测引擎 ---")
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    print(f"仓位模式: {config['position_mode']}")
    
    results = cerebro.run(runonce=False)
    
    # 获取结果
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    print(f"\n" + "="*50)
    print(f"最终净值: {final_value:.2f}")
    
    # 根据模式计算收益
    if config['position_mode'] == 'fixed':
        # fixed模式: 计算绝对收益和基于成本的收益率
        # 估算总投入成本 = 初始资金 - 最终现金（简化计算）
        # 更准确的是用累计买入金额，这里用净值变化近似
        total_pnl = final_value - config['initial_cash']
        total_return_pct = (total_pnl / config['initial_cash']) * 100
        print(f"绝对收益额: {total_pnl:.2f}")
        print(f"总收益率(基于初始资金): {total_return_pct:.2f}%")
    else:
        # percent模式: 复利收益率
        total_return_pct = (final_value / config['initial_cash'] - 1) * 100
        print(f"总收益率(复利): {total_return_pct:.2f}%")
    
    # 提取分析数值
    performance = {}
    try:
        ret_analysis = strat.analyzers.Returns.get_analysis()
        sharpe_analysis = strat.analyzers.Sharpe.get_analysis()
        dd_analysis = strat.analyzers.Drawdown.get_analysis()
        trade_analysis = strat.analyzers.TradeAnalyzer.get_analysis()

        ann_return = ret_analysis.get('rnorm100', 0)
        sharpe = sharpe_analysis.get('sharperatio', 0)
        max_dd = dd_analysis.max.drawdown if hasattr(dd_analysis, 'max') else 0
        
        # 交易统计
        total_trades = trade_analysis.get('total', {}).get('closed', 0) if trade_analysis else 0
        won_trades = trade_analysis.get('won', {}).get('total', 0) if trade_analysis else 0
        lost_trades = trade_analysis.get('lost', {}).get('total', 0) if trade_analysis else 0

        print(f"年化收益率: {ann_return:.2f}%")
        print(f"夏普比率: {sharpe:.2f}" if sharpe else "夏普比率: N/A")
        print(f"最大回撤: {max_dd:.2f}%")
        print(f"总交易次数: {total_trades}")
        if total_trades > 0:
            print(f"盈利次数: {won_trades}, 亏损次数: {lost_trades}")
            print(f"胜率: {won_trades/total_trades:.2%}")
        
        performance = {
            'exp_id': config['exp_id'],
            'start_date': config['start_date'].strftime('%Y-%m-%d'),
            'end_date': config['end_date'].strftime('%Y-%m-%d'),
            'initial_cash': config['initial_cash'],
            'final_value': round(final_value, 2),
            'position_mode': config['position_mode'],
            'total_return_pct': round(total_return_pct, 2),
            'annual_return_pct': round(ann_return, 2),
            'sharpe_ratio': round(sharpe, 4) if sharpe else None,
            'max_drawdown_pct': round(max_dd, 2) if max_dd else None,
            'commission': config['commission'],
            'top_n': config['top_n'],
            'rebalance_period': config['rebalance_period'],
            'trade_count': strat.trade_count,
            'total_trades': int(total_trades),
            'won_trades': int(won_trades),
            'lost_trades': int(lost_trades),
            'win_rate': round(won_trades/total_trades, 4) if total_trades > 0 else 0,
        }
        
        # 根据模式添加特定参数
        if config['position_mode'] == 'percent':
            performance['position_pct'] = config.get('position_pct', 0.90)
            performance['max_cash_per_stock'] = config.get('max_cash_per_stock', 50000)
        else:
            performance['fixed_amount_per_stock'] = config.get('fixed_amount_per_stock', 30000)
            performance['absolute_pnl'] = round(total_pnl, 2)
            
    except Exception as e:
        print(f"获取分析指标时出错: {e}")
        performance = {
            'exp_id': config['exp_id'],
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return_pct, 2),
            'position_mode': config['position_mode'],
            'error': str(e)
        }
        
    print("="*50)
    
    return cerebro, performance, strat.daily_nav

# ==========================================
# 6. HTML报告生成
# ==========================================
def generate_html_report(performance, report_dir, equity_curve_path, trade_signals_path):
    """生成HTML汇总报告"""
    
    # 读取收益曲线图片并转为base64
    with open(equity_curve_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 读取最近的交易记录
    recent_trades = ""
    if os.path.exists(trade_signals_path):
        trades_df = pd.read_csv(trade_signals_path)
        if not trades_df.empty:
            recent = trades_df.tail(20)
            recent_trades = recent.to_html(index=False, classes='table table-striped', border=0)
        else:
            recent_trades = "<p>暂无交易记录</p>"
    else:
        recent_trades = "<p>暂无交易记录</p>"

    # 根据模式显示不同的参数
    mode_specific_params = ""
    if performance.get('position_mode') == 'percent':
        mode_specific_params = f"""
            <tr><td>仓位比例</td><td>{performance.get('position_pct', 'N/A')}</td></tr>
            <tr><td>单只上限</td><td>{performance.get('max_cash_per_stock', 'N/A'):,.0f}</td></tr>
        """
    else:
        mode_specific_params = f"""
            <tr><td>固定金额(每只股票)</td><td>{performance.get('fixed_amount_per_stock', 'N/A'):,.0f}</td></tr>
            <tr><td>绝对收益额</td><td>{performance.get('absolute_pnl', 'N/A'):,.2f}</td></tr>
        """

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtrader回测报告 - {performance['exp_id']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .info-box {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card.negative {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-card.positive {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .equity-curve {{
            text-align: center;
            margin: 30px 0;
        }}
        .equity-curve img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
        .mode-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            background-color: {'#4facfe' if performance.get('position_mode') == 'percent' else '#f5576c'};
            color: white;
        }}
    </style>
</head>
<body>
    <h1>Backtrader 回测报告</h1>
    <p style="color: #666;">
        实验ID: <strong>{performance['exp_id']}</strong> 
        <span class="mode-badge">{performance.get('position_mode', 'percent').upper()} 模式</span>
    </p>
    
    <div class="info-box">
        <h2>回测配置</h2>
        <table>
            <tr><th>参数</th><th>值</th></tr>
            <tr><td>回测区间</td><td>{performance.get('start_date', 'N/A')} ~ {performance.get('end_date', 'N/A')}</td></tr>
            <tr><td>初始资金</td><td>{performance.get('initial_cash', 'N/A'):,.0f}</td></tr>
            <tr><td>选股数量</td><td>{performance.get('top_n', 'N/A')}</td></tr>
            <tr><td>调仓周期</td><td>{performance.get('rebalance_period', 'N/A')} 个交易日</td></tr>
            <tr><td>手续费率</td><td>{performance.get('commission', 'N/A'):.4f}</td></tr>
            <tr><td>滑点</td><td>0.001 (千分之一)</td></tr>
            {mode_specific_params}
        </table>
    </div>
    
    <div class="info-box">
        <h2>绩效指标</h2>
        <div class="metrics-grid">
            <div class="metric-card {'positive' if performance.get('total_return_pct', 0) >= 0 else 'negative'}">
                <div class="metric-label">总收益率</div>
                <div class="metric-value">{performance.get('total_return_pct', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value">{performance.get('annual_return_pct', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{performance.get('sharpe_ratio', 'N/A') if performance.get('sharpe_ratio') is not None else 'N/A'}</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value">{performance.get('max_drawdown_pct', 'N/A') if performance.get('max_drawdown_pct') is not None else 'N/A'}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最终净值</div>
                <div class="metric-value">{performance.get('final_value', 0):,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">总交易次数</div>
                <div class="metric-value">{performance.get('total_trades', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">胜率</div>
                <div class="metric-value">{performance.get('win_rate', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">盈利/亏损</div>
                <div class="metric-value">{performance.get('won_trades', 0)}/{performance.get('lost_trades', 0)}</div>
            </div>
        </div>
    </div>
    
    <div class="info-box">
        <h2>收益曲线</h2>
        <div class="equity-curve">
            <img src="data:image/png;base64,{img_base64}" alt="Equity Curve">
        </div>
    </div>
    
    <div class="info-box">
        <h2>最近交易记录</h2>
        {recent_trades}
    </div>
    
    <div class="footer">
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Backtrader Eval V3 - 双模式回测</p>
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(report_dir, 'backtrader_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML报告已生成: {html_path}")
    return html_path

# ==========================================
# 7. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description='Backtrader回测 - V3版本 (支持百分比/固定金额双模式)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 百分比模式 (默认)
  python backtrader_eval_v3.py --exp-id exp_001
  
  # 固定金额模式
  python backtrader_eval_v3.py --exp-id exp_001 --position-mode fixed --fixed-amount-per-stock 30000
        """
    )
    
    # 基础参数
    parser.add_argument('--exp-id', type=str, required=True, 
                        help='实验ID，如 exp_001 (必需)')
    parser.add_argument('--start-date', type=str, default='2010-01-01', 
                        help='回测开始日期，格式 YYYY-MM-DD (默认: 2010-01-01)')
    parser.add_argument('--end-date', type=str, default='2026-02-27', 
                        help='回测结束日期，格式 YYYY-MM-DD (默认: 2026-02-27)')
    parser.add_argument('--top-n', type=int, default=30, 
                        help='选股数量 (默认: 30)')
    parser.add_argument('--rebalance-period', type=int, default=20, 
                        help='调仓周期(交易日) (默认: 20)')
    parser.add_argument('--initial-cash', type=float, default=100000, 
                        help='初始资金 (默认: 100000)')
    parser.add_argument('--commission', type=float, default=0.002, 
                        help='手续费率 (默认: 0.002)')
    
    # 仓位模式参数
    parser.add_argument('--position-mode', type=str, default='percent', 
                        choices=['percent', 'fixed'],
                        help='仓位模式: percent(百分比) 或 fixed(固定金额) (默认: percent)')
    
    # 百分比模式专用参数
    parser.add_argument('--position-pct', type=float, default=0.90, 
                        help='[percent模式] 总仓位比例 (默认: 0.90)')
    parser.add_argument('--max-cash-per-stock', type=float, default=50000, 
                        help='[percent模式] 单只股票金额上限 (默认: 50000)')
    
    # 固定金额模式专用参数
    parser.add_argument('--fixed-amount-per-stock', type=float, default=30000, 
                        help='[fixed模式] 每只股票固定买入金额 (默认: 30000)')
    
    args = parser.parse_args()
    
    # 解析日期
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # 获取路径
    paths = get_paths(args.exp_id)
    
    # 创建报告目录
    report_dir = paths['report_dir']
    os.makedirs(report_dir, exist_ok=True)
    print(f"报告将输出到: {report_dir}")
    
    # 配置参数
    config = {
        'exp_id': args.exp_id,
        'start_date': start_date,
        'end_date': end_date,
        'top_n': args.top_n,
        'rebalance_period': args.rebalance_period,
        'initial_cash': args.initial_cash,
        'commission': args.commission,
        'position_mode': args.position_mode,
        'position_pct': args.position_pct,
        'max_cash_per_stock': args.max_cash_per_stock,
        'fixed_amount_per_stock': args.fixed_amount_per_stock,
    }
    
    print("\n" + "="*50)
    print(f"回测配置:")
    print(f"  实验ID: {config['exp_id']}")
    print(f"  回测区间: {config['start_date'].strftime('%Y-%m-%d')} ~ {config['end_date'].strftime('%Y-%m-%d')}")
    print(f"  选股数量: {config['top_n']}")
    print(f"  调仓周期: {config['rebalance_period']} 个交易日")
    print(f"  初始资金: {config['initial_cash']:,.0f}")
    print(f"  手续费率: {config['commission']:.4f}")
    print(f"  仓位模式: {config['position_mode'].upper()}")
    if config['position_mode'] == 'percent':
        print(f"  仓位比例: {config['position_pct']:.0%}")
        print(f"  单只上限: {config['max_cash_per_stock']:,.0f}")
    else:
        print(f"  固定金额(每只): {config['fixed_amount_per_stock']:,.0f}")
    print("="*50 + "\n")
    
    # 加载数据
    master_df = load_and_merge_data(paths)
    
    # 诊断信息
    print("\n[诊断信息]:")
    print(f"主表日期范围: {master_df.index.min()} 到 {master_df.index.max()}")
    print(f"非空预测评分的行数: {master_df['prediction'].notna().sum()}")
    
    # 生成信号
    buy_date, sell_date, stock_list = generate_signals(
        master_df,
        config['rebalance_period'],
        config['top_n'],
        config['start_date'],
        config['end_date']
    )
    
    print(f"\n涉及股票总数: {len(stock_list)}")
    print(f"调仓日期列表: {list(buy_date.keys())}")
    
    if not buy_date:
        print("错误：没有生成任何调仓信号！")
        return
    
    # 运行回测
    cerebro, performance, daily_nav = run_backtest(master_df, buy_date, sell_date, stock_list, config, report_dir)
    
    if cerebro is None:
        print("回测失败")
        return
    
    # 保存性能指标
    performance_path = os.path.join(report_dir, 'performance.json')
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2, ensure_ascii=False)
    print(f"性能指标已保存: {performance_path}")
    
    # 绘制并保存收益曲线
    if daily_nav:
        plt.figure(figsize=(12, 6))
        equity_df = pd.Series(daily_nav).sort_index()
        equity_df.plot(title='Strategy Equity Curve')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=config['initial_cash'], color='r', linestyle='--', alpha=0.5, label='Initial')
        plt.legend()
        
        equity_curve_path = os.path.join(report_dir, 'equity_curve.png')
        plt.savefig(equity_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"收益曲线已保存: {equity_curve_path}")
    else:
        equity_curve_path = None
    
    # 生成HTML报告
    trade_signals_path = os.path.join(report_dir, 'trade_signals.csv')
    if equity_curve_path:
        generate_html_report(performance, report_dir, equity_curve_path, trade_signals_path)
    
    print("\n" + "="*50)
    print("回测完成！")
    print(f"输出目录: {report_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
