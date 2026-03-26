"""
多因子回测脚本 - Backtrader Eval 1.1
====================================

使用示例:
---------
# 基本用法（使用 predictions.parquet）
python backtrader.eval_1.1.py --exp-id exp_001

# 指定实验ID
python backtrader.eval_1.1.py -e test_001_fined_v1

# 使用平滑后的预测（smoothed_predictions.parquet）
python backtrader.eval_1.1.py --exp-id ensemble_5d_20d_v1 --use-smooth

命令行参数:
-----------
--exp-id, -e    实验ID，对应 03模型训练层/experiments/{exp_id} (默认: exp_001)
--use-smooth    使用平滑后的预测文件 smoothed_predictions.parquet

回测参数配置:
-------------
请在下方 STRATEGY_PARAMS 字典中修改回测参数
"""

import pandas as pd
import numpy as np
import gc
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from pathlib import Path
import base64
from io import BytesIO


# ==========================================
# 0. 回测参数配置（在此处修改参数）
# ==========================================
STRATEGY_PARAMS = {
    'stocks_per_batch': 20,           # 每次选股数量
    'start_date': datetime(2020, 1, 1),  # 回测开始日期
    'end_date': datetime(2026, 3, 31),   # 回测结束日期
    'initial_cash': 50000,           # 初始资金
    'commission': 0.002               # 手续费率 (0.2%)
}

# 全局变量：记录每日净值
cash_value_history = {}

# ==========================================
# 1. 配置参数
# ==========================================
import os
# 获取项目根目录（当前文件的上上级目录）
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_PATHS = {
    'open': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'open.parquet',
    'close': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'close.parquet',
    'high': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'high.parquet',
    'low': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'low.parquet',
    'volume': PROJECT_ROOT / '02因子库' / 'processed_data' / 'market_data' / 'volume.parquet',
}

def parse_args():
    parser = argparse.ArgumentParser(description='多因子回测脚本')
    parser.add_argument('--exp-id', '-e', type=str, default='exp_001',
                        help='实验ID，如 test_001_v1')
    parser.add_argument('--use-smooth', action='store_true',
                        help='使用平滑后的预测 (smoothed_predictions.parquet)')
    return parser.parse_args()

def get_paths(exp_id, use_smooth=False):
    exp_dir = PROJECT_ROOT / '03模型训练层' / 'experiments' / exp_id
    paths = DEFAULT_PATHS.copy()
    
    # 根据参数选择文件
    if use_smooth:
        paths['pred'] = str(exp_dir / 'smoothed_predictions.parquet')
        paths['live_pred'] = str(exp_dir / 'smoothed_live_predictions.parquet')
        paths['pred_col'] = 'pred_score_smooth'
    else:
        paths['pred'] = str(exp_dir / 'predictions.parquet')
        paths['live_pred'] = str(exp_dir / 'live_predictions.parquet')
        paths['pred_col'] = 'pred_score'
    
    return paths, exp_dir

# ==========================================
# 2. 数据处理模块
# ==========================================
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
    
    df_long = df_long[df_long['stock_code'].str.match(r'^\d{6}\.(SZ|SH|BJ)$', na=False)]
    df_long[value_name] = df_long[value_name].astype('float32')
    df_long = df_long.dropna(subset=[value_name])
    return df_long

def load_and_merge_data(paths):
    print("--- 步骤1: 加载预测数据 ---")
    pred_col = paths['pred_col']
    
    # 检查文件是否存在
    if not os.path.exists(paths['pred']):
        raise FileNotFoundError(f"找不到预测文件: {paths['pred']}")
    
    prediction = pd.read_parquet(paths['pred'], columns=['date', 'stock_code', pred_col])
    prediction = prediction.rename(columns={pred_col: 'pred_score'})
    
    # live文件可能不存在
    if os.path.exists(paths['live_pred']):
        live_prediction = pd.read_parquet(paths['live_pred'], columns=['date', 'stock_code', pred_col])
        live_prediction = live_prediction.rename(columns={pred_col: 'pred_score'})
    else:
        print(f"  警告: 找不到live预测文件: {paths['live_pred']}")
        live_prediction = pd.DataFrame(columns=['date', 'stock_code', 'pred_score'])
    
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
    
    main_df = main_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    print(f"完成数据合并，形状: {main_df.shape}")
    print(f"数据时间范围: {main_df.index.min()} ~ {main_df.index.max()}")
    return main_df

# ==========================================
# 3. 信号生成模块
# ==========================================
def generate_signals(df, top_n, start_date, end_date):
    """
    生成调仓信号（每月第一个交易日调仓）
    """
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
    
    # 找到每月第一个交易日（固定调仓日，避免幸运日问题）
    rebalance_dates = []
    current_month = None
    for date in trading_days:
        if date >= pd.to_datetime(start_date) and date <= pd.to_datetime(end_date):
            if date.month != current_month:
                rebalance_dates.append(date)
                current_month = date.month
    
    print(f"\n--- 步骤2: 生成调仓信号（每月首个交易日） ---")
    print(f"回测范围内总交易日: {len(trading_days)}, 计划调仓次数: {len(rebalance_dates)}")
    if rebalance_dates:
        print(f"首个调仓日: {rebalance_dates[0].strftime('%Y-%m-%d')}")
        print(f"末个调仓日: {rebalance_dates[-1].strftime('%Y-%m-%d')}")
    
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
        
        # 1. 只保留主板股票（60/00开头）
        current_slice = current_slice[
            current_slice['stock_code'].str.match(r'^(60|00)\d{4}\.(SH|SZ)$', na=False)
        ]
        
        if current_slice.empty:
            continue
        
        # 2. 获取T+1日数据，过滤开盘涨停的
        future_dates = df_period[df_period.index > date].index.unique()
        if len(future_dates) == 0:
            continue  # 没有T+1数据，跳过
        next_date = future_dates[0]
        
        try:
            # 获取T+1的数据
            next_day_df = df_period.loc[next_date]
            if isinstance(next_day_df, pd.Series):
                next_day_df = next_day_df.to_frame().T
            
            # 合并T日和T+1日的数据（按stock_code）
            merged = current_slice[['stock_code', 'close']].merge(
                next_day_df[['stock_code', 'open']], 
                on='stock_code', 
                suffixes=('_t', '_t1')
            )
            
            if len(merged) > 0:
                # 计算开盘涨幅
                merged['open_return'] = merged['open_t1'] / merged['close_t'] - 1
                
                # 过滤开盘涨停（>=9.9%）的
                tradable_stocks = merged.loc[merged['open_return'] < 0.099, 'stock_code']
                tradable = current_slice[current_slice['stock_code'].isin(tradable_stocks)]
                
                filtered_count = len(merged) - len(tradable_stocks)
                if filtered_count > 0:
                    print(f"  {date_str}: 过滤 {filtered_count} 只开盘涨停股")
            else:
                tradable = current_slice
        except:
            # 如果出错，就不过滤
            tradable = current_slice
            
        if tradable.empty:
            continue
            
        # 3. 在可操作股票中选top N
        selected = tradable.sort_values(by='prediction', ascending=False).head(top_n)
        buy_list = selected['stock_code'].tolist()
        buy_list_set = set(buy_list)
        
        sell_list = sorted(list(current_position - buy_list_set))
        
        buy_dict[date_str] = buy_list
        sell_dict[date_str] = sell_list
        current_position = buy_list_set
        all_held_stocks.update(current_position)
        
    return buy_dict, sell_dict, sorted(list(all_held_stocks))

# ==========================================
# 4. Backtrader 策略类
# ==========================================
class MyMultiFactorStrategy(bt.Strategy):
    params = (
        ('buy_date', None), 
        ('sell_date', None), 
        ('trades', None),
        ('stop_loss', 0.15),      # 止损比例：15%
    )

    def __init__(self):
        self.trade_count = 0
        self.prenext_count = 0

    def prenext(self):
        self.prenext_count += 1
        self.next()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def next(self):
        curr_dt = self.datetime.date(0).strftime('%Y-%m-%d')
        
        # 记录每日净值（使用全局变量）
        global cash_value_history
        cash_value_history[curr_dt] = self.broker.getvalue()
        
        # === 步骤1：每日检查止损（优先执行）===
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:                      # 有持仓
                cost_price = pos.price             # 平均成本价
                current_price = data.close[0]      # 当前收盘价
                
                # 跌超8%就止损卖出
                if current_price < cost_price * (1 - self.p.stop_loss):
                    self.order_target_percent(data=data, target=0)
                    print(f"🛑 止损(15%): {data._name} 成本{cost_price:.2f} 现价{current_price:.2f} (跌{((current_price/cost_price-1)*100):.1f}%)")
        
        # === 步骤2：调仓日逻辑 ===
        is_buy_day = curr_dt in self.p.buy_date
        is_sell_day = curr_dt in self.p.sell_date
        
        if not is_buy_day and not is_sell_day:
            return
        
        self.trade_count += 1
        action = "初始建仓" if self.trade_count == 1 else f"调仓 #{self.trade_count}"
        print(f"\n--- {action}: {curr_dt} ---")
        
        # 1. 先执行卖出
        if is_sell_day and self.p.sell_date[curr_dt]:
            s_list = self.p.sell_date[curr_dt]
            for s_code in s_list:
                try:
                    if s_code in self.getdatanames():
                        data = self.getdatabyname(s_code)
                        pos = self.getposition(data)
                        if pos.size > 0:
                            self.order_target_percent(data=data, target=0)
                except Exception as e:
                    continue

        # 2. 再执行买入
        if is_buy_day and self.p.buy_date[curr_dt]:
            b_list = self.p.buy_date[curr_dt]
            if len(b_list) > 0:
                valid_stocks = [s for s in b_list if s in self.getdatanames()]
                
                if valid_stocks:
                    target_per = 0.90 / len(valid_stocks)
                    
                    for b_code in valid_stocks:
                        try:
                            data = self.getdatabyname(b_code)
                            self.order_target_percent(data=data, target=target_per)
                        except Exception as e:
                            continue

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datetime.date(0).strftime('%Y-%m-%d')
            is_buy = order.isbuy()
            action = 'BUY' if is_buy else 'SELL'
            price = order.executed.price
            size = order.executed.size
            
            self.p.trades.append({
                'date': dt,
                'stock_code': order.data._name,
                'action': action,
                'price': round(price, 2),
                'shares': abs(size)
            })

# ==========================================
# 5. 回测引擎模块
# ==========================================
def run_backtest(full_df, buy_date, sell_date, stock_list, strategy_params, trades_list):
    global cash_value_history
    cash_value_history = {}  # 清空历史
    
    cerebro = bt.Cerebro(runonce=False)
    
    print("\n--- 步骤3: 加载 Backtrader 数据源 ---")
    total_stocks = len(stock_list)
    valid_count = 0
    
    for i, stock_code in enumerate(stock_list):
        data_slice = full_df[full_df['stock_code'] == stock_code][['open', 'high', 'low', 'close', 'volume', 'openinterest']]
        
        if data_slice.empty:
            continue
        
        data_slice = data_slice[(data_slice.index >= strategy_params['start_date']) & 
                                (data_slice.index <= strategy_params['end_date'])]
        
        if len(data_slice) < 20:
            continue
            
        data = bt.feeds.PandasData(
            dataname=data_slice,
            fromdate=strategy_params['start_date'],
            todate=strategy_params['end_date']
        )
        cerebro.adddata(data, name=stock_code)
        valid_count += 1
        
        if (i+1) % 100 == 0 or (i+1) == total_stocks:
            print(f"加载进度: {i+1}/{total_stocks} (成功: {valid_count})")

    print(f"\n成功加载 {valid_count} 只股票")
    
    if valid_count == 0:
        print("错误：没有成功加载任何股票数据！")
        return None, None

    cerebro.addstrategy(MyMultiFactorStrategy, buy_date=buy_date, sell_date=sell_date, trades=trades_list)
    
    cerebro.broker.setcash(strategy_params['initial_cash'])
    cerebro.broker.setcommission(commission=strategy_params['commission'])
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='Sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='Drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='Returns', tann=252)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')

    print("\n--- 步骤4: 启动回测引擎 ---")
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    
    results = cerebro.run(runonce=False)
    
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    print(f"\n" + "="*50)
    print(f"最终净值: {final_value:.2f}")
    
    total_return = (final_value / strategy_params['initial_cash'] - 1) * 100
    print(f"总收益率: {total_return:.2f}%")
    
    metrics = {}
    try:
        # 通过 _name 获取分析器
        ret_analysis = strat.analyzers.getbyname('Returns').get_analysis()
        sharpe_analysis = strat.analyzers.getbyname('Sharpe').get_analysis()
        dd_analysis = strat.analyzers.getbyname('Drawdown').get_analysis()
        trade_analysis = strat.analyzers.getbyname('TradeAnalyzer').get_analysis()

        metrics['ann_return'] = ret_analysis.get('rnorm100', 0) if ret_analysis else 0
        metrics['sharpe'] = sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0
        
        # 安全获取最大回撤
        max_dd = 0
        if dd_analysis:
            max_dd = dd_analysis.get('max', {}).get('drawdown', 0) if isinstance(dd_analysis, dict) else 0
        metrics['max_dd'] = max_dd
        
        # 安全获取交易统计
        total_trades = 0
        win_rate = 0
        avg_win = 0      # 平均盈利
        avg_loss = 0     # 平均亏损
        profit_factor = 0  # 盈亏比
        
        if trade_analysis and isinstance(trade_analysis, dict):
            total = trade_analysis.get('total', {}).get('total', 0)
            won = trade_analysis.get('won', {}).get('total', 0)
            total_trades = total
            win_rate = (won / total * 100) if total > 0 else 0
            
            # 获取平均盈利和平均亏损
            try:
                won_analysis = trade_analysis.get('won', {})
                lost_analysis = trade_analysis.get('lost', {})
                
                if won_analysis and 'pnl' in won_analysis:
                    avg_win = won_analysis['pnl'].get('average', 0)
                if lost_analysis and 'pnl' in lost_analysis:
                    avg_loss = abs(lost_analysis['pnl'].get('average', 0))  # 转为正数
                    
                # 计算盈亏比
                if avg_loss > 0:
                    profit_factor = avg_win / avg_loss
                elif avg_win > 0:
                    profit_factor = float('inf')  # 只盈利无亏损
            except:
                pass
        
        metrics['total_trades'] = total_trades
        metrics['win_rate'] = win_rate
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['profit_factor'] = profit_factor

        print(f"年化收益率: {metrics['ann_return']:.2f}%")
        print(f"夏普比率: {metrics['sharpe']:.2f}" if metrics['sharpe'] else "夏普比率: N/A")
        print(f"最大回撤: {metrics['max_dd']:.2f}%")
        print(f"胜率: {win_rate:.1f}% | 盈亏比: {profit_factor:.2f} (平均盈利{avg_win:.0f}/平均亏损{avg_loss:.0f})")
    except Exception as e:
        print(f"获取分析指标时出错: {e}")
        
    print("="*50)
    
    # 转换净值历史为 Series
    equity_df = pd.Series(cash_value_history)
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_df = equity_df.sort_index()
    
    return cerebro, metrics, equity_df

# ==========================================
# 6. HTML报告生成
# ==========================================
def generate_html_report(exp_id, metrics, equity_df, output_dir):
    """生成HTML报告"""
    
    # 将图片转为base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
    
    # 绘制净值曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    equity_df.plot(ax=ax, title=f'Strategy Equity Curve - {exp_id}')
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=STRATEGY_PARAMS['initial_cash'], color='r', linestyle='--', alpha=0.5, label='Initial')
    ax.legend()
    
    img_base64 = fig_to_base64(fig)
    plt.close()
    
    total_return = (equity_df.iloc[-1] / STRATEGY_PARAMS['initial_cash'] - 1) * 100
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Backtest Report - {exp_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                      padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }}
        .metric-box {{ text-align: center; padding: 20px; background: #f8f9fa; 
                      border-radius: 8px; min-width: 150px; margin: 10px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #FF9800; }}
        .bad {{ color: #f44336; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 回测报告 - {exp_id}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 关键指标</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-value {'good' if total_return > 0 else 'bad'}">{total_return:.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if metrics.get('ann_return', 0) > 0 else 'bad'}">{metrics.get('ann_return', 0):.2f}%</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'good' if metrics.get('sharpe', 0) > 1 else 'warning' if metrics.get('sharpe', 0) > 0.5 else 'bad'}">{metrics.get('sharpe', 0):.2f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {'bad' if metrics.get('max_dd', 0) > 20 else 'warning' if metrics.get('max_dd', 0) > 10 else 'good'}">{metrics.get('max_dd', 0):.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{int(metrics.get('total_trades', 0))}</div>
                <div class="metric-label">总交易次数</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                <div class="metric-label">胜率</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">盈亏比</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('avg_win', 0):.0f}</div>
                <div class="metric-label">平均盈利</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics.get('avg_loss', 0):.0f}</div>
                <div class="metric-label">平均亏损</div>
            </div>
        </div>
        
        <div class="summary">
            <h3>📋 回测设置</h3>
            <ul>
                <li><strong>模型:</strong> {exp_id}</li>
                <li><strong>回测区间:</strong> {equity_df.index[0].strftime('%Y-%m-%d')} ~ {equity_df.index[-1].strftime('%Y-%m-%d')}</li>
                <li><strong>调仓周期:</strong> 每月第一个交易日</li>
                <li><strong>持仓数量:</strong> {STRATEGY_PARAMS['stocks_per_batch']}只</li>
                <li><strong>初始资金:</strong> {STRATEGY_PARAMS['initial_cash']:,}</li>
            </ul>
        </div>
        
        <h2>📈 净值曲线</h2>
        <img src="data:image/png;base64,{img_base64}" alt="Equity Curve">
        
        <p style="color: #999; font-size: 12px; margin-top: 30px;">
            Generated by Backtrader | Data: {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
</body>
</html>'''
    
    html_path = output_dir / 'backtest_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已保存: {html_path}")

# ==========================================
# 7. 执行入口
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    exp_id = args.exp_id
    
    print("="*60)
    print(f"多因子回测 - {exp_id}")
    print("="*60)
    
    # 获取路径
    PATHS, exp_dir = get_paths(exp_id, args.use_smooth)
    

    
    # 创建输出目录
    output_dir = PROJECT_ROOT / '04回测层' / 'reports' / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    master_df = load_and_merge_data(PATHS)
    
    # 诊断信息
    print(f"\n[诊断] 主表日期范围: {master_df.index.min()} ~ {master_df.index.max()}")

    # 生成信号（每月第一个交易日调仓）
    buy_date, sell_date, stock_list = generate_signals(
        master_df,
        STRATEGY_PARAMS['stocks_per_batch'],
        STRATEGY_PARAMS['start_date'],
        STRATEGY_PARAMS['end_date']
    )
    
    print(f"涉及股票总数: {len(stock_list)}")
    
    if not buy_date:
        print("错误：没有生成任何调仓信号！")
    else:
        # 交易记录列表
        trades_list = []
        
        # 运行回测
        cerebro, metrics, equity_df = run_backtest(master_df, buy_date, sell_date, stock_list, 
                                                    STRATEGY_PARAMS, trades_list)
        
        if cerebro and metrics and equity_df is not None:
            # 1. 保存交易记录
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                trades_path = output_dir / 'trades.csv'
                trades_df.to_csv(trades_path, index=False)
                print(f"交易记录已保存: {trades_path} ({len(trades_df)} 笔)")
            
            # 2. 保存净值曲线图
            fig, ax = plt.subplots(figsize=(12, 6))
            equity_df.plot(ax=ax, title=f'Strategy Equity Curve - {exp_id}')
            ax.set_ylabel('Portfolio Value')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=STRATEGY_PARAMS['initial_cash'], color='r', linestyle='--', alpha=0.5, label='Initial')
            ax.legend()
            
            png_path = output_dir / 'equity_curve.png'
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"净值曲线已保存: {png_path}")
            
            # 打印最终收益（和HTML一致）
            final_return = (equity_df.iloc[-1] / STRATEGY_PARAMS['initial_cash'] - 1) * 100
            print(f"\n最终收益: {equity_df.iloc[-1]:.2f} (收益率: {final_return:.2f}%)")
            
            # 3. 生成HTML报告
            generate_html_report(exp_id, metrics, equity_df, output_dir)
            
        print(f"\n所有输出文件保存在: {output_dir}")
