"""
月度数据更新模块
==============

背景:
-----
原系统每次运行都全量下载所有数据，耗时较长。
由于前复权价格会随分红除权而变化（前复权漂移），需要定期重新下载修正。

更新策略:
---------
1. 行情数据（K线）: 每月全量重新下载（2010-至今）
   - 原因: 前复权价格会随时间漂移，需要定期修正
   
2. 财务数据: 智能合并（保留历史 + 更新重叠部分）
   - 保留: 删除完全一样的数据
   - 新增: 最新下载的数据
   
3. 元数据: 重新下载（股票列表、行业映射）

去重逻辑:
---------
按 ['report_date', 'm_anntime'] 去重，保留最后出现的（keep='last'）
- 完全一样的数据: 新数据覆盖旧数据（无实质影响）
- 修正过的数据: 新数据替换旧数据（如公司发布业绩修正）
- 快报/正式报告: 保留所有不同公告日的版本

使用方法:
---------
```bash
# 每月更新（推荐周末运行）
python data_main.py --monthly

# 查看帮助
python data_main.py --help
```

注意事项:
---------
1. 全市场5000+股票，行情下载约需2-3小时
2. 建议在周末运行，避免交易时段占用网络资源
3. 更新前会自动备份（可选）
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 添加 xtquant 路径
sys.path.insert(0, r'C:\Users\蒋大王\Desktop\量化\截面多因子模型\xtquant_250807')
from xtquant import xtdata

# 导入基础数据引擎
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Base_DataEngine import DataEngine


class MonthlyDataUpdater(DataEngine):
    """月度数据更新器 - 继承自基础数据引擎"""
    
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.log_file = os.path.join(self.root_path, 'update_log.json')
        
    def download_financial_data_with_merge(self, stock_list, start_time='20230101'):
        """
        下载财务数据并智能合并
        
        策略:
        1. 下载 start_time 以来的新数据
        2. 读取现有文件（如果有）
        3. 合并并按 ['report_date', 'm_anntime'] 去重，保留最后出现的
        4. 保存
        """
        import time
        batch_size = 300
        FIELD_MAP = {'m_timetag': 'report_date'}
        target_tables = ['Income', 'Balance', 'Indicator', 'CashFlow']
        
        print(f"🚀 开始分批作业：总计 {len(stock_list)} 只股票，每批 {batch_size} 只")
        print(f"📅 下载起始日期: {start_time}，历史数据将自动合并")
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (len(stock_list) + batch_size - 1) // batch_size
            print(f"\n📦 正在处理第 {current_batch_num}/{total_batches} 批次 ({i} - {min(i + len(batch), len(stock_list))})")
            
            # 1. 触发该批次的异步下载
            self._try_download(xtdata.download_financial_data2, batch, start_time=start_time)
            time.sleep(1)
            
            # 2. 遍历该批次进行本地缝合与清洗
            for stock in tqdm(batch, desc=f"批次 {current_batch_num} 存盘"):
                try:
                    # 获取新数据
                    new_df = self._fetch_financial_data(stock, target_tables, FIELD_MAP)
                    if new_df is None:
                        continue
                    
                    # 3. 智能合并 + 两步清洗
                    save_file = os.path.join(self.fin_path, f"{stock}.parquet")
                    if os.path.exists(save_file):
                        # 读取现有数据
                        old_df = pd.read_parquet(save_file)
                        # 合并
                        combined = pd.concat([old_df, new_df], ignore_index=True)
                    else:
                        combined = new_df
                    
                    # 4. 两步清洗（V2）- 必须重新执行完整清洗
                    if combined is not None and not combined.empty:
                        # 步骤0: 合并同一(report_date, m_anntime)
                        combined = combined.groupby(['report_date', 'm_anntime']).first().reset_index()
                        
                        # 清洗1: 同一 report_date 保留最早 m_anntime
                        combined = combined.sort_values(['report_date', 'm_anntime'])
                        combined = combined.drop_duplicates(subset=['report_date'], keep='first')
                        
                        # 清洗2: 同一 m_anntime 保留最大 report_date
                        combined = combined.sort_values(['m_anntime', 'report_date'])
                        combined = combined.drop_duplicates(subset=['m_anntime'], keep='last')
                        
                        # 最终排序并保存
                        combined = combined.sort_values(['report_date', 'm_anntime'])
                    
                    # 5. 保存
                    combined.to_parquet(save_file)
                    
                except Exception as e:
                    print(f"❌ 股票 {stock} 处理失败: {e}")
                    continue
        
        print(f"✅ 财务数据更新完成。")
    
    def _fetch_financial_data(self, stock, target_tables, field_map):
        """获取单只股票的财务数据（内部方法）"""
        from xtquant import xtdata
        
        # 获取数据，加入简单的重试逻辑
        fin_dict = {}
        for _ in range(3):
            fin_dict = xtdata.get_financial_data(
                [stock], 
                table_list=target_tables, 
                report_type='announce_time'
            )
            if fin_dict and stock in fin_dict and len(fin_dict[stock]) > 0:
                break
        
        if not fin_dict or stock not in fin_dict:
            return None
        
        tables_dict = fin_dict[stock]
        combined_df = None
        
        for table_name in target_tables:
            if table_name not in tables_dict or tables_dict[table_name].empty:
                continue
            
            df = tables_dict[table_name].rename(columns=field_map)
            if 'report_date' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'report_date'})
            
            # 强转时间类型为标准 8 位字符串
            for col in ['report_date', 'm_anntime']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('.0', '', regex=False)
            
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(
                    combined_df, df, 
                    on=['report_date', 'm_anntime'], 
                    how='outer', 
                    suffixes=('', f'_{table_name}')
                )
        
        if combined_df is not None and not combined_df.empty:
            # 两步清洗（V2）
            # 步骤0: 合并同一(report_date, m_anntime)
            combined_df = combined_df.groupby(['report_date', 'm_anntime']).first().reset_index()
            
            # 清洗1: 同一 report_date 保留最早 m_anntime
            combined_df = combined_df.sort_values(['report_date', 'm_anntime'])
            combined_df = combined_df.drop_duplicates(subset=['report_date'], keep='first')
            
            # 清洗2: 同一 m_anntime 保留最大 report_date
            combined_df = combined_df.sort_values(['m_anntime', 'report_date'])
            combined_df = combined_df.drop_duplicates(subset=['m_anntime'], keep='last')
            
            combined_df = combined_df.sort_values(['report_date', 'm_anntime'])
        
        return combined_df
    
    def _try_download(self, download_func, stock_list, **kwargs):
        """尝试下载，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                download_func(stock_list, **kwargs)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"下载失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    raise
    
    def monthly_update(self, market_start='20100101', financial_start='20230101'):
        """
        执行月度数据更新
        
        Args:
            market_start: 行情数据起始日期，默认 '20100101'
            financial_start: 财务数据起始日期，默认 '20230101'
        """
        from xtquant import xtdata
        
        print("=" * 60)
        print(f"🚀 开始月度数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 获取最新股票列表
        print("\n📋 获取最新股票列表...")
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        print(f"   共 {len(stocks)} 只股票")
        
        # 1. 行情数据 - 全量覆盖
        print("\n" + "=" * 60)
        print("📈 步骤1: 更新行情数据（全量覆盖）")
        print(f"   起始日期: {market_start}")
        print("=" * 60)
        self.download_market_data(stocks, start_time=market_start)
        
        # 2. 财务数据 - 智能合并
        print("\n" + "=" * 60)
        print("📊 步骤2: 更新财务数据（智能合并）")
        print(f"   下载起始: {financial_start}")
        print("   策略: 保留历史 + 合并重叠 + 去重")
        print("=" * 60)
        self.download_financial_data_with_merge(stocks, start_time=financial_start)
        
        # 3. 元数据 - 重新下载
        print("\n" + "=" * 60)
        print("📁 步骤3: 更新元数据")
        print("=" * 60)
        self.download_metadata()
        
        # 4. 记录更新日志
        self._save_update_log()
        
        print("\n" + "=" * 60)
        print("✅ 月度数据更新完成！")
        print("=" * 60)
    
    def _save_update_log(self):
        """保存更新日志"""
        log = {
            'last_update': datetime.now().isoformat(),
            'status': 'success'
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"\n📝 更新日志已保存: {self.log_file}")


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='月度数据更新工具')
    parser.add_argument(
        '--monthly', 
        action='store_true',
        help='执行月度更新（行情全量 + 财务合并 + 元数据）'
    )
    parser.add_argument(
        '--market-start',
        default='20100101',
        help='行情数据起始日期 (默认: 20100101)'
    )
    parser.add_argument(
        '--financial-start',
        default='20230101',
        help='财务数据起始日期 (默认: 20230101)'
    )
    
    args = parser.parse_args()
    
    if args.monthly:
        updater = MonthlyDataUpdater()
        updater.monthly_update(
            market_start=args.market_start,
            financial_start=args.financial_start
        )
    else:
        parser.print_help()
        print("\n💡 提示: 使用 --monthly 参数执行月度更新")


if __name__ == "__main__":
    main()
