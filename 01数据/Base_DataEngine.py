import os        
import pandas as pd 
from pathlib import Path
from xtquant import xtdata
from tqdm import tqdm

class DataEngine:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).parent / 'data' / 'raw_data'
        self.root_path = Path(data_path)
        self.price_path = os.path.join(data_path, 'market_data')
        self.fin_path = os.path.join(data_path, 'financial_data')
        
        os.makedirs(self.price_path, exist_ok=True)
        os.makedirs(self.fin_path, exist_ok=True)

    def download_market_data(self, stock_list, start_time='20100101', end_time=''):
        """下载行情数据并保存 (强制前复权)
        
        参数:
        -----
        stock_list : list
            股票列表
        start_time : str
            开始日期，格式 'YYYYMMDD'，默认 '20100101'
        end_time : str
            结束日期，格式 'YYYYMMDD'，默认 '' (表示最新)
            建议：盘中下载时指定昨天日期，避免获取到未收盘的数据
        """
        print(f"开始下载行情数据，共{len(stock_list)}只股票...")
        if end_time:
            print(f"日期范围: {start_time} ~ {end_time}")
        else:
            print(f"日期范围: {start_time} ~ 最新")
        
        xtdata.download_history_data2(stock_list, period='1d', start_time=start_time, end_time=end_time)
        
        for stock in tqdm(stock_list, desc="保存行情"):
            # 关键修改：dividend_type='front'
            data = xtdata.get_market_data_ex([], [stock], period='1d', start_time=start_time, end_time=end_time, dividend_type='front_ratio')
            if stock in data:
                df = data[stock]
                if not df.empty:
                    df.to_parquet(os.path.join(self.price_path, f"{stock}.parquet"))

    def download_financial_data(self, stock_list, start_time='20100101', end_time=''):
        """
        下载财务数据：分批机制 + 自动重试 + PIT 缝合
        
        参数:
        -----
        stock_list : list
            股票列表
        start_time : str
            开始日期，格式 'YYYYMMDD'，默认 '20100101'
        end_time : str
            结束日期，格式 'YYYYMMDD'，默认 '' (表示最新)
        """
        import time
        batch_size = 300 # 按照你的要求，每批处理 300 只
        
        FIELD_MAP = {'m_timetag': 'report_date'}
        target_tables = ['Income', 'Balance', 'CashFlow', 'PershareIndex']
        # 注意：QMT中没有Indicator表，财务指标数据在PershareIndex中提供

        print(f"[启动] 开始分批作业：总计 {len(stock_list)} 只股票，每批 {batch_size} 只")

        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            print(f"\n[批次] 正在处理第 {current_batch_num} 批次 ({i} - {i + len(batch)})")

            # 1. 触发该批次的异步下载
            xtdata.download_financial_data2(batch, start_time=start_time, end_time=end_time)
            
            # 2. 尊重同步时间。给 QMT 1 秒时间将数据从服务器写入本地缓存
            time.sleep(1) 

            # 3. 遍历该批次进行本地缝合与清洗
            for stock in tqdm(batch, desc=f"批次 {current_batch_num} 存盘"):
                try:
                    # 获取数据，加入简单的重试逻辑防止 QMT 还没写完
                    fin_dict = {}
                    for _ in range(3): # 最多试 3 次
                        fin_dict = xtdata.get_financial_data([stock], table_list=target_tables, report_type='announce_time')
                        if fin_dict and stock in fin_dict and len(fin_dict[stock]) > 0:
                            break

                    if not fin_dict or stock not in fin_dict:
                        continue
                    
                    tables_dict = fin_dict[stock]
                    combined_df = None

                    for table_name in target_tables:
                        if table_name not in tables_dict or tables_dict[table_name].empty:
                            continue
                        
                        df = tables_dict[table_name].rename(columns=FIELD_MAP)
                        if 'report_date' not in df.columns:
                            df = df.reset_index().rename(columns={'index': 'report_date'})
                        
                        # 强转时间类型为标准 8 位字符串
                        for col in ['report_date', 'm_anntime']:
                            if col in df.columns:
                                df[col] = df[col].astype(str).str.replace('.0', '', regex=False)

                        if combined_df is None:
                            combined_df = df
                        else:
                            combined_df = pd.merge(combined_df, df, on=['report_date', 'm_anntime'], 
                                                how='outer', suffixes=('', f'_{table_name}'))

                    if combined_df is not None and not combined_df.empty:
                        # --- 两步清洗（V2）---
                        # 步骤0: 合并同一(report_date, m_anntime)的多行（四表外连接后去重）
                        combined_df = combined_df.groupby(['report_date', 'm_anntime']).first().reset_index()
                        
                        # 清洗1: 同一 report_date 保留最早 m_anntime（防未来函数）
                        combined_df = combined_df.sort_values(['report_date', 'm_anntime'])
                        combined_df = combined_df.drop_duplicates(subset=['report_date'], keep='first')
                        
                        # 清洗2: 同一 m_anntime 保留最大 report_date（处理同天多报告）
                        combined_df = combined_df.sort_values(['m_anntime', 'report_date'])
                        combined_df = combined_df.drop_duplicates(subset=['m_anntime'], keep='last')
                        
                        # 最终排序并保存
                        combined_df = combined_df.sort_values(['report_date', 'm_anntime'])
                        save_file = os.path.join(self.fin_path, f"{stock}.parquet")
                        combined_df.to_parquet(save_file)
                        
                except Exception as e:
                    print(f"[失败] 股票 {stock} 处理失败: {e}")
                    continue

        print(f"[完成] 全市场财务数据分批同步完成。")
    def get_all_industry_map(self):
        """构建股票->行业映射 - 申万一级"""
        all_sectors = xtdata.get_sector_list()
        # 1. 锁定 SW1 且排除掉“加权”和“等权”干扰项
        # 逻辑：以 SW1 开头，且名字里不包含“加权”或“等权”
        sw_l1_sectors = [s for s in all_sectors if s.startswith('SW1') 
                         and '加权' not in s and '等权' not in s]
        print(f"[发现] 发现 {len(sw_l1_sectors)} 个有效的申万一级行业索引")
        
        industry_mapping = []
        for sector in tqdm(sw_l1_sectors, desc="抓取行业成员"):
            # 获取该行业下的所有股票代码
            stocks = xtdata.get_stock_list_in_sector(sector)
            # 提取纯净的行业名称，比如 'SW1银行' -> '银行'
            clean_name = sector.replace('SW1', '')
            for stock in stocks:
                industry_mapping.append({
                    'order_book_id': stock,
                    'industry_name': clean_name
                })
        
        df_ind = pd.DataFrame(industry_mapping)
        # 2. 去重校验 万一一只股票横跨两个行业（虽然申万一级理论上不会），保留最后一个
        df_ind = df_ind.drop_duplicates(subset=['order_book_id'], keep='last')
        return df_ind

    def get_all_stock_details(self, stock_list):
        """获取股票静态详情"""
        details = []
        for stock in tqdm(stock_list, desc="爬取详情"):
            info = xtdata.get_instrument_detail(stock)
            if info:
                details.append({
                    'order_book_id': stock,
                    'symbol': info['InstrumentName'],
                    'list_date': info['OpenDate'],
                    'exchange': info['ExchangeID']
                })
        return pd.DataFrame(details)
        
    def download_metadata(self):
        """元数据一键搬运"""
        all_a_stocks = xtdata.get_stock_list_in_sector('沪深A股')
        
        # 保存股票详情
        df_details = self.get_all_stock_details(all_a_stocks)
        df_details.to_parquet(os.path.join(self.root_path, 'stock_info.parquet'))
        
        # 保存行业映射
        df_industry = self.get_all_industry_map()
        df_industry.to_csv(os.path.join(self.root_path, 'industry_map.csv'), index=False)
        
        print("[完成] 元数据搬运完成！")