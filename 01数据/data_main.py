import sys
import os
import argparse

# 添加 xtquant 路径
sys.path.insert(0, r'C:\Users\蒋大王\Desktop\量化\截面多因子模型\xtquant_250807')

from xtquant import xtdata
from tqdm import tqdm
from Base_DataEngine import DataEngine
from monthly_update import MonthlyDataUpdater


def full_download(engine, market_end_date=''):
    """全量下载（原始方式）"""
    print("🚀 执行全量下载模式...")
    stocks = xtdata.get_stock_list_in_sector('沪深A股')
    
    # 下载行情
    print("\n📈 下载行情数据...")
    engine.download_market_data(stocks, end_time=market_end_date)
    
    # 下载财务
    print("\n📊 下载财务数据...")
    engine.download_financial_data(stocks)
    
    # 下载元数据
    print("\n📁 下载元数据...")
    engine.download_metadata()
    
    print("\n✅ 全量下载完成！")


def monthly_update():
    """月度更新（智能合并模式）"""
    updater = MonthlyDataUpdater()
    updater.monthly_update(
        market_start='20100101',
        financial_start='20230101'
    )


def main():
    parser = argparse.ArgumentParser(
        description='数据下载工具 - 支持全量下载和月度增量更新',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 全量下载（首次使用或需要完整刷新）
  python data_main.py --full
  
  # 全量下载但只到指定日期（避免盘中获取未收盘数据）
  python data_main.py --full --end-date 20260318
  
  # 月度更新（推荐每月周末运行）
  python data_main.py --monthly
  
  # 自定义财务数据起始日期
  python data_main.py --monthly --financial-start 20240101

模式说明:
  --full:   全量覆盖所有数据（行情从2010年，财务从2010年）
  --monthly: 智能更新（行情全量覆盖，财务合并2023年以来的数据）
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='全量下载模式（覆盖所有历史数据）'
    )
    parser.add_argument(
        '--monthly',
        action='store_true',
        help='月度更新模式（行情全量 + 财务智能合并 + 元数据）'
    )
    parser.add_argument(
        '--market-start',
        default='20100101',
        help='行情数据起始日期 (默认: 20100101)'
    )
    parser.add_argument(
        '--financial-start',
        default='20230101',
        help='财务数据下载起始日期，用于月度更新 (默认: 20230101)'
    )
    parser.add_argument(
        '--end-date',
        default='',
        help='行情数据结束日期，格式 YYYYMMDD，默认空表示最新。建议盘中下载时指定昨天日期 (如: 20260318)'
    )
    
    args = parser.parse_args()
    
    if args.monthly:
        # 月度更新模式
        monthly_update()
    elif args.full:
        # 全量下载模式
        engine = DataEngine()
        full_download(engine, market_end_date=args.end_date)
    else:
        parser.print_help()
        print("\n💡 请选择运行模式:")
        print("   python data_main.py --full    # 全量下载")
        print("   python data_main.py --monthly # 月度更新")


if __name__ == "__main__":
    main()
