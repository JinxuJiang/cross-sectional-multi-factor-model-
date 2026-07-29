# -*- coding: utf-8 -*-
"""
Tushare 月度数据更新模块
========================

与旧 QMT 的 monthly_update.py 对应，继承 TushareDataEngine 增加增量策略。

更新策略:
---------
1. 元数据: 全量重抓（股票列表、交易日历会变化，成本低）
2. 行情: 只抓缺失交易日 + 全量重建 per-stock 文件
   - 原因: 等比前复权因子随分红除权漂移，历史价格需整体修正
3. 财务: 重抓最近 8 个季度分区（--overwrite 原子替换）
   - 原因: 年报披露可拖至次年 4 月底，且公司可能发布跨年业绩修正
4. 状态: 只抓缺失交易日 + 重建宽表
5. 总验证: 行情覆盖 / 财务分区 / 状态宽表 / 元数据完整性
6. 更新日志

使用方法:
---------
```bash
python tushare_data_main.py --monthly
# 或直接
python tushare_monthly_update.py
```
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Base_TushareEngine import TushareDataEngine


class MonthlyTushareUpdater(TushareDataEngine):
    """Tushare 月度数据更新器 - 继承自基础数据引擎"""

    @property
    def log_file(self) -> Path:
        return self.root_path / "update_log.json"

    def monthly_update(self, end_date: str | None = None,
                       financial_lookback_quarters: int = 8):
        """
        执行月度数据更新

        参数:
        -----
        end_date : 结束日期 'YYYYMMDD'，默认今天。盘中运行建议指定昨天
        financial_lookback_quarters : 财务重抓最近几个季度分区，默认 8
            （覆盖年报延迟披露与跨年业绩修正）
        """
        end = end_date or datetime.now().strftime("%Y%m%d")
        # 只覆盖已到来的季度报告期，避免抓取未来季度返回空数据
        end_period = self.latest_quarter_period(end)

        print("=" * 60)
        print(f"🚀 开始 Tushare 月度数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. 元数据 - 全量重抓（同时刷新交易日历，是后续步骤的前提）
        print("\n" + "=" * 60)
        print("📁 步骤1: 更新元数据（全量重抓）")
        print("=" * 60)
        self.download_metadata(end_date=end)

        # 2. 行情 - 缺失补齐 + 全量重建（前复权漂移）
        print("\n" + "=" * 60)
        print("📈 步骤2: 更新行情数据（缺失补齐 + 全量重建）")
        print("=" * 60)
        self.download_market_data(start_date="20100101", end_date=end,
                                  missing_only=True, build=True)

        # 3. 财务 - 重抓最近 N 个季度分区（覆盖披露季补全与业绩修正）
        print("\n" + "=" * 60)
        print(f"📊 步骤3: 更新财务数据（重抓最近 {financial_lookback_quarters} 个季度）")
        print("=" * 60)
        all_periods = self.quarter_periods("20100331", end)
        recent_periods = all_periods[-financial_lookback_quarters:]
        if recent_periods:
            self.download_financial_data(start_period=recent_periods[0],
                                         end_period=recent_periods[-1],
                                         overwrite=True)

        # 4. 状态 - 缺失补齐 + 重建宽表
        print("\n" + "=" * 60)
        print("🚨 步骤4: 更新状态数据（缺失补齐 + 重建宽表）")
        print("=" * 60)
        self.download_status_data(start_date="20100101", end_date=end,
                                  missing_only=True, build=True)

        # 5. 总验证 - 行情覆盖 / 财务分区 / 状态宽表 / 元数据
        print("\n" + "=" * 60)
        print("✔️ 步骤5: 总验证")
        print("=" * 60)
        report = self.validate_all(end_date=end)
        n_fail = report["summary"]["fail"]
        n_warn = report["summary"]["warn"]

        # 6. 记录更新日志
        self._save_update_log(n_fail=n_fail, n_warn=n_warn)

        print("\n" + "=" * 60)
        if n_fail:
            print(f"⚠️ 月度更新完成，但验证有 {n_fail} 项 FAIL，请检查报告！")
            raise SystemExit(1)
        print("✅ 月度数据更新完成！")
        print("=" * 60)

    def _save_update_log(self, n_fail: int, n_warn: int):
        log = {
            "last_update": datetime.now().isoformat(),
            "status": "success" if n_fail == 0 else "validation_failed",
            "validation": {"fail": n_fail, "warn": n_warn},
        }
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"\n📝 更新日志已保存: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(description="Tushare 月度数据更新工具")
    parser.add_argument("--end-date", default="",
                        help="结束日期 YYYYMMDD，默认今天。盘中运行建议指定昨天")
    parser.add_argument("--financial-lookback-quarters", type=int, default=8,
                        help="财务重抓最近几个季度分区 (默认: 8)")
    args = parser.parse_args()

    updater = MonthlyTushareUpdater()
    updater.monthly_update(
        end_date=args.end_date or None,
        financial_lookback_quarters=args.financial_lookback_quarters,
    )


if __name__ == "__main__":
    main()
