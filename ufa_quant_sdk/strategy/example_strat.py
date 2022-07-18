from datetime import datetime, timedelta, timezone

from ufa_quant_sdk.apis.finance_data import get_kline
from ufa_quant_sdk.apis.trade import make_order
from ufa_quant_sdk.config import STRATEGY_NAME
from ufa_quant_sdk.run_strategy import AccountContext
from ufa_quant_sdk.utils import abspath
from ufa_quant_sdk.utils.logger_tools import get_general_logger
from pprint import pprint

logger = get_general_logger(STRATEGY_NAME, path=abspath("logs"))


def main(context: AccountContext):
    # 股票代码
    symbol = "SZ.000100"

    # 获取近5日K线数据
    kline_end = datetime.now(timezone(timedelta(hours=8)))
    kline_start = kline_end - timedelta(days=6) + timedelta(seconds=1)
    kline = get_kline(
        symbol,
        kline_start.strftime("%Y-%m-%d %H:%M:%S"),
        kline_end.strftime("%Y-%m-%d %H:%M:%S"),
        "1d",  # 天级
    )
    # pprint(kline)
    if len(kline) == 0:
        logger.warning(f"未查询到K线信息，请检查股票代码({symbol})或时间是否正确。若无误，请联系管理员。")
        return

    # 获取当前持仓
    pos_amount = sum(
        pos["amount"]
        for pos in filter(
            lambda pos: symbol == pos["symbol"], context.positions["avaliable"]
        )
    )
    logger.info(f"当前持仓：{pos_amount}")

    # 计算近5日均值
    ma_close = sum([info["close"] / len(kline) for info in kline])
    latest_close = kline[-1]["close"]
    diff_pct = round((latest_close - ma_close) * 100 / ma_close, 2)
    logger.info(f"平均: {ma_close}, 最新: {latest_close}, 差值: {diff_pct}%")

    # 样例策略
    max_pos_amount = 300  # 最大持仓量

    # 买入
    buy_amount = 100  # 买入量（须为100的倍数）
    if (
        context.cash_avaliable > latest_close * buy_amount  # 现金充足
        and pos_amount + buy_amount <= max_pos_amount  # 交易后不超过最大持仓量
        and latest_close >= ma_close * 1.01  # 现价超过均价至少1%
    ):
        make_order(symbol, "market", "buy", buy_amount)
        logger.info(f"买入策略已执行")
        return

    # 卖出
    if pos_amount > max_pos_amount + 100:  # 实际持仓量大于最大持仓量100股
        sell_amount = (pos_amount - max_pos_amount) // 100 * 100  # 卖出超过阈值的持仓量（须为100的倍数）
        make_order(symbol, "market", "sell", sell_amount)
        logger.info(f"卖出策略已执行")
        return
