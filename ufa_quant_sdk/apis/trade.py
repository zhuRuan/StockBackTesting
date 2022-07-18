from ufa_quant_sdk.apis import _post_request_data
from ufa_quant_sdk.utils import pprint
from ufa_quant_sdk.config import API_KEY, ENVIRONMENT

# 量化开赛前会更新API BASE-URL
TEST_TRADE_URL = "https://ufacareer.com/test_trade/api/"
PROD_TRADE_URL = "https://ufacareer.com/trade/api/"

TRADE_URL = PROD_TRADE_URL if ENVIRONMENT == 'prod' else TEST_TRADE_URL

# ---------- 账户 ACCOUNT ---------- #

# 查询余额
ROUTE_ACCOUNT_CASH = "account/cash"
def get_cash_avaliable():
    return _post_request_data(TRADE_URL, ROUTE_ACCOUNT_CASH, {'API_KEY': API_KEY})

# 查询当前总资产
ROUTE_ACCOUNT_TOTAL_ASSET = "account/asset_total"
def get_total_asset():
    return _post_request_data(TRADE_URL, ROUTE_ACCOUNT_TOTAL_ASSET, {'API_KEY': API_KEY})

# 查询持仓
ROUTE_ACCOUNT_POSITIONS = "account/positions"
def get_positions():
    return _post_request_data(TRADE_URL, ROUTE_ACCOUNT_POSITIONS, {'API_KEY': API_KEY})

# 查询资产曲线
ROUTE_ACCOUNT_ASSET_HIST = "account/asset_hist"
def get_asset_hist(days):
    return _post_request_data(TRADE_URL, ROUTE_ACCOUNT_ASSET_HIST, {'API_KEY': API_KEY, 'days': days})


# ---------- 交易 TRADE ---------- #

# 查询订单
ROUTE_ACCOUNT_ORDERS = "trade/orders"
def get_orders(status="open"):
    return _post_request_data(TRADE_URL, ROUTE_ACCOUNT_ORDERS, {'API_KEY': API_KEY, 'status': status})

# 下单
ROUTE_MAKE_ORDER = "trade/make_order"
def make_order(symbol, order_type, side, amount, order_price=None):
    return _post_request_data(TRADE_URL, ROUTE_MAKE_ORDER, {
        'API_KEY': API_KEY,
        'symbol': symbol,
        'side': side,
        'order_type': order_type,
        'amount': amount,
        'order_price': order_price
    })

# 撤单
ROUTE_CANCEL_ORDER = 'trade/cancel_order'
def cancel_order(order_id):
    return _post_request_data(TRADE_URL, ROUTE_CANCEL_ORDER, {
        'API_KEY': API_KEY,
        'order_id': order_id,
    })

# if __name__ == '__main__':
#     pprint(get_cash_avaliable())
#     pprint(get_positions())
#     pprint(get_orders())
