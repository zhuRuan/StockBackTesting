from ufa_quant_sdk.apis import _get_request_data, _post_request_data
from ufa_quant_sdk.utils import pprint

# 量化开赛前会更新API BASE-URL
FINANCE_URL = "https://ufacareer.com/finance/api/"


# ---------- 通用行情 GENERAL ---------- #

# 股票列表
ROUTE_SYMBOL_LIST = "general/symbols"
def get_symbol_list():
    return _get_request_data(FINANCE_URL, ROUTE_SYMBOL_LIST, {'type': 'a'})

# 指数列表
ROUTE_INDEX_LIST = "general/indexs"
def get_index_list():
    return _get_request_data(FINANCE_URL, ROUTE_INDEX_LIST, {'type': 'a'})

# 概念板块列表
ROUTE_CONCEPT_BOARD_LIST = "general/concept_board"
def get_concept_board_list():
    return _get_request_data(FINANCE_URL, ROUTE_CONCEPT_BOARD_LIST, {})

# 行业板块列表
ROUTE_INDUSTRY_BOARD_LIST = "general/industry_board"
def get_industry_board_list():
    return _get_request_data(FINANCE_URL, ROUTE_INDUSTRY_BOARD_LIST, {})

# 概念成分股
ROUTE_CONCEPT_MEMBER = "general/concept_member"
def get_concept_member(concept):
    return _get_request_data(FINANCE_URL, ROUTE_CONCEPT_MEMBER, {'concept': concept})

# 行业成分股
ROUTE_INDUSTRY_MEMBER = "general/industry_member"
def get_industry_member(industry):
    return _get_request_data(FINANCE_URL, ROUTE_INDUSTRY_MEMBER, {'industry': industry})


# ---------- 历史行情 GENERAL ---------- #

# K线
ROUTE_KLINE = "hist/kline"
def get_kline(symbol, start, end, tf='1d'):
    payload = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "tf": tf
    }
    return _post_request_data(FINANCE_URL, ROUTE_KLINE, payload)


# ---------- 实时行情 LIVE ---------- #

# 个股信息（单一）
ROUTE_STOCK_INFO = 'live/stock_info'
def get_stock_info(symbol):
    return _get_request_data(FINANCE_URL, ROUTE_STOCK_INFO, {'symbol': symbol})

# 个股信息（批量）
ROUTE_STOCK_INFO_LIST = "live/stock_infos"
def get_stock_info_list(symbols):
    return _post_request_data(FINANCE_URL, ROUTE_STOCK_INFO_LIST, {'symbols': symbols})

# 指数信息（单一）
ROUTE_INDEX_INFO = 'live/index_info'
def get_index_info(symbol):
    return _get_request_data(FINANCE_URL, ROUTE_INDEX_INFO, {'symbol': symbol})

# 指数信息（批量）
ROUTE_INDEX_INFO_LIST = "live/index_infos"
def get_index_info_list(symbols):
    return _post_request_data(FINANCE_URL, ROUTE_INDEX_INFO_LIST, {'symbols': symbols})


# if __name__ == '__main__':
#     pprint(get_kline("SZ.000001", "2022-03-01 00:00:00", "2022-03-10 00:00:00"))
