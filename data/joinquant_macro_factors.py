import time

from jqdatasdk import *
import pandas as pd

def get_macro_factors ():
    q= query(macro.MAC_RMB_EXCHANGE_RATE).limit(10)
    df = macro.run_query(q)
    print(df)


if __name__ == '__main__':
    # 主程序开始
    begin_time = time.time()

    auth('18620290503', 'gxqh2019')