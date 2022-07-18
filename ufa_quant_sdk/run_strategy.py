from math import ceil
from apis.trade import get_cash_avaliable, get_orders, get_positions, get_total_asset
from utils.logger_tools import get_general_logger
from utils.thread_tools import IntervalThread
from utils import abspath
from config import STRATEGY_NAME, STRATEGY_INTERVAL, API_KEY
import importlib
import time

logger = get_general_logger(STRATEGY_NAME, path=abspath('logs'))

class AccountContext:

    def __init__(self) -> None:
        self.cash_avaliable = 0
        self.total_asset = 0
        self.positions = {"available": [], "locked": [], "new": []}
        self.open_orders = []
    
    def update(self):
        self.cash_avaliable = get_cash_avaliable()
        self.total_asset = get_total_asset()
        self.positions = get_positions()
        self.open_orders = get_orders('open')

class StrategyExecutor:

    def __init__(self) -> None:
        self.context = AccountContext()

    def monitor(self):
        self.context.update()
        logger.info(f'>>>>>>>> Account Report >>>>>>>>')
        logger.info(f'Cash avaliable: {self.context.cash_avaliable} Total asset: {self.context.total_asset}')

    def strategy(self):
        module = importlib.import_module(f"strategy.{STRATEGY_NAME}")
        importlib.reload(module)
        
        t1 = time.time()
        logger.info("<<<<<<<< Strategy Start <<<<<<<<")
        try:
            module.main(self.context)
        except Exception as e:
            logger.error(e, exc_info=True)
        logger.info("<<<<<<<< Strategy Finish <<<<<<<<")
        t2 = time.time()
        logger.info(f"Time cost: {round(t2 - t1, 2)}s")

    def run(self):

        logger.info("Initializing...")
        self.monitor_thread = IntervalThread(interval=ceil(STRATEGY_INTERVAL * 60), target=self.monitor)
        self.monitor_thread.start()

        time.sleep(5)

        self.strategy_thread = IntervalThread(interval=ceil(STRATEGY_INTERVAL * 60), target=self.strategy)
        self.strategy_thread.start()

        self.monitor_thread.join()


if __name__ == "__main__":
    system = StrategyExecutor()
    system.run()