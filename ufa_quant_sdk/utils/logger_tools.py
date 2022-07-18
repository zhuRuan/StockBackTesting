import os
import logging
from logging.handlers import RotatingFileHandler

def get_general_logger(name='unnamed', path='logs', level=logging.DEBUG):
    logger = logging.getLogger(name)
    if logger.level != 0:
      return logger
    os.makedirs(path, exist_ok=True)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_name = os.path.join(path, '{}.log'.format(name))
    file_handler = RotatingFileHandler(
        file_name, mode='a', maxBytes=20 * 1024 * 1024,
        backupCount=3, encoding='utf8', delay=0)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.DEBUG)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
