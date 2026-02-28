import logging
from pathlib import Path

def init_logger(log_file=None, log_file_level=logging.INFO):
    if isinstance(log_file, Path):
        log_file = str(log_file)

    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    logger = logging.getLogger()          # root logger
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # 覆盖已有 handlers，防止重复输出
    logger.handlers = [console_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)   # 关键：别注释
        logger.addHandler(file_handler)

    return logger