"""统一日志配置模块"""
import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[37m',       # 白色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "AniVoiceChat",
    level: int = logging.INFO,
    log_file: str = None,
    colored: bool = True
) -> logging.Logger:
    """
    设置并返回配置好的logger

    Args:
        name: logger名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        colored: 是否使用彩色输出

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 格式化器
    log_format = '[%(levelname)s] %(name)s - %(message)s'
    if colored:
        formatter = ColoredFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称，默认使用调用模块名

    Returns:
        logger实例
    """
    if name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'AniVoiceChat')

    return logging.getLogger(name)


# 默认logger
default_logger = setup_logger()


# 简化的日志函数，兼容旧的print风格
def info(msg: str, *args, **kwargs):
    """INFO级别日志"""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """WARNING级别日志"""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """ERROR级别日志"""
    default_logger.error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    """DEBUG级别日志"""
    default_logger.debug(msg, *args, **kwargs)
