"""统一日志配置模块"""
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 从环境变量读取日志文件配置
_LOG_FILE = os.getenv("LOG_FILE")


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅用于控制台）"""

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
        # 保存原始 levelname
        original_levelname = record.levelname
        # 添加颜色
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        # 格式化
        result = super().format(record)
        # 恢复原始 levelname（避免污染其他 handler）
        record.levelname = original_levelname
        return result


# 根 logger 配置状态
_root_configured = False


def _configure_root():
    """配置根 logger（只执行一次）"""
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    if not root.handlers:
        # 控制台 handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter('[%(levelname)s] %(name)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

        # 文件 handler（如果环境变量配置了 LOG_FILE）
        if _LOG_FILE:
            log_path = Path(_LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(_LOG_FILE, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)

        root.setLevel(logging.INFO)

    _root_configured = True


def get_logger(name: str = None) -> logging.Logger:
    """
    获取 logger 实例

    文件日志通过 .env 的 LOG_FILE 配置

    Args:
        name: logger 名称，默认使用调用模块名

    Returns:
        logger 实例
    """
    _configure_root()

    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'AniVoiceChat')

    return logging.getLogger(name)


# 模块导入时自动配置根 logger
_configure_root()
