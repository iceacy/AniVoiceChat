"""CLI命令基类"""
import argparse
from abc import ABC, abstractmethod


class Command(ABC):
    """命令基类"""

    @abstractmethod
    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数

        Args:
            parser: 子命令的ArgumentParser实例
        """
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """执行命令

        Args:
            args: 解析后的命令行参数

        Returns:
            退出码，0表示成功，非0表示失败
        """
        pass

    @staticmethod
    def print_header(title: str) -> None:
        """打印标题头部"""
        print("=" * 50)
        print(title)
        print("=" * 50)
