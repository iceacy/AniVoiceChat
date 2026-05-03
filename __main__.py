"""AniVoiceChat - 动漫角色语音对话系统

统一CLI入口点

Usage:
    python -m AniVoiceChat chat              # 交互式文本对话
    python -m AniVoiceChat voice             # 语音对话模式
    python -m AniVoiceChat check             # 检查配置
    python -m AniVoiceChat test-asr          # 测试ASR模块
    python -m AniVoiceChat test-llm          # 测试LLM模块
    python -m AniVoiceChat test-tts          # 测试TTS模块
"""
import sys
import argparse

# 添加项目根目录到路径
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli import ChatCommand, VoiceCommand, CheckCommand
from cli.test import TestASRCommand, TestLLMCommand, TestTTSCommand


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="AniVoiceChat",
        description="动漫角色语音对话系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m AniVoiceChat chat              启动交互式对话
  python -m AniVoiceChat voice             启动语音对话
  python -m AniVoiceChat check --verbose   详细检查配置
  python -m AniVoiceChat test-asr test.wav 测试ASR识别
  python -m AniVoiceChat test-tts "你好"   测试TTS合成
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="可用命令",
        description="使用 'python -m AniVoiceChat <command> --help' 查看各命令详细帮助",
        metavar="command",
    )

    # 注册所有命令
    commands = {
        "chat": ChatCommand(),
        "voice": VoiceCommand(),
        "check": CheckCommand(),
        "test-asr": TestASRCommand(),
        "test-llm": TestLLMCommand(),
        "test-tts": TestTTSCommand(),
    }

    for name, cmd in commands.items():
        cmd.register(subparsers.add_parser(
            name,
            help=_get_command_help(name),
            add_help=True,
        ))

    return parser


def _get_command_help(command: str) -> str:
    """获取命令帮助文本"""
    helps = {
        "chat": "交互式文本对话模式",
        "voice": "语音对话模式（需要麦克风）",
        "check": "检查配置和依赖",
        "test-asr": "测试ASR语音识别模块",
        "test-llm": "测试LLM对话模块",
        "test-tts": "测试TTS语音合成模块",
    }
    return helps.get(command, "")


def main() -> int:
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # 命令映射
    commands = {
        "chat": ChatCommand(),
        "voice": VoiceCommand(),
        "check": CheckCommand(),
        "test-asr": TestASRCommand(),
        "test-llm": TestLLMCommand(),
        "test-tts": TestTTSCommand(),
    }

    if args.command in commands:
        return commands[args.command].run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
