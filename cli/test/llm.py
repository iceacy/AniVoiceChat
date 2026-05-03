"""LLM测试命令"""
import argparse
from cli.base import Command


class TestLLMCommand(Command):
    """测试LLM模块命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("text", nargs="?", help="测试文本")
        parser.add_argument("--character", help="指定角色")

    def run(self, args: argparse.Namespace) -> int:
        """执行LLM测试"""
        self.print_header("LLM模块测试")

        try:
            from src.config import Config
            from src.llm import DeepSeekLLM

            config = Config.load()

            if not config.llm.api_key:
                print("[ERROR] LLM API密钥未配置")
                return 1

            print("\n[INIT] 初始化LLM...")
            print(f"[INFO] 提供商: {config.llm.provider}")
            print(f"[INFO] 模型: {config.llm.model}")

            llm = DeepSeekLLM(config.llm)

            if args.character:
                llm.set_character(args.character)

            if args.text:
                print(f"\n[TEST] 输入: {args.text}")
                print("[TEST] 生成回复...")

                response = llm.chat(args.text, [])

                if response:
                    print(f"[RESULT] 回复: {response}")
                else:
                    print("[ERROR] 生成失败")
                    return 1
            else:
                print("\n[提示] 使用参数指定测试文本")
                print("示例: python -m AniVoiceChat test-llm '你好'")

            return 0

        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
