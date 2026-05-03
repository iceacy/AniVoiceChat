"""TTS测试命令"""
import argparse
from pathlib import Path
from cli.base import Command


class TestTTSCommand(Command):
    """测试TTS模块命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("text", nargs="?", help="要合成的文本")
        parser.add_argument("--output", "-o", help="输出音频文件路径")
        parser.add_argument("--character", help="指定角色")

    def run(self, args: argparse.Namespace) -> int:
        """执行TTS测试"""
        self.print_header("TTS模块测试")

        try:
            from src.config import Config
            from src.tts import GPTSoVITS

            config = Config.load()

            if args.character:
                config.tts.character = args.character

            print(f"\n[INIT] 初始化TTS...")
            print(f"[INFO] API地址: {config.tts.api_url}")
            print(f"[INFO] 角色: {config.tts.character}")

            tts = GPTSoVITS(config.tts)

            if not tts.check_server():
                print("[ERROR] TTS服务器未运行，请启动GPT-SoVITS API服务")
                return 1

            print("[SUCCESS] TTS服务器连接正常")

            if args.text:
                print(f"\n[TEST] 文本: {args.text}")
                print("[TEST] 合成语音...")

                result = tts.synthesize(args.text)

                if result:
                    audio_data, sample_rate = result
                    duration = len(audio_data) / sample_rate
                    print(f"[RESULT] 合成成功，时长: {duration:.2f}秒")

                    if args.output:
                        output_path = Path(args.output)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        if tts.save_audio(audio_data, str(output_path)):
                            print(f"[SUCCESS] 音频已保存: {output_path}")
                        else:
                            print("[ERROR] 保存音频失败")
                            return 1
                else:
                    print("[ERROR] 合成失败")
                    return 1
            else:
                print("\n[提示] 使用参数指定要合成的文本")
                print("示例: python -m AniVoiceChat test-tts '你好，很高兴见到你'")

            return 0

        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
