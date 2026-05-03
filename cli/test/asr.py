"""ASR测试命令"""
import argparse
from pathlib import Path
from cli.base import Command


class TestASRCommand(Command):
    """测试ASR模块命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("audio_path", nargs="?", help="音频文件路径")
        parser.add_argument("--language", help="指定识别语言 (zh/ja/en/auto)")

    def run(self, args: argparse.Namespace) -> int:
        """执行ASR测试"""
        self.print_header("ASR模块测试")

        try:
            from src.config import Config
            from src.asr import QwenASR

            config = Config.load()

            if not config.asr.model_path or not Path(config.asr.model_path).exists():
                print(f"[ERROR] ASR模型路径不存在: {config.asr.model_path}")
                return 1

            print("\n[INIT] 加载ASR模型...")
            asr = QwenASR(config.asr)

            if not asr.load():
                print("[ERROR] ASR模型加载失败")
                return 1

            print("[SUCCESS] ASR模型加载成功")

            if args.audio_path:
                audio_path = Path(args.audio_path)
                if not audio_path.exists():
                    print(f"[ERROR] 音频文件不存在: {audio_path}")
                    return 1

                print(f"\n[TEST] 识别音频: {audio_path}")
                text, lang = asr.transcribe(str(audio_path), args.language)

                if text:
                    print(f"[RESULT] 识别文本: {text}")
                    print(f"[RESULT] 检测语言: {lang}")
                else:
                    print("[ERROR] 识别失败")
                    return 1
            else:
                print("\n[提示] 使用 --audio 参数指定音频文件进行测试")
                print("示例: python -m AniVoiceChat test-asr test.wav")

            return 0

        except Exception as e:
            print(f"[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
