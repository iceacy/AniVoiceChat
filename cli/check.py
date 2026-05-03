"""检查配置命令"""
import argparse
from .base import Command


class CheckCommand(Command):
    """检查配置命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    def run(self, args: argparse.Namespace) -> int:
        """执行检查"""
        self.print_header("AniVoiceChat 配置检查")

        try:
            from src.config import Config
            from src.tts import GPTSoVITS

            config = Config.load()

            print("\n[配置项]")
            print(f"ASR模型路径: {config.asr.model_path}")
            print(f"ASR设备: {config.asr.device}")
            print(f"LLM提供商: {config.llm.provider}")
            print(f"LLM模型: {config.llm.model}")
            print(f"LLM Base URL: {config.llm.base_url}")
            print(f"TTS API: {config.tts.api_url}")
            print(f"角色: {config.tts.character}")
            print(f"采样率: {config.audio.sample_rate}Hz")

            print("\n[验证]")
            valid = config.validate()

            if valid:
                print("[SUCCESS] 配置验证通过")

                tts = GPTSoVITS(config.tts)
                if tts.check_server():
                    print("[SUCCESS] TTS服务器连接正常")
                else:
                    print("[WARNING] TTS服务器未运行，请启动GPT-SoVITS API服务")
                    print("[INFO] 运行: python start_tts_server.bat")

                if args.verbose:
                    print("\n[详细信息]")
                    print(f"ASR最大tokens: {config.asr.max_new_tokens}")
                    print(f"LLM温度: {config.llm.temperature}")
                    print(f"TTS语速系数: {config.tts.speed_factor}")
                    print(f"TTS停顿时长: {config.tts.break_step}ms")

            return 0 if valid else 1

        except Exception as e:
            print(f"[ERROR] 配置检查失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
