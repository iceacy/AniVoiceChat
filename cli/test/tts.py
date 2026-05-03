"""TTS测试命令"""
import argparse
from pathlib import Path
from cli.base import Command
from src.logger import get_logger

logger = get_logger(__name__)


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

            logger.info("初始化TTS...")
            logger.info(f"API地址: {config.tts.api_url}")
            logger.info(f"角色: {config.tts.character}")

            tts = GPTSoVITS(config.tts)

            if not tts.check_server():
                logger.error("TTS服务器未运行，请启动GPT-SoVITS API服务")
                return 1

            logger.info("TTS服务器连接正常")

            if args.text:
                logger.info(f"文本: {args.text}")
                logger.info("合成语音...")

                result = tts.synthesize(args.text)

                if result:
                    audio_data, sample_rate = result
                    duration = len(audio_data) / sample_rate
                    logger.info(f"合成成功，时长: {duration:.2f}秒")

                    if args.output:
                        output_path = Path(args.output)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        if tts.save_audio(audio_data, str(output_path)):
                            logger.info(f"音频已保存: {output_path}")
                        else:
                            logger.error("保存音频失败")
                            return 1
                else:
                    logger.error("合成失败")
                    return 1
            else:
                print("\n[提示] 使用参数指定要合成的文本")
                print("示例: python -m AniVoiceChat test-tts '你好，很高兴见到你'")

            return 0

        except Exception as e:
            logger.error(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
