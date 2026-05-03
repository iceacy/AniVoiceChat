"""交互式文本对话命令"""
import argparse
from pathlib import Path
from .base import Command


class ChatCommand(Command):
    """交互式文本对话命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("--character", help="指定角色名称")

    def run(self, args: argparse.Namespace) -> int:
        """执行对话"""
        self.print_header("AniVoiceChat 交互式对话")

        try:
            from src.config import Config
            from src.dialogue_pipeline import DialoguePipeline

            config = Config.load()

            if args.character:
                config.tts.character = args.character

            if not config.validate():
                print("[ERROR] 配置验证失败，请运行 'python -m AniVoiceChat check' 检查")
                return 1

            print("\n[INIT] 初始化对话管道...")
            pipeline = DialoguePipeline(config)

            print("\n提示:")
            print("  - 直接输入文本进行对话")
            print("  - 输入 'quit' 或 'exit' 退出")
            print("  - 输入 'clear' 清空对话历史")
            print("  - 输入 'history' 查看对话历史")
            print("  - 输入 'audio <路径>' 处理音频文件")

            print("\n" + "=" * 50)

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            while True:
                try:
                    user_input = input("\n你: ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("[INFO] 再见!")
                        break

                    if user_input.lower() == "clear":
                        pipeline.clear_history()
                        continue

                    if user_input.lower() == "history":
                        self._show_history(pipeline)
                        continue

                    if user_input.lower().startswith("audio "):
                        audio_path = user_input[6:].strip()
                        if not Path(audio_path).exists():
                            print(f"[ERROR] 文件不存在: {audio_path}")
                            continue
                        result = pipeline.process_audio_input(audio_path)
                    else:
                        result = pipeline.process_text_input(user_input)

                    if result:
                        self._handle_result(pipeline, result, output_dir)

                except KeyboardInterrupt:
                    print("\n[INFO] 再见!")
                    break
                except Exception as e:
                    print(f"[ERROR] 处理失败: {e}")
                    import traceback
                    traceback.print_exc()

            return 0

        except Exception as e:
            print(f"[ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _show_history(self, pipeline) -> None:
        """显示对话历史"""
        history = pipeline.get_history()
        print("\n[对话历史]")
        for i, msg in enumerate(history[-10:], 1):
            role = "你" if msg["role"] == "user" else "角色"
            print(f"  {i}. {role}: {msg['content']}")

    def _handle_result(self, pipeline, result, output_dir: Path) -> None:
        """处理对话结果"""
        if isinstance(result, tuple) and len(result) == 4:
            user_text, response_text, audio_data, sample_rate = result
            print(f"\n角色: {response_text}")
        else:
            response_text, audio_data, sample_rate = result
            print(f"\n角色: {response_text}")

        if audio_data is not None:
            output_path = output_dir / f"response_{len(pipeline.history)//2}.wav"
            pipeline.save_response_audio(audio_data, str(output_path))
            print(f"[INFO] 音频已保存: {output_path}")
