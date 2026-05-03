"""语音对话命令"""
import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
from .base import Command

if TYPE_CHECKING:
    from src.audio import MicrophoneRecorder, AudioPlayer
    from src.dialogue_pipeline import DialoguePipeline


class VoiceCommand(Command):
    """语音对话命令"""

    def register(self, parser: argparse.ArgumentParser) -> None:
        """注册命令参数"""
        parser.add_argument("--character", help="指定角色名称")
        parser.add_argument("--no-save", action="store_true", help="不保存回复音频")

    def run(self, args: argparse.Namespace) -> int:
        """执行语音对话"""
        self.print_header("AniVoiceChat 语音对话模式")

        try:
            from src.config import Config
            from src.dialogue_pipeline import DialoguePipeline
            from src.audio import MicrophoneRecorder, AudioPlayer

            config = Config.load()

            if args.character:
                config.tts.character = args.character

            if not config.validate():
                print("[ERROR] 配置验证失败")
                return 1

            print("\n[INIT] 初始化对话管道...")
            pipeline = DialoguePipeline(config)

            print("\n[INIT] 初始化音频设备...")
            recorder = MicrophoneRecorder(
                sample_rate=config.audio.sample_rate,
                chunk_size=config.audio.chunk_size,
            )
            player = AudioPlayer(sample_rate=config.audio.sample_rate)

            print("\n提示:")
            print("  - 按回车键开始录音")
            print("  - 说话完成后再次按回车键停止")
            print("  - 或者等待自动检测静音停止（约1.5秒静音后）")
            print("  - 输入 'quit' 退出")
            print("  - 输入 'clear' 清空对话历史")
            print("\n" + "=" * 50)

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            return self._main_loop(recorder, player, pipeline, output_dir, args.no_save)

        except Exception as e:
            print(f"[ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            if 'recorder' in locals():
                recorder.cleanup()
            if 'player' in locals():
                player.cleanup()

    def _main_loop(
        self,
        recorder: "MicrophoneRecorder",
        player: "AudioPlayer",
        pipeline: "DialoguePipeline",
        output_dir: Path,
        no_save: bool,
    ) -> int:
        """主循环"""
        recording_in_progress = False
        stop_recording_event = threading.Event()

        while True:
            try:
                user_input = input("\n按回车开始录音... > ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("[INFO] 再见!")
                    break

                if user_input.lower() == "clear":
                    pipeline.clear_history()
                    continue

                print("\n[录音] 录音中...")
                print("[录音] 请对着麦克风说话")
                print("[录音] 说完后按回车键停止，或等待自动检测")

                recorder.start_recording()

                stop_recording_event.clear()
                recording_in_progress = True

                def wait_for_stop():
                    stop_input = input("")
                    if recording_in_progress:
                        stop_recording_event.set()

                stop_thread = threading.Thread(target=wait_for_stop)
                stop_thread.daemon = True
                stop_thread.start()

                audio_data = self._record_with_silence_detection(
                    recorder, stop_recording_event
                )

                recording_in_progress = False
                audio_data = recorder.stop_recording()

                duration = len(audio_data) / 16000
                print(f"[录音] 完成，时长: {duration:.2f}秒")

                if duration < 0.3:
                    print("[WARNING] 录音太短，请重试")
                    continue

                print("\n[处理] ASR识别中...")
                result = pipeline.process_audio_input(audio_data)

                if result:
                    user_text, response_text, audio_output, sample_rate = result

                    print(f"\n你: {user_text}")
                    print(f"角色: {response_text}")

                    if audio_output is not None:
                        print("[播放] 播放回复语音...")
                        player.play(audio_output, blocking=True)

                        if not no_save:
                            output_path = output_dir / f"response_{len(pipeline.history)//2}.wav"
                            pipeline.save_response_audio(audio_output, str(output_path))
                            print(f"[INFO] 音频已保存: {output_path}")

            except KeyboardInterrupt:
                print("\n[INFO] 再见!")
                break
            except Exception as e:
                print(f"[ERROR] 处理失败: {e}")
                import traceback
                traceback.print_exc()

        return 0

    def _record_with_silence_detection(
        self,
        recorder: "MicrophoneRecorder",
        stop_event: threading.Event,
    ) -> np.ndarray:
        """录音并检测静音"""
        import numpy as np

        silence_chunks = int(1.5 * 16000 / 1024)
        min_chunks = int(0.5 * 16000 / 1024)
        total_chunks = 0
        last_buffer_size = 0
        silent_count = 0

        start_time = time.time()
        check_interval = 0.05

        while time.time() - start_time < 30.0:
            if stop_event.is_set():
                print("\n[录音] 手动停止")
                break

            time.sleep(check_interval)

            current_buffer_size = len(recorder.audio_buffer)
            chunks_added = current_buffer_size - last_buffer_size

            if chunks_added > 0:
                buffer_list = list(recorder.audio_buffer)

                for i in range(max(0, len(buffer_list) - chunks_added), len(buffer_list)):
                    if i >= len(buffer_list):
                        break

                    chunk = buffer_list[i]
                    total_chunks += 1

                    if total_chunks >= min_chunks:
                        energy = abs(chunk.astype(np.float32) / 32768.0).mean()
                        if energy < 0.02:
                            silent_count += 1
                            if silent_count >= silence_chunks:
                                print("\n[录音] 检测到静音，自动停止")
                                stop_event.set()
                                break
                        else:
                            silent_count = 0

                    if stop_event.is_set():
                        break

                last_buffer_size = current_buffer_size
                if stop_event.is_set():
                    break

        return np.array([])
