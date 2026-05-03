"""测试对话管道"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.dialogue_pipeline import DialoguePipeline


def test_pipeline():
    print("=" * 50)
    print("AniVoiceChat 对话管道测试")
    print("=" * 50)

    config = Config.load()

    print("\n[CONFIG] 配置信息:")
    print(f"  ASR模型: {config.asr.model_path}")
    print(f"  ASR设备: {config.asr.device}")
    print(f"  LLM模型: {config.llm.model}")
    print(f"  TTS服务: {config.tts.api_url}")
    print(f"  角色: {config.tts.character}")

    if not config.validate():
        print("\n[ERROR] 配置验证失败，请检查.env配置")
        return

    print("\n[INIT] 初始化对话管道...")
    pipeline = DialoguePipeline(config)

    print("\n" + "=" * 50)
    print("测试模式选择:")
    print("1. 文本输入测试")
    print("2. 音频文件测试")
    print("3. 退出")
    print("=" * 50)

    while True:
        choice = input("\n请选择测试模式 (1-3): ").strip()

        if choice == "1":
            test_text_input(pipeline)
        elif choice == "2":
            test_audio_input(pipeline)
        elif choice == "3":
            print("[INFO] 退出测试")
            break
        else:
            print("[WARNING] 无效选择")


def test_text_input(pipeline: DialoguePipeline):
    """测试文本输入"""
    print("\n--- 文本输入模式 ---")
    print("提示: 输入 'quit' 退出, 'clear' 清空历史, 'history' 查看历史")

    while True:
        user_input = input("\n请输入文本: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if user_input.lower() == "clear":
            pipeline.clear_history()
            continue

        if user_input.lower() == "history":
            history = pipeline.get_history()
            print("\n[对话历史]")
            for msg in history[-6:]:
                role = msg["role"]
                content = msg["content"]
                print(f"  {role}: {content}")
            continue

        result = pipeline.process_text_input(user_input)

        if result:
            response_text, audio_data, sample_rate = result

            if audio_data is not None:
                save = input("\n是否保存音频? (y/n): ").strip().lower()
                if save == "y":
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"response_{len(pipeline.history)//2}.wav"
                    pipeline.save_response_audio(audio_data, str(output_path))
                    print(f"[INFO] 音频已保存: {output_path}")


def test_audio_input(pipeline: DialoguePipeline):
    """测试音频输入"""
    print("\n--- 音频文件模式 ---")

    audio_path = input("请输入音频文件路径: ").strip()

    if not Path(audio_path).exists():
        print(f"[ERROR] 文件不存在: {audio_path}")
        return

    result = pipeline.process_audio_input(audio_path)

    if result:
        user_text, response_text, audio_data, sample_rate = result

        print(f"\n[结果]")
        print(f"  用户: {user_text}")
        print(f"  回复: {response_text}")

        if audio_data is not None:
            save = input("\n是否保存音频? (y/n): ").strip().lower()
            if save == "y":
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"response_{len(pipeline.history)//2}.wav"
                pipeline.save_response_audio(audio_data, str(output_path))
                print(f"[INFO] 音频已保存: {output_path}")


if __name__ == "__main__":
    test_pipeline()
