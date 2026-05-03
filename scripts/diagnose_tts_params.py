"""查看GPT-SoVITS调用参数"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.tts import GPTSoVITS


def show_tts_params():
    print("=" * 60)
    print("GPT-SoVITS 调用参数详情")
    print("=" * 60)

    config = Config.load()

    print(f"\n[配置信息]")
    print(f"  API URL: {config.tts.api_url}")
    print(f"  GPT-SoVITS根目录: {config.tts.gpt_sovits_root}")
    print(f"  角色: {config.tts.character}")
    print(f"  目标采样率: {config.tts.sample_rate}")

    print(f"\n[参考音频配置]")
    print(f"  参考音频路径: {config.tts.ref_audio_path}")
    print(f"  参考文本: {config.tts.prompt_text}")
    print(f"  参考语言: {config.tts.prompt_lang}")

    # 测试文本
    test_text = "ねえ、今日一緒に帰らない？途中で新しいクレープ屋さん見つけたんだけど、すごく美味しそうなの！寄ってみない？"

    tts = GPTSoVITS(config.tts)

    print(f"\n[输入文本]")
    print(f"  文本: {test_text}")

    detected_lang = tts._detect_language(test_text)
    print(f"  检测语言: {detected_lang}")

    print(f"\n[发送到GPT-SoVITS API的参数]")
    payload = {
        "text": test_text,
        "text_lang": detected_lang,
        "ref_audio_path": config.tts.ref_audio_path,
        "prompt_text": config.tts.prompt_text,
        "prompt_lang": config.tts.prompt_lang,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n[参数说明]")
    print(f"  text: 要合成的文本")
    print(f"  text_lang: 文本语言 (zh=中文, ja=日文, en=英文)")
    print(f"  ref_audio_path: 参考音频文件路径")
    print(f"  prompt_text: 参考音频对应的文本")
    print(f"  prompt_lang: 参考音频的语言 (必须与参考音频实际语言匹配)")

    print(f"\n[重要提醒]")
    print(f"  prompt_lang 必须与参考音频的实际语言一致！")
    print(f"  如果参考音频是日文但 prompt_lang=zh，会导致音质异常")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    show_tts_params()
