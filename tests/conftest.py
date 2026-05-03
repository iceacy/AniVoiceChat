"""Pytest 配置和共享 fixtures"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.config import Config


@pytest.fixture
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_audio_int16():
    """生成示例音频数据（int16 格式）"""
    duration = 1.0  # 秒
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # 生成正弦波测试音频
    t = np.linspace(0, duration, num_samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    return audio


@pytest.fixture
def sample_audio_float32():
    """生成示例音频数据（float32 格式，范围 [-1, 1]）"""
    duration = 1.0  # 秒
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # 生成正弦波测试音频
    t = np.linspace(0, duration, num_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    return audio


@pytest.fixture
def mock_config():
    """Mock 配置对象"""
    config = Mock(spec=Config)

    # ASR 配置
    config.asr = Mock()
    config.asr.model_path = "/fake/model/path"
    config.asr.device = "cpu"
    config.asr.max_new_tokens = 256
    config.asr.max_inference_batch_size = 32

    # LLM 配置
    config.llm = Mock()
    config.llm.api_key = "test_api_key"
    config.llm.base_url = "https://api.test.com/v1"
    config.llm.model = "test-model"
    config.llm.provider = "test"
    config.llm.temperature = 0.8
    config.llm.max_tokens = 1000

    # TTS 配置
    config.tts = Mock()
    config.tts.api_url = "http://127.0.0.1:9880"
    config.tts.gpt_sovits_root = "/fake/gpt-sovits"
    config.tts.character = "测试角色"
    config.tts.sample_rate = 16000
    config.tts.ref_audio_path = "/fake/ref.wav"
    config.tts.prompt_text = "测试参考文本"
    config.tts.prompt_lang = "zh"
    config.tts.speed_factor = 1.0
    config.tts.break_step = 50

    # 音频配置
    config.audio = Mock()
    config.audio.sample_rate = 16000
    config.audio.chunk_size = 1024
    config.audio.channels = 1

    return config


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录 fixture"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_tts_response():
    """Mock TTS API 响应"""
    duration = 2.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # 生成模拟音频响应
    audio_data = np.random.uniform(-0.5, 0.5, num_samples).astype(np.float32)

    # 模拟 WAV 文件字节
    import io
    from scipy.io import wavfile

    audio_int16 = (audio_data * 32767).astype(np.int16)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, audio_int16)
    wav_bytes = wav_buffer.getvalue()

    return wav_bytes


@pytest.fixture(autouse=True)
def reset_config(monkeypatch):
    """每个测试前重置配置环境变量"""
    monkeypatch.setenv("LLM_PROVIDER", "glm")
    monkeypatch.setenv("GLM_API_KEY", "test_key_for_testing")
    monkeypatch.setenv("DEVICE", "cpu")
