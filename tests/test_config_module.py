"""配置模块单元测试"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.config import (
    ASRConfig,
    LLMConfig,
    TTSConfig,
    AudioConfig,
    Config,
    CharacterPrompt,
)


class TestASRConfig:
    """ASR 配置测试"""

    def test_from_env_default_values(self, monkeypatch):
        """测试从环境变量加载 ASR 配置（默认值）"""
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("ASR_MAX_NEW_TOKENS", "256")
        monkeypatch.setenv("ASR_BATCH_SIZE", "32")

        config = ASRConfig.from_env()

        assert config.device == "cuda"
        assert config.max_new_tokens == 256
        assert config.max_inference_batch_size == 32


class TestLLMConfig:
    """LLM 配置测试"""

    def test_provider_validation(self, monkeypatch):
        """测试 LLM 提供商验证"""
        monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
        monkeypatch.setenv("GLM_API_KEY", "test_key")

        with pytest.raises(ValueError, match="不支持的LLM提供商"):
            LLMConfig.from_env()

    def test_glm_provider(self, monkeypatch):
        """测试 GLM 提供商配置"""
        monkeypatch.setenv("LLM_PROVIDER", "glm")
        monkeypatch.setenv("GLM_API_KEY", "test_glm_key")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api/v1")
        monkeypatch.setenv("LLM_MODEL", "glm-4-custom")

        config = LLMConfig.from_env()

        assert config.provider == "glm"
        assert config.api_key == "test_glm_key"
        assert config.base_url == "https://custom.api/v1"
        assert config.model == "glm-4-custom"

    def test_deepseek_provider(self, monkeypatch):
        """测试 DeepSeek 提供商配置"""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test_deepseek_key")
        # 清除可能影响的其他环境变量
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        config = LLMConfig.from_env()

        assert config.provider == "deepseek"
        assert config.api_key == "test_deepseek_key"
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.model == "deepseek-chat"

    def test_missing_api_key(self, monkeypatch):
        """测试缺少 API 密钥"""
        monkeypatch.setenv("LLM_PROVIDER", "glm")
        monkeypatch.delenv("GLM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GLM_API_KEY"):
            LLMConfig.from_env()


class TestTTSConfig:
    """TTS 配置测试"""

    def test_from_env_defaults(self, monkeypatch):
        """测试 TTS 配置默认值"""
        monkeypatch.setenv("GPT_SOVITS_ROOT", "/fake/root")
        monkeypatch.setenv("DEFAULT_CHARACTER", "测试角色_ZH")
        monkeypatch.setenv("SAMPLE_RATE", "22050")
        monkeypatch.setenv("TTS_SPEED_FACTOR", "1.2")
        monkeypatch.setenv("TTS_BREAK_STEP", "60")

        config = TTSConfig.from_env()

        assert config.gpt_sovits_root == "/fake/root"
        assert config.character == "测试角色_ZH"
        assert config.sample_rate == 22050
        assert config.speed_factor == 1.2
        assert config.break_step == 60

    def test_language_detection_from_character_name(self, monkeypatch):
        """测试从角色名称检测语言"""
        test_cases = [
            ("角色_JA", "ja"),
            ("角色_ZH", "zh"),
            ("角色_EN", "en"),
            ("角色", "zh"),  # 默认中文
        ]

        for character, expected_lang in test_cases:
            monkeypatch.setenv("DEFAULT_CHARACTER", character)
            monkeypatch.setenv("GPT_SOVITS_ROOT", "/fake/root")

            config = TTSConfig.from_env()
            assert config.prompt_lang == expected_lang, f"Failed for {character}"


class TestAudioConfig:
    """音频配置测试"""

    def test_from_env(self, monkeypatch):
        """测试音频配置加载"""
        monkeypatch.setenv("SAMPLE_RATE", "48000")
        monkeypatch.setenv("CHUNK_SIZE", "2048")

        config = AudioConfig.from_env()

        assert config.sample_rate == 48000
        assert config.chunk_size == 2048
        assert config.channels == 1


class TestConfig:
    """总配置测试"""

    def test_config_loads_all_sub_configs(self, monkeypatch):
        """测试配置加载所有子配置"""
        # 设置有效的环境变量
        monkeypatch.setenv("LLM_PROVIDER", "glm")
        monkeypatch.setenv("GLM_API_KEY", "test_key_for_config")

        config = Config.load()

        assert hasattr(config, "asr")
        assert hasattr(config, "llm")
        assert hasattr(config, "tts")
        assert hasattr(config, "audio")

    def test_validate_returns_false_on_missing_model_path(self, mock_config):
        """测试配置验证 - 模型路径缺失"""
        mock_config.asr.model_path = ""
        # 让 validate 返回 False 而不是 Mock 对象
        mock_config.validate = Config.validate

        result = Config.validate(mock_config)
        assert result is False

    def test_validate_returns_false_on_missing_api_key(self, mock_config):
        """测试配置验证 - API 密钥缺失"""
        mock_config.llm.api_key = ""
        # 让 validate 返回 False 而不是 Mock 对象
        mock_config.validate = Config.validate

        result = Config.validate(mock_config)
        assert result is False


class TestCharacterPrompt:
    """角色提示词测试"""

    def test_get_prompt_for_nilou(self):
        """测试获取妮露角色提示词"""
        prompt = CharacterPrompt.get_prompt("妮露_JA")

        assert "妮露" in prompt
        assert "Nilou" in prompt
        assert "须弥" in prompt

    def test_get_prompt_for_unknown_character(self):
        """测试获取未知角色提示词（返回默认）"""
        prompt = CharacterPrompt.get_prompt("未知角色")

        assert "动漫角色" in prompt
        assert "1-3句话" in prompt

    def test_get_prompt_nilou_variants(self):
        """测试妮露角色变体"""
        variants = ["妮露_JA", "妮露_ZH", "妮露"]

        for variant in variants:
            prompt = CharacterPrompt.get_prompt(variant)
            assert "妮露" in prompt, f"Failed for variant: {variant}"
