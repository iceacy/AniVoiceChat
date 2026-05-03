"""配置管理模块"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

load_dotenv()

@dataclass
class ASRConfig:
    """ASR配置"""
    model_path: str = ""
    device: str = "cuda"
    max_new_tokens: int = 256
    max_inference_batch_size: int = 32

    @classmethod
    def from_env(cls) -> "ASRConfig":
        model_path = os.getenv("QWEN_ASR_MODEL_PATH", "")
        if not model_path:
            base_dir = "D:/Vscode_Program/AniVoiceChat/models/Qwen"
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    if "Qwen3-ASR" in item:
                        model_path = os.path.join(base_dir, item)
                        break

        return cls(
            model_path=model_path,
            device=os.getenv("DEVICE", "cuda"),
            max_new_tokens=int(os.getenv("ASR_MAX_NEW_TOKENS", "256")),
            max_inference_batch_size=int(os.getenv("ASR_BATCH_SIZE", "32")),
        )


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    provider: str = "deepseek"
    temperature: float = 0.8
    max_tokens: int = 1000

    # 预设提供商配置
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "glm": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-flash",
            "env_key": "GLM_API_KEY",
        },
    }

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

        if provider not in cls.PROVIDERS:
            raise ValueError(f"不支持的LLM提供商: {provider}，支持: {list(cls.PROVIDERS.keys())}")

        provider_config = cls.PROVIDERS[provider]
        env_key = provider_config["env_key"]

        api_key = os.getenv(env_key, "")
        if not api_key:
            raise ValueError(f"{env_key} 未找到，请检查.env配置")

        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", provider_config["base_url"]),
            model=os.getenv("LLM_MODEL", provider_config["model"]),
            provider=provider,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.8")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        )


@dataclass
class TTSConfig:
    """TTS配置"""
    api_url: str = "http://127.0.0.1:9880"
    gpt_sovits_root: str = ""
    character: str = "妮露_JA"
    sample_rate: int = 16000

    ref_audio_path: Optional[str] = None
    prompt_text: str = ""
    prompt_lang: str = "zh"

    # 语音合成参数
    speed_factor: float = 1.1  # 语速系数，1.1略快可减少停顿感
    break_step: int = 50       # 标点停顿时长(ms)，默认50ms减少停顿

    @classmethod
    def from_env(cls) -> "TTSConfig":
        gpt_root = os.getenv("GPT_SOVITS_ROOT", "")
        character = os.getenv("DEFAULT_CHARACTER", "妮露_JA")

        # 从角色名称中提取语言标识 (_JA, _ZH, _EN)
        prompt_l = "zh"  # 默认中文
        if "_JA" in character.upper():
            prompt_l = "ja"
        elif "_EN" in character.upper():
            prompt_l = "en"

        ref_audio = None
        prompt_t = ""

        if gpt_root:
            references_dir = Path(gpt_root) / "references" / character
            if references_dir.exists():
                ref_files = list(references_dir.glob("*.wav"))
                if ref_files:
                    ref_audio = str(ref_files[0])
                    ref_info_path = references_dir / "reference.txt"
                    if ref_info_path.exists():
                        with open(ref_info_path, "r", encoding="utf-8") as f:
                            prompt_t = f.read().strip()

        return cls(
            api_url=os.getenv("TTS_API_URL", "http://127.0.0.1:9880"),
            gpt_sovits_root=gpt_root,
            character=character,
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            ref_audio_path=ref_audio,
            prompt_text=prompt_t,
            prompt_lang=prompt_l,
            speed_factor=float(os.getenv("TTS_SPEED_FACTOR", "1.1")),
            break_step=int(os.getenv("TTS_BREAK_STEP", "50")),
        )


@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1

    @classmethod
    def from_env(cls) -> "AudioConfig":
        return cls(
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
            channels=1,
        )


@dataclass
class LanguageMapConfig:
    """语言到角色的映射配置"""
    zh: str = ""  # 中文使用的角色
    ja: str = ""  # 日文使用的角色
    en: str = ""  # 英文使用的角色

    @classmethod
    def from_env(cls) -> "LanguageMapConfig":
        return cls(
            zh=os.getenv("LANG_MAP_ZH", ""),
            ja=os.getenv("LANG_MAP_JA", ""),
            en=os.getenv("LANG_MAP_EN", ""),
        )

    def get_character(self, lang: str) -> Optional[str]:
        """根据语言获取对应的角色ID，未配置则返回None"""
        lang_map = {"zh": self.zh, "ja": self.ja, "en": self.en}
        char_id = lang_map.get(lang)
        return char_id if char_id else None


@dataclass
class CharacterPrompt:
    """角色提示词 - 向后兼容，使用 CharacterManager 加载"""

    name: str = "妮露"
    description: str = "你是原神角色妮露，一位温柔、善良的舞者。"
    personality: str = """
    你是妮露，来自须弥城的祖拜尔剧场的舞者。
    性格特点：温柔、善良、热爱艺术、关心他人。
    说话风格：温和有礼，喜欢用比喻，有时会提及舞蹈和艺术。
    语言：能够使用中文和日文交流。
    """

    @classmethod
    def get_prompt(cls, character: str = "妮露_JA") -> str:
        """
        获取角色提示词

        优先从 YAML 文件加载，失败时使用内置默认值
        """
        from .character_manager import get_character_manager

        manager = get_character_manager()
        config = manager.load_character(character)

        if config and config.system_prompt:
            return config.system_prompt

        return cls._get_default_prompt(character)

    @classmethod
    def _get_default_prompt(cls, character: str) -> str:
        """内置默认提示词（向后兼容）"""
        default_prompt = """
【角色设定】
你是原神中的角色妮露（Nilou），须弥祖拜尔剧场的明星舞者。

【性格特征】
- 温柔善良，对艺术充满热情
- 关心他人，善解人意
- 积极乐观，但也珍惜美好时光
- 谦逊有礼，不喜欢摆架子

【说话风格】
- 语气温柔，常用"呢"、"呀"等语气词
- 喜欢用花朵、舞蹈等艺术相关的比喻
- 偶尔会提及"莎芭妮"、"剧场"相关内容
- 回复简练但充满温暖

【回复原则】
- 保持角色设定，不要出戏
- 简洁回复，通常1-3句话
- 根据用户语言自动切换中文/日文
"""
        if character.startswith("妮露"):
            return default_prompt

        return "你是一位动漫角色，请以角色的语气回复用户。保持简洁，1-3句话即可。"


@dataclass
class Config:
    """总配置"""
    asr: ASRConfig = field(default_factory=ASRConfig.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    tts: TTSConfig = field(default_factory=TTSConfig.from_env)
    audio: AudioConfig = field(default_factory=AudioConfig.from_env)
    lang_map: LanguageMapConfig = field(default_factory=LanguageMapConfig.from_env)

    @classmethod
    def load(cls) -> "Config":
        return cls()

    def validate(self) -> bool:
        """验证配置完整性"""
        errors = []

        if not self.asr.model_path or not os.path.exists(self.asr.model_path):
            errors.append(f"ASR模型路径不存在: {self.asr.model_path}")

        if not self.llm.api_key:
            errors.append("DeepSeek API密钥未配置")

        if not self.tts.ref_audio_path:
            errors.append(f"TTS参考音频未找到: {self.tts.ref_audio_path}")

        if errors:
            logger.error("配置验证失败:")
            for err in errors:
                logger.error(f"  - {err}")
            return False

        return True


def get_config() -> Config:
    """获取全局配置"""
    return Config.load()
