"""对话管道模块 - ASR -> LLM -> TTS"""
from typing import Optional, Tuple
import numpy as np
from pathlib import Path

from .asr import QwenASR
from .llm import create_llm, BaseLLM
from .tts import GPTSoVITS
from .config import Config, ASRConfig, LLMConfig, TTSConfig
from .character_manager import get_character_manager, CharacterConfig
from .logger import get_logger

logger = get_logger(__name__)


class DialoguePipeline:
    """对话管道"""

    def __init__(
        self,
        config: Optional[Config] = None,
        asr_config: Optional[ASRConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None,
    ):
        if config is None:
            config = Config.load()

        self.config = config
        self.history = []

        self.asr = QwenASR(asr_config or config.asr)
        self.llm = create_llm(llm_config or config.llm)
        self.tts = GPTSoVITS(tts_config or config.tts)

        self.character_manager = get_character_manager()
        self.lang_map = config.lang_map
        self._asr_loaded = False

    def initialize_asr(self) -> bool:
        """初始化ASR模型"""
        if not self._asr_loaded:
            self._asr_loaded = self.asr.load()
        return self._asr_loaded

    def process_audio_input(
        self,
        audio_input,
        language: Optional[str] = None,
    ) -> Optional[Tuple[str, str, Optional[np.ndarray], int]]:
        """
        处理音频输入，返回完整对话结果

        Args:
            audio_input: 音频文件路径或numpy数组
            language: 指定识别语言 (None=自动)

        Returns:
            (用户文本, 回复文本, 音频数据, 采样率) 或 None
        """
        if not self.initialize_asr():
            return None

        logger.info("[1/3] ASR识别中...")
        user_text, detected_lang = self.asr.transcribe(audio_input, language)

        if not user_text:
            logger.error("ASR识别失败")
            return None

        logger.info(f"[ASR] 检测语言: {detected_lang}")
        logger.info(f"[ASR] 识别文本: {user_text}")

        logger.info("[2/3] LLM生成回复...")
        response_text = self.llm.chat(user_text, self.history)

        if not response_text:
            logger.error("LLM生成失败")
            return user_text, None, None, 0

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text})

        logger.info(f"[LLM] 回复: {response_text}")

        logger.info("[3/3] TTS合成语音...")
        # 根据回复文本语言获取对应的角色和参考音频
        target_character = self._get_character_for_text_language(response_text)
        ref_audio = None
        prompt_lang = None
        prompt_text = None

        if target_character:
            # 切换到对应角色的模型
            self._switch_model_for_character(target_character)
            ref_audio = self._get_ref_audio_for_character(target_character)
            prompt_lang = self._get_prompt_lang_for_character(target_character)
            prompt_text = self._get_prompt_text_for_character(target_character)

        # 如果没有配置语言映射，使用默认行为
        text_lang = self._map_language(detected_lang) if not target_character else self._detect_language(response_text)
        audio_result = self.tts.synthesize(
            response_text,
            text_lang=text_lang,
            ref_audio_path=ref_audio,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text
        )

        if audio_result is None:
            logger.warning("TTS合成失败，返回文本回复")
            return user_text, response_text, None, 0

        audio_data, sample_rate = audio_result
        logger.info(f"[TTS] 合成完成, 时长: {len(audio_data)/sample_rate:.2f}秒")

        return user_text, response_text, audio_data, sample_rate

    def process_text_input(
        self,
        user_text: str,
    ) -> Optional[Tuple[str, Optional[np.ndarray], int]]:
        """
        处理文本输入

        Args:
            user_text: 用户输入文本

        Returns:
            (回复文本, 音频数据, 采样率) 或 None
        """
        logger.info(f"[INPUT] {user_text}")
        logger.info("[LLM] 生成回复...")

        response_text = self.llm.chat(user_text, self.history)

        if not response_text:
            logger.error("LLM生成失败")
            return None

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response_text})

        logger.info(f"[LLM] 回复: {response_text}")

        # 根据回复文本语言获取对应的角色和参考音频
        target_character = self._get_character_for_text_language(response_text)
        ref_audio = None
        prompt_lang = None
        prompt_text = None

        if target_character:
            # 切换到对应角色的模型
            self._switch_model_for_character(target_character)
            ref_audio = self._get_ref_audio_for_character(target_character)
            prompt_lang = self._get_prompt_lang_for_character(target_character)
            prompt_text = self._get_prompt_text_for_character(target_character)

        text_lang = self._detect_language(response_text)
        audio_result = self.tts.synthesize(
            response_text,
            text_lang=text_lang,
            ref_audio_path=ref_audio,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text
        )

        if audio_result is None:
            logger.warning("TTS合成失败")
            return response_text, None, 0

        audio_data, sample_rate = audio_result
        return response_text, audio_data, sample_rate

    def _map_language(self, lang_code: str) -> str:
        """映射语言代码到TTS格式"""
        mapping = {
            "zh": "zh",
            "en": "en",
            "ja": "ja",
            "yue": "zh",
            "Chinese": "zh",
            "English": "en",
            "Japanese": "ja",
        }
        return mapping.get(lang_code, "zh")

    def _detect_language(self, text: str) -> str:
        """检测文本语言 - 优先检测日文假名"""
        has_kana = False
        has_hanzi = False

        for char in text:
            # 日文假名（平假名、片假名）
            if ("぀" <= char <= "ゟ" or   # 平假名
                "゠" <= char <= "ヿ" or   # 片假名
                "ㇰ" <= char <= "ㇿ"):    # 片假名扩展
                has_kana = True
            # 中日韩汉字
            elif "一" <= char <= "鿿":
                has_hanzi = True

        # 如果有假名，判定为日文
        if has_kana:
            return "ja"
        # 只有汉字，判定为中文
        if has_hanzi:
            return "zh"
        return "en"

    def _get_character_for_text_language(self, text: str) -> Optional[str]:
        """
        根据文本语言从配置中获取对应的角色ID

        Returns:
            角色ID，如果该语言未配置则返回None
        """
        detected_lang = self._detect_language(text)
        character_id = self.lang_map.get_character(detected_lang)

        if character_id:
            logger.info(f"检测到{detected_lang.upper()}语言，使用角色: {character_id}")

        return character_id

    def _get_ref_audio_for_character(self, character_id: str) -> Optional[str]:
        """
        获取角色的参考音频路径

        Returns:
            参考音频路径，如果不存在则返回None
        """
        gpt_root = self.config.tts.gpt_sovits_root
        if not gpt_root:
            return None

        references_dir = Path(gpt_root) / "references" / character_id
        if references_dir.exists():
            ref_files = list(references_dir.glob("*.wav"))
            if ref_files:
                return str(ref_files[0])

        return None

    def _get_prompt_text_for_character(self, character_id: str) -> Optional[str]:
        """
        从参考音频文件夹中读取 reference.txt（参考音频对应的文本）

        Returns:
            参考文本内容，如果文件不存在则返回None
        """
        gpt_root = self.config.tts.gpt_sovits_root
        if not gpt_root:
            return None

        reference_txt = Path(gpt_root) / "references" / character_id / "reference.txt"
        if not reference_txt.exists():
            return None

        try:
            with open(reference_txt, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"读取参考文本失败 {reference_txt}: {e}")
            return None

    def _get_prompt_lang_for_character(self, character_id: str) -> str:
        """
        从角色ID中提取参考音频的语言

        Returns:
            语言代码 ("zh", "ja", "en")
        """
        if "_JA" in character_id.upper():
            return "ja"
        elif "_EN" in character_id.upper():
            return "en"
        return "zh"

    def _get_model_config(self, character_id: str) -> Optional[dict]:
        """
        从角色参考音频文件夹中读取模型配置

        Returns:
            {"gpt_model": "...", "sovits_model": "..."} 或 None
        """
        gpt_root = self.config.tts.gpt_sovits_root
        if not gpt_root:
            return None

        model_json = Path(gpt_root) / "references" / character_id / "model.json"
        if not model_json.exists():
            return None

        try:
            import json
            with open(model_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取模型配置失败 {model_json}: {e}")
            return None

    def _switch_model_for_character(self, character_id: str) -> bool:
        """
        切换到指定角色的模型

        Returns:
            是否成功切换
        """
        model_config = self._get_model_config(character_id)
        if not model_config:
            return False

        gpt_model = model_config.get("gpt_model")
        sovits_model = model_config.get("sovits_model")

        if not gpt_model or not sovits_model:
            logger.warning(f"模型配置不完整: {model_config}")
            return False

        return self.tts.set_model(gpt_model, sovits_model)

    def save_response_audio(
        self,
        audio_data: np.ndarray,
        output_path: str,
    ) -> bool:
        """保存回复音频"""
        return self.tts.save_audio(audio_data, output_path)

    def clear_history(self):
        """清空对话历史"""
        self.history = []
        logger.info("对话历史已清空")

    def set_character(self, character: str) -> None:
        """设置角色"""
        config = self.character_manager.load_character(character)
        if config:
            self.llm.set_character(character)
            logger.info(f"角色已设置为: {character} ({config.name})")
        else:
            logger.warning(f"角色配置未找到: {character}，使用默认配置")
            self.llm.set_character(character)

    def get_history(self) -> list:
        """获取对话历史"""
        return self.history.copy()
