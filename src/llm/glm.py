"""GLM LLM 实现"""
from openai import OpenAI
from typing import Optional
import os
from .base import BaseLLM
from ..config import LLMConfig, CharacterPrompt
from ..logger import get_logger

logger = get_logger(__name__)


class GLMLLM(BaseLLM):
    """GLM LLM 实现"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig.from_env()

        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.model
        # 从环境变量读取默认角色
        default_character = os.getenv("DEFAULT_CHARACTER", "妮露_JA")
        self.character = default_character
        self.system_prompt = CharacterPrompt.get_prompt(default_character)

    def chat(self, user_input: str, history: Optional[list] = None) -> str:
        """
        对话生成

        Args:
            user_input: 用户输入文本
            history: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]

        Returns:
            模型回复文本
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GLM API 调用失败: {e}")
            return "抱歉，我现在无法回复。"

    def set_character(self, character: str) -> None:
        """设置角色"""
        self.system_prompt = CharacterPrompt.get_prompt(character)
