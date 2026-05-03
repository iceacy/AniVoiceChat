"""LLM 抽象基类"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class BaseLLM(ABC):
    """LLM 提供商抽象接口"""

    def __init__(self, config):
        self.config = config
        self.system_prompt = ""

    @abstractmethod
    def chat(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """
        对话生成

        Args:
            user_input: 用户输入文本
            history: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]

        Returns:
            模型回复文本
        """
        pass

    @abstractmethod
    def set_character(self, character: str) -> None:
        """设置角色提示词"""
        pass

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def chat_with_history(self, user_input: str, history: list) -> tuple:
        """
        带历史记录的对话，自动更新历史

        Args:
            user_input: 用户输入
            history: 对话历史

        Returns:
            (回复文本, 更新后的历史)
        """
        response = self.chat(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return response, history
