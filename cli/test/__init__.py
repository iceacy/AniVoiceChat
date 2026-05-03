"""测试命令模块"""
from .asr import TestASRCommand
from .llm import TestLLMCommand
from .tts import TestTTSCommand

__all__ = ["TestASRCommand", "TestLLMCommand", "TestTTSCommand"]
