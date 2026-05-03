"""CLI命令模块"""
from .base import Command
from .chat import ChatCommand
from .voice import VoiceCommand
from .check import CheckCommand

__all__ = ["Command", "ChatCommand", "VoiceCommand", "CheckCommand"]
