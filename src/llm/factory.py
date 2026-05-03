"""LLM 工厂"""
from typing import Optional
from ..config import LLMConfig
from .base import BaseLLM
from .deepseek import DeepSeekLLM
from .glm import GLMLLM


# 提供商注册表
LLM_PROVIDERS = {
    "deepseek": DeepSeekLLM,
    "glm": GLMLLM,
}


def create_llm(config: Optional[LLMConfig] = None) -> BaseLLM:
    """
    创建 LLM 实例

    Args:
        config: LLM 配置，默认从环境变量加载

    Returns:
        LLM 实例

    Raises:
        ValueError: 当提供商不支持时
    """
    if config is None:
        config = LLMConfig.from_env()

    provider = config.provider

    if provider not in LLM_PROVIDERS:
        supported = list(LLM_PROVIDERS.keys())
        raise ValueError(f"不支持的 LLM 提供商: {provider}，支持: {supported}")

    llm_class = LLM_PROVIDERS[provider]
    return llm_class(config)


def register_provider(name: str, llm_class: type) -> None:
    """
    注册新的 LLM 提供商

    Args:
        name: 提供商名称
        llm_class: LLM 实现类（需继承 BaseLLM）
    """
    if not issubclass(llm_class, BaseLLM):
        raise TypeError(f"{llm_class.__name__} 必须继承 BaseLLM")

    LLM_PROVIDERS[name] = llm_class
