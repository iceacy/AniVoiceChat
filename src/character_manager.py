"""角色管理模块 - 外部化角色提示词配置"""
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class CharacterConfig:
    """角色配置"""
    name: str
    character_id: str
    language: str
    system_prompt: str
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterConfig":
        """从字典创建配置"""
        return cls(
            name=data.get("name", ""),
            character_id=data.get("character_id", ""),
            language=data.get("language", "zh"),
            system_prompt=data.get("system_prompt", ""),
            description=data.get("description", ""),
        )


class CharacterManager:
    """角色管理器"""

    def __init__(self, characters_dir: Optional[str] = None):
        """
        初始化角色管理器

        Args:
            characters_dir: 角色配置目录路径，默认从环境变量读取
        """
        if characters_dir is None:
            characters_dir = os.getenv("CHARACTERS_DIR", "characters")

        self.characters_dir = Path(characters_dir)
        self._cache: Dict[str, CharacterConfig] = {}
        self._user_characters_dir = Path.home() / ".anicvoicechat" / "characters"

    def load_character(self, character_id: str) -> Optional[CharacterConfig]:
        """
        加载角色配置

        Args:
            character_id: 角色ID（如 "妮露_JA"）

        Returns:
            CharacterConfig 或 None
        """
        if character_id in self._cache:
            return self._cache[character_id]

        config = self._load_from_file(character_id)
        if config is None:
            config = self._load_user_character(character_id)

        if config:
            self._cache[character_id] = config

        return config

    def _load_from_file(self, character_id: str) -> Optional[CharacterConfig]:
        """从内置角色配置文件加载"""
        config_file = self.characters_dir / f"{character_id}.yaml"

        if not config_file.exists():
            return None

        return self._parse_yaml(config_file)

    def _load_user_character(self, character_id: str) -> Optional[CharacterConfig]:
        """从用户目录加载自定义角色"""
        config_file = self._user_characters_dir / f"{character_id}.yaml"

        if not config_file.exists():
            return None

        return self._parse_yaml(config_file)

    def _parse_yaml(self, config_file: Path) -> Optional[CharacterConfig]:
        """解析YAML配置文件"""
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return CharacterConfig.from_dict(data)
        except ImportError:
            print(f"[ERROR] pyyaml 未安装，请运行: pip install pyyaml")
            return None
        except Exception as e:
            print(f"[ERROR] 解析角色配置失败 {config_file}: {e}")
            return None

    def list_characters(self) -> list[str]:
        """列出所有可用的内置角色"""
        characters = []

        if self.characters_dir.exists():
            for f in self.characters_dir.glob("*.yaml"):
                if f.stem != "default":
                    characters.append(f.stem)

        return sorted(characters)

    def list_user_characters(self) -> list[str]:
        """列出用户自定义角色"""
        characters = []

        if self._user_characters_dir.exists():
            for f in self._user_characters_dir.glob("*.yaml"):
                characters.append(f.stem)

        return sorted(characters)

    def reload_character(self, character_id: str) -> Optional[CharacterConfig]:
        """重新加载角色配置（清除缓存）"""
        if character_id in self._cache:
            del self._cache[character_id]
        return self.load_character(character_id)

    def get_default_character(self) -> str:
        """获取默认角色ID"""
        return os.getenv("DEFAULT_CHARACTER", "妮露_JA")


_global_manager: Optional[CharacterManager] = None


def get_character_manager() -> CharacterManager:
    """获取全局角色管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = CharacterManager()
    return _global_manager
