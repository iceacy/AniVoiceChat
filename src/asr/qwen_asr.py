"""Qwen3-ASR模块封装"""
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional, Union
from qwen_asr import Qwen3ASRModel
from ..config import ASRConfig
from ..logger import get_logger

logger = get_logger(__name__)


class QwenASR:
    """Qwen3-ASR 语音识别类"""

    def __init__(self, config: Optional[ASRConfig] = None):
        if config is None:
            config = ASRConfig.from_env()

        self.config = config
        self.model: Optional[Qwen3ASRModel] = None

    def load(self) -> bool:
        """加载模型"""
        if self.model is not None:
            return True

        if not self.config.model_path or not Path(self.config.model_path).exists():
            logger.error(f"模型路径不存在: {self.config.model_path}")
            return False

        try:
            logger.info(f"加载ASR模型: {self.config.model_path}")
            self.model = Qwen3ASRModel.from_pretrained(
                self.config.model_path,
                dtype=torch.float16,
                device_map=self.config.device,
                max_inference_batch_size=self.config.max_inference_batch_size,
                max_new_tokens=self.config.max_new_tokens,
            )
            logger.info("ASR模型加载成功")
            return True
        except Exception as e:
            logger.error(f"ASR模型加载失败: {e}")
            return False

    def transcribe(
        self,
        audio: Union[str, np.ndarray, Path],
        language: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        """
        语音转文字

        Args:
            audio: 音频文件路径或numpy数组
            language: 语言代码 ("zh", "en", "ja", "yue", None=自动)

        Returns:
            (识别文本, 语言) 或 None
        """
        if self.model is None:
            if not self.load():
                return None

        temp_file = None
        original_audio = audio

        if isinstance(audio, np.ndarray):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            temp_path = temp_file.name

            try:
                from scipy.io import wavfile
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio = (audio * 32767).astype(np.int16)
                elif audio.dtype != np.int16:
                    audio = audio.astype(np.int16)

                wavfile.write(temp_path, 16000, audio)
                audio = temp_path
            except Exception as e:
                logger.error(f"保存临时音频文件失败: {e}")
                if temp_file and os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None

        try:
            results = self.model.transcribe(
                audio=audio,
                language=language,
                return_time_stamps=False,
            )

            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                return result.text, result.language
            else:
                logger.warning(f"识别结果格式异常: {results}")
                return None

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        """识别音频文件"""
        if not Path(audio_path).exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        return self.transcribe(audio_path, language)
