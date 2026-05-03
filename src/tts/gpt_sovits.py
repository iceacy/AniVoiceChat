"""GPT-SoVITS TTS模块"""
import requests
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from ..config import TTSConfig
from ..logger import get_logger

logger = get_logger(__name__)


class GPTSoVITS:
    """GPT-SoVITS TTS调用类"""

    def __init__(self, config: Optional[TTSConfig] = None):
        if config is None:
            config = TTSConfig.from_env()

        self.config = config
        self.api_url = config.api_url

    def synthesize(
        self,
        text: str,
        text_lang: str = "auto",
        ref_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_lang: Optional[str] = None,
        speed_factor: Optional[float] = None,
        break_step: Optional[int] = None,
    ) -> Optional[tuple[np.ndarray, int]]:
        """
        语音合成

        Args:
            text: 要合成的文本
            text_lang: 文本语言 ("zh", "ja", "en", "auto")
            ref_audio_path: 参考音频路径
            prompt_text: 参考音频对应的文本
            prompt_lang: 参考音频语言
            speed_factor: 语速系数 (1.0=正常, >1.0=更快, <1.0=更慢, None=使用配置默认值)
            break_step: 标点停顿时长 (ms, None=使用配置默认值)

        Returns:
            (音频数据numpy数组, 采样率) 或 None
        """
        ref_audio = ref_audio_path or self.config.ref_audio_path
        prompt_t = prompt_text or self.config.prompt_text
        prompt_l = prompt_lang or self.config.prompt_lang

        # 使用配置中的默认值
        speed = speed_factor if speed_factor is not None else self.config.speed_factor
        break_step_val = break_step if break_step is not None else self.config.break_step

        if not ref_audio:
            logger.error("参考音频未指定")
            return None

        if text_lang == "auto":
            text_lang = self._detect_language(text)

        payload = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_t,
            "prompt_lang": prompt_l,
            "speed_factor": speed,
            "break_step": break_step_val,
        }

        logger.info(f"[TTS] 调用参数: text_lang={text_lang}, prompt_lang={prompt_l}, ref_audio={ref_audio}")

        try:
            response = requests.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                audio_bytes = response.content
                audio_data = self._decode_wav(audio_bytes)
                return audio_data, self.config.sample_rate
            else:
                logger.error(f"TTS请求失败: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到TTS服务器: {self.api_url}")
            logger.info("请确保GPT-SoVITS API服务器正在运行")
            return None
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            return None

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

    def _decode_wav(self, wav_bytes: bytes) -> np.ndarray:
        """解码WAV音频数据"""
        import io
        from scipy.io import wavfile

        audio_io = io.BytesIO(wav_bytes)
        actual_sample_rate, audio_data = wavfile.read(audio_io)

        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # 采样率转换
        if actual_sample_rate != self.config.sample_rate:
            logger.info(f"采样率转换: {actual_sample_rate} Hz -> {self.config.sample_rate} Hz")
            audio_data = self._resample(audio_data, actual_sample_rate, self.config.sample_rate)

        return audio_data

    def _resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样音频数据"""
        from scipy import signal

        # 计算重采样比例
        ratio = target_sr / orig_sr

        # 计算新的长度
        new_length = int(len(audio_data) * ratio)

        # 使用FFT进行重采样（质量更好）
        resampled = signal.resample(audio_data, new_length)

        return resampled

    def save_audio(self, audio_data: np.ndarray, output_path: str) -> bool:
        """保存音频到文件"""
        try:
            from scipy.io import wavfile

            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(output_path, self.config.sample_rate, audio_int16)
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False

    def check_server(self) -> bool:
        """检查TTS服务器是否可用"""
        try:
            response = requests.get(f"{self.api_url}/control", params={"command": "status"}, timeout=5)
            return response.status_code in [200, 400]
        except:
            return False

    def set_model(self, gpt_path: str, sovits_path: str) -> bool:
        """
        切换GPT和SoVITS模型

        Args:
            gpt_path: GPT模型文件路径（相对于GPT-SoVITS根目录）
            sovits_path: SoVITS模型文件路径（相对于GPT-SoVITS根目录）

        Returns:
            是否成功切换
        """
        try:
            # 切换GPT模型
            gpt_response = requests.get(
                f"{self.api_url}/set_gpt_weights",
                params={"weights_path": gpt_path},
                timeout=10,
            )

            if gpt_response.status_code != 200:
                logger.error(f"切换GPT模型失败: {gpt_response.text}")
                return False

            # 切换SoVITS模型
            sovits_response = requests.get(
                f"{self.api_url}/set_sovits_weights",
                params={"weights_path": sovits_path},
                timeout=10,
            )

            if sovits_response.status_code != 200:
                logger.error(f"切换SoVITS模型失败: {sovits_response.text}")
                return False

            logger.info(f"模型切换成功: GPT={gpt_path}, SoVITS={sovits_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"切换模型时网络错误: {e}")
            return False
        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            return False
