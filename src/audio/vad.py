"""语音活动检测(VAD)模块"""
import numpy as np
from collections import deque
from typing import Optional, Callable


class VADSilenceDetector:
    """基于能量阈值的静音检测器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        silence_threshold: float = 0.02,
        silence_duration: float = 1.5,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ):
        """
        初始化静音检测器

        Args:
            sample_rate: 采样率
            chunk_size: 音频块大小
            silence_threshold: 静音阈值（0-1），低于此值视为静音
            silence_duration: 静音持续时间（秒），连续静音超过此时长触发停止
            min_duration: 最小录音时长（秒），在此之前不会触发静音停止
            max_duration: 最大录音时长（秒）
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_duration = min_duration
        self.max_duration = max_duration

        # 计算块数阈值
        self.silence_chunks = int(silence_duration * sample_rate / chunk_size)
        self.min_chunks = int(min_duration * sample_rate / chunk_size)
        self.max_chunks = int(max_duration * sample_rate / chunk_size)

        # 状态跟踪
        self.silent_count = 0
        self.total_chunks = 0
        self.last_buffer_size = 0

    def reset(self):
        """重置检测器状态"""
        self.silent_count = 0
        self.total_chunks = 0
        self.last_buffer_size = 0

    def start(self):
        """开始检测（重置状态）"""
        self.reset()

    def compute_energy(self, chunk: np.ndarray) -> float:
        """
        计算音频块能量

        Args:
            chunk: 音频数据（int16格式）

        Returns:
            能量值（0-1范围）
        """
        return float(np.abs(chunk.astype(np.float32) / 32768.0).mean())

    def is_silence(self, chunk: np.ndarray) -> bool:
        """
        判断音频块是否为静音

        Args:
            chunk: 音频数据

        Returns:
            True if silence, False otherwise
        """
        energy = self.compute_energy(chunk)
        return energy < self.silence_threshold

    def should_stop(self, audio_buffer: deque) -> tuple[bool, str]:
        """
        检查是否应该停止录音

        Args:
            audio_buffer: 音频缓冲区（deque）

        Returns:
            (should_stop, reason): 是否停止及原因
        """
        # 获取新增的块数
        current_buffer_size = len(audio_buffer)
        chunks_added = current_buffer_size - self.last_buffer_size

        if chunks_added <= 0:
            return False, ""

        # 转换为列表进行检测，不消耗原始缓冲区
        buffer_list = list(audio_buffer)

        # 检测新增的块
        for i in range(max(0, len(buffer_list) - chunks_added), len(buffer_list)):
            if i >= len(buffer_list):
                break

            chunk = buffer_list[i]
            self.total_chunks += 1

            # 只有超过最小时长后才开始检测静音
            if self.total_chunks >= self.min_chunks:
                if self.is_silence(chunk):
                    self.silent_count += 1
                    if self.silent_count >= self.silence_chunks:
                        return True, "检测到静音"
                else:
                    self.silent_count = 0

            # 检查最大时长
            if self.total_chunks >= self.max_chunks:
                return True, "达到最大时长"

        self.last_buffer_size = current_buffer_size
        return False, ""


class VADRecorder:
    """集成VAD的录音控制器"""

    def __init__(
        self,
        detector: VADSilenceDetector,
        on_audio_callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        初始化VAD录音控制器

        Args:
            detector: 静音检测器
            on_audio_callback: 音频数据回调函数
        """
        self.detector = detector
        self.on_audio_callback = on_audio_callback
        self._start_time = None

    def start(self):
        """开始录音检测"""
        import time
        self.detector.reset()
        self._start_time = time.time()

    def should_stop(self, audio_buffer: deque) -> tuple[bool, str]:
        """
        检查是否应该停止录音

        Args:
            audio_buffer: 音频缓冲区

        Returns:
            (should_stop, reason)
        """
        import time

        # 检查最大时长
        if self._start_time and time.time() - self._start_time > self.detector.max_duration:
            return True, "达到最大时长"

        # 使用检测器判断
        return self.detector.should_stop(audio_buffer)


def create_default_vad(sample_rate: int = 16000, chunk_size: int = 1024) -> VADSilenceDetector:
    """
    创建默认配置的VAD检测器

    Args:
        sample_rate: 采样率
        chunk_size: 音频块大小

    Returns:
        VADSilenceDetector实例
    """
    return VADSilenceDetector(
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        silence_threshold=0.02,
        silence_duration=1.5,
        min_duration=0.5,
        max_duration=30.0,
    )
