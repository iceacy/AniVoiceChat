"""麦克风录音模块"""
import pyaudio
import numpy as np
from threading import Thread, Event
from typing import Optional, Callable
from collections import deque

from .vad import VADSilenceDetector, create_default_vad


class MicrophoneRecorder:
    """麦克风录音器"""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.format = pyaudio.paInt16

        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.stop_event = Event()

        self.audio_buffer = deque()
        self.recording_thread: Optional[Thread] = None

    def __enter__(self):
        """进入上下文时初始化 PyAudio"""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理资源"""
        self.cleanup()
        return False

    def start_recording(self, callback: Optional[Callable[[np.ndarray], None]] = None):
        """开始录音"""
        if self.is_recording:
            return

        # 确保 PyAudio 已初始化（兼容非 with 语句使用）
        if self.audio is None:
            self.audio = pyaudio.PyAudio()

        self.is_recording = True
        self.stop_event.clear()
        self.audio_buffer.clear()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        self.stream.start_stream()

        if callback:
            self.recording_thread = Thread(target=self._process_audio, args=(callback,))
            self.recording_thread.start()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_buffer.append(audio_data)
        return (None, pyaudio.paContinue)

    def _process_audio(self, callback: Callable[[np.ndarray], None]):
        """处理音频数据"""
        while not self.stop_event.is_set() and self.is_recording:
            if len(self.audio_buffer) > 0:
                chunk = self.audio_buffer.popleft()
                callback(chunk)

    def stop_recording(self) -> np.ndarray:
        """停止录音并返回音频数据"""
        if not self.is_recording:
            return np.array([], dtype=np.int16)

        self.is_recording = False
        self.stop_event.set()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.recording_thread:
            self.recording_thread.join(timeout=1)
            self.recording_thread = None

        audio_data = np.concatenate(list(self.audio_buffer)) if self.audio_buffer else np.array([], dtype=np.int16)
        self.audio_buffer.clear()

        return audio_data

    def record_duration(self, duration: float) -> np.ndarray:
        """录音指定时长"""
        self.start_recording()

        import time
        time.sleep(duration)

        return self.stop_recording()

    def record_until_silence(
        self,
        max_duration: float = 30.0,
        silence_threshold: float = 0.02,
        silence_duration: float = 1.5,
        min_duration: float = 0.5,
    ) -> np.ndarray:
        """
        录音直到检测到静音

        Args:
            max_duration: 最大录音时长
            silence_threshold: 静音阈值（0-1）
            silence_duration: 静音持续时间（秒）
            min_duration: 最小录音时长
        """
        import time

        self.start_recording()

        # 创建VAD检测器
        vad = VADSilenceDetector(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        start_time = time.time()
        check_interval = 0.05

        while time.time() - start_time < max_duration:
            time.sleep(check_interval)

            should_stop, reason = vad.should_stop(self.audio_buffer)
            if should_stop:
                return self.stop_recording()

        return self.stop_recording()

    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_recording()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None
