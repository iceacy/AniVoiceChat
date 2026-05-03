"""音频播放模块"""
import pyaudio
import numpy as np
from threading import Thread
from typing import Optional


class AudioPlayer:
    """音频播放器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.channels = 1
        self.format = pyaudio.paInt16

        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_playing = False

    def __enter__(self):
        """进入上下文时初始化 PyAudio"""
        if self.audio is None:
            import pyaudio
            self.audio = pyaudio.PyAudio()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动清理资源"""
        self.cleanup()
        return False

    def play(self, audio_data: np.ndarray, blocking: bool = True):
        """
        播放音频

        Args:
            audio_data: 音频数据（float32或int16）
            blocking: 是否阻塞等待播放完成
        """
        if audio_data.size == 0:
            return

        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        if blocking:
            self._play_blocking(audio_data)
        else:
            thread = Thread(target=self._play_blocking, args=(audio_data,))
            thread.start()

    def _play_blocking(self, audio_data: np.ndarray):
        """阻塞播放音频"""
        self.is_playing = True

        # 确保 PyAudio 已初始化（兼容非 with 语句使用）
        if self.audio is None:
            import pyaudio
            self.audio = pyaudio.PyAudio()

        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
            )

            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                if not self.is_playing:
                    break
                chunk = audio_data[i:i + chunk_size]

                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

                self.stream.write(chunk.tobytes())

        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self.is_playing = False

    def stop(self):
        """停止播放"""
        self.is_playing = False

    def cleanup(self):
        """清理资源"""
        self.stop()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None
