"""音频模块单元测试"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.audio.microphone import MicrophoneRecorder
from src.audio.player import AudioPlayer


class TestMicrophoneRecorder:
    """麦克风录音器测试"""

    def test_init_default_values(self):
        """测试初始化默认值"""
        recorder = MicrophoneRecorder()

        assert recorder.sample_rate == 16000
        assert recorder.chunk_size == 1024
        assert recorder.channels == 1
        assert recorder.is_recording is False
        assert len(recorder.audio_buffer) == 0

    def test_init_custom_values(self):
        """测试自定义初始化参数"""
        recorder = MicrophoneRecorder(sample_rate=48000, chunk_size=2048)

        assert recorder.sample_rate == 48000
        assert recorder.chunk_size == 2048

    def test_audio_callback_when_recording(self):
        """测试录音时的音频回调"""
        import pyaudio

        recorder = MicrophoneRecorder()
        recorder.is_recording = True

        # 模拟音频数据
        test_audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
        in_data = test_audio.tobytes()

        result = recorder._audio_callback(in_data, 1024, {}, None)

        assert result == (None, pyaudio.paContinue)
        assert len(recorder.audio_buffer) == 1

    def test_audio_callback_when_not_recording(self):
        """测试非录音时的音频回调"""
        recorder = MicrophoneRecorder()
        recorder.is_recording = False

        test_audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
        in_data = test_audio.tobytes()

        recorder._audio_callback(in_data, 1024, {}, None)

        # 不应该添加到缓冲区
        assert len(recorder.audio_buffer) == 0

    def test_stop_recording_when_not_recording(self):
        """测试停止未开始的录音"""
        recorder = MicrophoneRecorder()

        audio_data = recorder.stop_recording()

        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) == 0

    def test_stop_recording_with_buffer(self):
        """测试停止录音并返回数据"""
        recorder = MicrophoneRecorder()

        # 手动添加测试音频
        test_audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
        recorder.audio_buffer.append(test_audio)

        recorder.is_recording = True
        audio_data = recorder.stop_recording()

        assert len(audio_data) == 1024
        assert len(recorder.audio_buffer) == 0

    def test_record_duration(self):
        """测试录音指定时长"""
        recorder = MicrophoneRecorder()

        with patch.object(recorder, 'start_recording'):
            with patch('time.sleep'):
                with patch.object(recorder, 'stop_recording', return_value=np.array([1, 2, 3])):
                    result = recorder.record_duration(1.0)

                    # 验证调用了正确的序列
                    assert result is not None

    def test_cleanup(self):
        """测试资源清理"""
        recorder = MicrophoneRecorder()

        # 模拟录音状态
        recorder.is_recording = True
        recorder.stream = Mock()

        recorder.cleanup()

        assert recorder.is_recording is False
        assert recorder.stream is None


class TestAudioPlayer:
    """音频播放器测试"""

    def test_init_default_values(self):
        """测试初始化默认值"""
        player = AudioPlayer()

        assert player.sample_rate == 16000
        assert player.channels == 1
        assert player.is_playing is False

    def test_init_custom_sample_rate(self):
        """测试自定义采样率"""
        player = AudioPlayer(sample_rate=48000)

        assert player.sample_rate == 48000

    def test_play_empty_audio(self):
        """测试播放空音频"""
        player = AudioPlayer()
        empty_audio = np.array([], dtype=np.int16)

        # 不应该抛出异常
        player.play(empty_audio)

    def test_play_float32_conversion(self):
        """测试 float32 音频格式转换"""
        player = AudioPlayer()

        with patch.object(player, '_play_blocking') as mock_play:
            float_audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)

            player.play(float_audio, blocking=True)

            # 验证调用了播放方法
            mock_play.assert_called_once()

            # 验证音频被转换为 int16
            played_audio = mock_play.call_args[0][0]
            assert played_audio.dtype == np.int16

    def test_play_int16_passthrough(self):
        """测试 int16 音频直接播放"""
        player = AudioPlayer()

        with patch.object(player, '_play_blocking') as mock_play:
            int16_audio = np.array([1000, -1000, 500], dtype=np.int16)

            player.play(int16_audio, blocking=True)

            played_audio = mock_play.call_args[0][0]
            assert played_audio.dtype == np.int16
            # 数据应该保持不变
            assert np.array_equal(played_audio, int16_audio)

    def test_play_non_blocking(self):
        """测试非阻塞播放"""
        player = AudioPlayer()

        with patch('src.audio.player.Thread') as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            audio_data = np.array([1000, -1000], dtype=np.int16)

            player.play(audio_data, blocking=False)

            # 验证创建了线程
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_stop(self):
        """测试停止播放"""
        player = AudioPlayer()
        player.is_playing = True

        player.stop()

        assert player.is_playing is False

    def test_cleanup(self):
        """测试资源清理"""
        player = AudioPlayer()
        player.is_playing = True
        player.stream = Mock()

        player.cleanup()

        assert player.is_playing is False
        assert player.stream is None


class TestAudioBufferOperations:
    """音频缓冲区操作测试"""

    def test_buffer_is_deque(self):
        """测试音频缓冲区是 deque 类型"""
        from collections import deque

        recorder = MicrophoneRecorder()

        assert isinstance(recorder.audio_buffer, deque)

    def test_buffer_append_and_retrieve(self):
        """测试缓冲区添加和检索"""
        recorder = MicrophoneRecorder()

        # 添加多个音频块
        chunk1 = np.random.randint(-1000, 1000, 512, dtype=np.int16)
        chunk2 = np.random.randint(-1000, 1000, 512, dtype=np.int16)

        recorder.audio_buffer.append(chunk1)
        recorder.audio_buffer.append(chunk2)

        assert len(recorder.audio_buffer) == 2

        # 检索第一个
        retrieved = recorder.audio_buffer.popleft()
        assert np.array_equal(retrieved, chunk1)
        assert len(recorder.audio_buffer) == 1

    def test_buffer_thread_safety(self):
        """测试缓冲区线程安全性（deque 是线程安全的）"""
        import threading

        recorder = MicrophoneRecorder()
        results = []

        def append_audio():
            for _ in range(100):
                chunk = np.random.randint(-1000, 1000, 100, dtype=np.int16)
                recorder.audio_buffer.append(chunk)

        threads = [threading.Thread(target=append_audio) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有音频都应该被添加
        assert len(recorder.audio_buffer) == 500
