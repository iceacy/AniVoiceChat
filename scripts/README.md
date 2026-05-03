# 工具脚本目录

此目录包含独立的诊断工具，用于手动运行和调试。

## 脚本说明

| 脚本 | 用途 | 运行方式 |
|------|------|---------|
| `diagnose_tts_params.py` | 查看 TTS 调用参数详情 | `python scripts/diagnose_tts_params.py` |

## 与其他测试的区别

- **scripts/**: 诊断工具脚本，手动运行，打印详细信息
- **tests/**: 单元测试，使用 pytest 框架，包含断言和 mock
- **cli/test/**: 通过 CLI 调用的测试命令，快速功能验证

## CLI测试命令

使用统一CLI进行快速测试：

```bash
# 测试各个模块
python -m AniVoiceChat test-asr <audio_file>
python -m AniVoiceChat test-llm "测试文本"
python -m AniVoiceChat test-tts "测试文本"
```
