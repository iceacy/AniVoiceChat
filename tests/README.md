# 单元测试

此目录包含使用 pytest 框架的单元测试。

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定测试文件
```bash
pytest tests/test_config_module.py
pytest tests/test_tts_module.py
pytest tests/test_audio_module.py
```

### 运行特定测试类或函数
```bash
pytest tests/test_config_module.py::TestLLMConfig::test_glm_provider
```

### 查看详细输出
```bash
pytest tests/ -v
```

### 显示打印输出
```bash
pytest tests/ -s
```

### 生成覆盖率报告
```bash
pytest tests/ --cov=src --cov-report=html
```

## 测试文件说明

| 文件 | 测试模块 |
|------|---------|
| `test_config_module.py` | 配置管理模块 |
| `test_tts_module.py` | TTS 语音合成模块 |
| `test_audio_module.py` | 音频录制和播放模块 |
| `conftest.py` | pytest 配置和共享 fixtures |

## 编写新测试

1. 在对应的测试文件中添加测试函数
2. 使用 `@pytest.fixture` 创建可复用的测试数据
3. 使用 `assert` 验证预期行为
4. 使用 `unittest.mock` mock 外部依赖

### 测试示例

```python
def test_example():
    """测试示例"""
    # Arrange（准备）
    expected = 42

    # Act（执行）
    result = calculate()

    # Assert（断言）
    assert result == expected
```

## 与 scripts/ 的区别

- **tests/**: 使用 pytest 的单元测试，自动运行，包含断言和 mock
- **scripts/**: 独立测试脚本，手动运行，用于调试和诊断
