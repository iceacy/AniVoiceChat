# AniVoiceChat - 动漫角色语音对话系统

基于 Qwen3-ASR + GLM/DeepSeek + GPT-SoVITS 的中日双语实时动漫角色对话系统。

## 系统架构

```
麦克风/音频文件
    ↓
Qwen3-ASR (本地) → 语音识别
    ↓
GLM/DeepSeek API (云端) → 对话生成
    ↓
GPT-SoVITS (本地) → 语音合成
    ↓
扬声器/音频文件
```

## 环境要求

- Python 3.10+
- CUDA 12.x (推荐 12.8)
- RTX 8GB+ 显存
- GLM API Key 或 DeepSeek API Key

## 安装步骤

### 1. 克隆项目

```bash
cd D:/Vscode_Program/AniVoiceChat
```

### 2. 创建虚拟环境

```bash
# 使用 Python 3.12
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM配置 (支持 deepseek 或 glm)
LLM_PROVIDER=glm
GLM_API_KEY=your_glm_api_key_here
# 或者使用DeepSeek
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 模型路径配置
QWEN_ASR_MODEL_PATH=D:/Vscode_Program/AniVoiceChat/models/Qwen/Qwen3-ASR-0_6B
GPT_SOVITS_ROOT=D:/Vscode_Program/GPT-SoVITS
DEFAULT_CHARACTER=妮露_JA

# 设备配置
DEVICE=cuda

# 音频配置
SAMPLE_RATE=16000
CHUNK_SIZE=1024
```

### 5. 下载ASR模型

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-0.6B', cache_dir='D:/Vscode_Program/AniVoiceChat/models')"
```

### 6. 启动GPT-SoVITS TTS服务

在新的终端中进入GPT-SoVITS目录：

```bash
cd D:/Vscode_Program/GPT-SoVITS

# 激活GPT-SoVITS环境（如果使用独立环境）
# runtime\Scripts\activate

# 启动API服务
python api_v2.py -a 127.0.0.1 -p 9880
```

## 使用方法

### 统一CLI入口（推荐）

使用 `anic` 脚本或 `python -m AniVoiceChat`：

```bash
# Windows 用户可以使用快捷脚本
anic.bat <command> [options]

# 或者使用完整命令
python -m AniVoiceChat <command> [options]
```

#### 可用命令

```bash
# 检查配置和依赖
anic.bat check

# 交互式文本对话
anic.bat chat

# 语音对话模式
anic.bat voice

# 测试ASR模块
anic.bat test-asr <audio_file>

# 测试LLM模块
anic.bat test-llm "你好"

# 测试TTS模块
anic.bat test-tts "你好，很高兴见到你"

# 查看帮助
anic.bat --help
anic.bat <command> --help
```

#### 语音对话流程

```bash
anic.bat voice
```

1. 按回车键开始录音
2. 说话
3. 按回车键停止（或自动检测静音）
4. 等待ASR识别 → LLM生成 → TTS合成
5. 自动播放回复语音

#### 交互式文本对话

```bash
anic.bat chat
```

交互式命令：
- 直接输入文本进行对话
- `audio <路径>` - 处理音频文件
- `clear` - 清空对话历史
- `history` - 查看对话历史
- `quit` / `exit` - 退出

#### 命令选项

所有命令都支持 `--character` 选项指定角色：

```bash
anic.bat chat --character 神里绫华_ZH
anic.bat voice --character 妮露_JA
```

### 运行测试

```bash
# 单元测试（需要pytest）
pytest tests/

# 使用CLI进行功能测试
python -m AniVoiceChat test-asr <audio_file>
python -m AniVoiceChat test-llm "你好"
python -m AniVoiceChat test-tts "测试文本"
```

## 项目结构

```
AniVoiceChat/
├── __main__.py                  # 统一CLI入口点
├── anic.bat                     # Windows快捷启动脚本
├── anic.sh                      # Linux/Mac快捷启动脚本
├── cli/                         # CLI命令模块
│   ├── base.py                  # 命令基类
│   ├── chat.py                  # chat命令
│   ├── voice.py                 # voice命令
│   ├── check.py                 # check命令
│   └── test/                    # 测试命令
│       ├── __init__.py
│       ├── asr.py
│       ├── llm.py
│       └── tts.py
├── src/                         # 核心源代码
│   ├── asr/
│   │   └── qwen_asr.py          # Qwen3-ASR 封装
│   ├── llm/
│   │   ├── base.py              # LLM抽象基类
│   │   ├── factory.py           # LLM工厂方法
│   │   ├── deepseek.py          # DeepSeek 实现
│   │   └── glm.py               # GLM 实现
│   ├── tts/
│   │   └── gpt_sovits.py        # GPT-SoVITS 调用
│   ├── audio/
│   │   ├── microphone.py        # 麦克风录音
│   │   ├── player.py            # 音频播放
│   │   └── vad.py               # 语音活动检测
│   ├── config.py                # 配置管理
│   ├── character_manager.py     # 角色管理器
│   ├── logger.py                # 日志系统
│   └── dialogue_pipeline.py     # 对话管道
├── characters/                  # 角色配置目录
│   ├── default.yaml
│   ├── 妮露_JA.yaml
│   └── 神里绫华_ZH.yaml
├── scripts/                     # 诊断工具
│   └── diagnose_tts_params.py
├── tests/                       # pytest单元测试
│   ├── test_audio_module.py
│   ├── test_config_module.py
│   └── test_tts_module.py
├── requirements.txt             # 依赖清单
├── .env.example                 # 配置模板
└── README.md                    # 本文档
```

## 支持的角色

目前内置角色（位于 `characters/` 目录）：

- 妮露_JA (Nilou - 原神)
- 神里绫华_ZH (Ayaka - 原神)

添加自定义角色：在 `characters/` 目录下创建 YAML 配置文件，参考 `妮露_JA.yaml`。

可以通过修改 `.env` 中的 `DEFAULT_CHARACTER` 来切换角色。

## 常见问题

### Q: TTS服务器连接失败？

A: 请确保GPT-SoVITS API服务正在运行：

```bash
cd D:/Vscode_Program/GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880
```

### Q: ASR模型加载失败？

A: 检查模型路径是否正确，确保模型已完整下载：

```bash
ls "D:/Vscode_Program/AniVoiceChat/models/Qwen/"
```

### Q: LLM API调用失败？

A: 检查 `.env` 中的LLM配置：
- 如果使用GLM：检查 `GLM_API_KEY`
- 如果使用DeepSeek：检查 `DEEPSEEK_API_KEY`
- 确认 `LLM_PROVIDER` 设置正确

### Q: 如何获取API密钥？

A:
- **GLM**: 访问 https://open.bigmodel.cn/ 注册并获取API密钥
- **DeepSeek**: 访问 https://platform.deepseek.com/ 注册并获取API密钥

## 性能优化建议

1. **GPU内存优化**: 使用 float16 精度可减少显存占用
2. **批处理**: 调整 `ASR_BATCH_SIZE` 提高吞吐量
3. **流式响应**: 未来可支持流式TTS，降低首字延迟

## 许可证

MIT License

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - 语音识别模型
- [GLM](https://open.bigmodel.cn/) - 智谱AI大语言模型API
- [DeepSeek](https://www.deepseek.com/) - 大语言模型API
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - 语音合成模型
