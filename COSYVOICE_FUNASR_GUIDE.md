# CosyVoice 和 FunASR 模型使用指南

## 1. CosyVoice 运行指南

### 1.1 CosyVoice 简介

CosyVoice 是一种高质量的文本转语音（TTS）模型，支持多种合成模式和语音风格。在 X-Talk 项目中，提供了两种 CosyVoice 实现：

- `CosyVoice`：基于阿里云 DashScope API 的在线 TTS 服务
- `CosyVoiceLocal`：基于本地 gRPC 服务的离线 TTS 实现

### 1.2 运行在线 CosyVoice

#### 1.2.1 配置要求

- 有效的阿里云 DashScope API 密钥
- 安装依赖：
  ```bash
  pip install dashscope
  ```

#### 1.2.2 使用方法

在配置文件中设置 TTS 为 CosyVoice：

```json
{
  "tts": {
    "type": "CosyVoice",
    "params": {
      "api_key": "your-dashscope-api-key",
      "model": "cosyvoice-v3-flash",
      "voice": "longanyang"
    }
  }
}
```

### 1.3 运行本地 CosyVoice

#### 1.3.1 配置要求

- 安装 CosyVoice 依赖和模型
- 启动 CosyVoice gRPC 服务

#### 1.3.2 安装步骤

1. 克隆 CosyVoice 仓库：
   ```bash
   git clone https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 下载 CosyVoice 模型（例如 Fun-CosyVoice3-0.5B-2512）：
   ```bash
   # 通过 ModelScope 下载
   python -m modelscope.hub.snapshot_download --model-id=iic/CosyVoice-300M --local_dir=./models/CosyVoice-300M
   ```

4. 启动 gRPC 服务：
   ```bash
   python -m cosyvoice.cli.server --model_dir ./models/CosyVoice-300M --port 50000
   ```

   ```bash
   conda activate cosyvoice && cd /Users/yfruan/Documents/Workspace/CosyVoice/runtime/python/grpc && python server.py --model_dir /Users/yfruan/Documents/Workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512
  ```

#### 1.3.3 在 X-Talk 中配置

在 `local_config.json` 中设置本地 CosyVoice：

```json
{
  "tts": {
    "type": "CosyVoiceLocal",
    "params": {
      "host": "localhost",
      "port": 50000,
      "mode": "zero_shot_by_spk_id",
      "spk_id": "001",
      "sample_rate": 48000
    }
  }
}
```

## 2. FunASR 模型下载指南

### 2.1 FunASR 简介

FunASR 是一个开源的语音识别工具包，提供了多种高性能的 ASR 模型。在 X-Talk 项目中，主要使用 SenseVoiceSmall 模型。

### 2.2 下载 SenseVoiceSmall 模型

#### 2.2.1 通过 Hugging Face 下载

```bash
# 直接使用 FunASR 的 AutoModel 会自动下载
python -c "from funasr import AutoModel; AutoModel(model='FunAudioLLM/SenseVoiceSmall')"
```

#### 2.2.2 通过 ModelScope 下载（推荐，速度更快）

```bash
# 使用 ModelScope CLI 下载
python -m modelscope.hub.snapshot_download --model-id=iic/SenseVoiceSmall --local_dir=./models/SenseVoiceSmall
```

或者在代码中指定 hub 为 'ms'：

```python
from funasr import AutoModel

# 从 ModelScope 下载并加载模型
model = AutoModel(
    model='iic/SenseVoiceSmall',
    hub='ms',
    device='cpu'
)
```

### 2.3 在 X-Talk 中配置 FunASR

在 `local_config.json` 中设置 ASR 为 SenseVoiceSmallLocal：

```json
{
  "asr": {
    "type": "SenseVoiceSmallLocal",
    "params": {
      "model": "SenseVoiceSmall",
      "device": "cpu",
      "hub": "ms",
      "language": "auto"
    }
  }
}
```

## 3. 常见问题

### 3.1 CosyVoice 相关

- **问题**：gRPC 连接失败
  **解决**：确保 CosyVoice 服务已启动，端口配置正确

- **问题**：模型下载慢
  **解决**：使用 ModelScope 下载，或配置国内镜像源

### 3.2 FunASR 相关

- **问题**：模型未找到
  **解决**：检查模型名称是否正确，确保已下载到正确路径

- **问题**：识别效果差
  **解决**：尝试调整语言设置，或使用更高精度的模型

## 4. 参考链接

- [CosyVoice 官方仓库](https://github.com/FunAudioLLM/CosyVoice)
- [FunASR 官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [ModelScope 模型仓库](https://modelscope.cn/)
- [Hugging Face 模型仓库](https://huggingface.co/)
