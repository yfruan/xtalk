from typing import Optional
from ..interfaces import ASR
import numpy as np
import io
import wave


class MLXFunASR(ASR):
    """ASR implementation using mlx-audio-plus with Fun-ASR-Nano-2512-4bit model."""

    def __init__(self, model_id: str = "mlx-community/Fun-ASR-Nano-2512-4bit", **kwargs):
        """Initialize the MLX FunASR model.

        Args:
            model_id: The model ID to use, default is "mlx-community/Fun-ASR-Nano-2512-4bit"
                     This can be a local path, Hugging Face repository ID, or ModelScope ID/URL
            **kwargs: Additional parameters to pass to Model.from_pretrained
        """
        from mlx_audio.stt.models.funasr import Model
        import os
        import tempfile
        self.model_id = model_id
        
        # Handle ModelScope ID/URL
        if model_id.startswith("https://www.modelscope.cn/models/"):
            # Convert ModelScope URL to model ID
            # https://www.modelscope.cn/models/mlx-community/Fun-ASR-Nano-2512-4bit
            # -> mlx-community/Fun-ASR-Nano-2512-4bit
            modelscope_id = model_id.replace("https://www.modelscope.cn/models/", "")
            
            # Download model from ModelScope
            from modelscope import snapshot_download
            model_path = snapshot_download(
                model_id=modelscope_id,
                cache_dir=kwargs.get("cache_dir", os.path.join(tempfile.gettempdir(), "modelscope"))
            )
        elif "/" in model_id and not os.path.exists(model_id):
            # Assume it's a Hugging Face model ID
            model_path = model_id
        else:
            # Assume it's a local path
            model_path = model_id
        
        # 加载模型
        import mlx.core as mx
        kwargs_copy = kwargs.copy()
        
        # 加载模型到指定设备
        self.model = Model.from_pretrained(model_path, **kwargs_copy)
        
        # 初始化流式识别状态
        self._stream_audio = bytearray()
        self._recognized_text = ""

    def recognize(self, audio: bytes) -> str:
        """Recognize speech from audio bytes.

        Args:
            audio: Audio data in PCM 16-bit mono format (16kHz)

        Returns:
            Recognized text
        """
        if not audio:
            return ""
            
        # Convert PCM bytes to numpy array
        import numpy as np
        import mlx.core as mx
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 默认使用GPU执行推理
        try:
            # 直接使用GPU设备
            result = self.model.generate(audio_array)
        except Exception as e:
            # 如果GPU推理失败，回退到CPU
            print(f"GPU推理失败，回退到CPU: {e}")
            cpu_device = mx.Device("cpu")
            original_device = mx.default_device()
            try:
                mx.set_default_device(cpu_device)
                result = self.model.generate(audio_array)
            finally:
                mx.set_default_device(original_device)
        
        return result.text
    
    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        """Incremental streaming interface.

        Args:
            audio: Audio data in bytes
            is_final: Whether this is the final chunk

        Returns:
            Partially recognized text
        """
        # 缓冲音频数据
        if audio:
            self._stream_audio.extend(audio)
        
        # 如果是最终块，处理所有缓冲的音频
        if is_final and self._stream_audio:
            self._recognized_text = self.recognize(bytes(self._stream_audio))
            self._stream_audio.clear()
            return self._recognized_text
        
        # 对于非最终块，每积累一定量的音频进行一次识别
        # 这里使用较小的块大小，以便更快地返回结果
        if len(self._stream_audio) >= 48000:  # 约3秒音频 (16000Hz * 2字节 * 3秒)
            self._recognized_text = self.recognize(bytes(self._stream_audio))
        
        return self._recognized_text

    def reset(self) -> None:
        """Reset internal recognition state."""
        self._stream_audio.clear()
        self._recognized_text = ""

    def clone(self) -> "MLXFunASR":
        """Clone the ASR instance with shared weights and separate state."""
        return MLXFunASR(self.model_id)
