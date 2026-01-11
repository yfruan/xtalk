import numpy as np
import soundfile as sf
import librosa
import logging
from typing import Optional, Generator
from ..interfaces import TTS

# Append dependency helper
import importlib

# Default output sample rate (Hz) from the TTS service
DEFAULT_SERVER_SAMPLE_RATE = 24000

# Lazily import soxr to avoid raising errors when unused
_soxr_spec = importlib.util.find_spec("soxr")
if _soxr_spec is not None:
    import soxr  # type: ignore
else:
    soxr = None  # type: ignore

# Lazily import mlx_audio to avoid raising errors when unused
_mlx_audio_spec = importlib.util.find_spec("mlx_audio")
if _mlx_audio_spec is not None:
    from mlx_audio.tts.generate import generate_audio
else:
    generate_audio = None  # type: ignore


class MLXCosyVoice3(TTS):
    """
    MLX implementation of CosyVoice3 TTS using mlx-audio-plus.
    Supports synthesis modes including cross-lingual, zero-shot, instruct, and voice conversion.
    """

    def _patch_s3_tokenizer_from_pretrained(self, local_path: str):
        """
        Patch the S3TokenizerV3.from_pretrained method to use a local path instead of downloading from Hugging Face Hub.
        
        Args:
            local_path: Path to the local S3TokenizerV3 directory.
        """
        import mlx.core as mx
        from mlx_audio.codec.models.s3tokenizer.model_v3 import S3TokenizerV3
        from pathlib import Path
        
        original_from_pretrained = S3TokenizerV3.from_pretrained
        
        def patched_from_pretrained(
            cls,
            name: str = "speech_tokenizer_v3",
            repo_id: str = "mlx-community/S3TokenizerV3",
        ) -> "S3TokenizerV3":
            """Patched method that loads from local path instead of Hugging Face Hub."""
            logging.info(f"Using patched S3TokenizerV3.from_pretrained, loading from local path: {local_path}")
            
            path = Path(local_path)
            if not path.exists():
                raise ValueError(f"Local path {local_path} does not exist")
            
            model = S3TokenizerV3(name)
            model_path = path / "model.safetensors"
            if not model_path.exists():
                raise ValueError(f"Model file not found at {model_path}")
            
            weights = mx.load(model_path.as_posix(), format="safetensors")
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
            
            return model
        
        # Apply the patch
        S3TokenizerV3.from_pretrained = classmethod(patched_from_pretrained)
        logging.info("Successfully patched S3TokenizerV3.from_pretrained to use local path")
    
    def __init__(
        self,
        model: str = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
        s3_tokenizer_path: Optional[str] = None,
        mode: str = "cross-lingual",  # Default to cross-lingual as per web reference
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        sample_rate: int = 48000,  # Default to 48kHz for consistency with other TTS implementations
        format: str = "pcm",  # Default PCM 16-bit
        save_audio_to_file: bool = False,  # Enable/disable audio file saving
        audio_output_path: str = "./output_audio.wav",  # Default output path
    ):
        """
        Initialize the MLX CosyVoice3 TTS engine.

        Args:
            model: Model name or path for mlx-audio-plus.
            s3_tokenizer_path: S3 speech tokenizer path (optional, defaults to downloading from Hub).
            mode: Inference mode, supports cross-lingual, zero-shot, instruct, voice_conversion.
            ref_audio: Reference audio path (required for most modes).
            ref_text: Reference text (used in zero-shot mode for better quality).
            instruct_text: Instruction text for style control (used in instruct mode).
            sample_rate: Target output sample rate (default 48000).
            format: Audio format ("pcm" for PCM 16-bit, "" for raw float32).
            save_audio_to_file: Whether to save synthesized audio to a file.
            audio_output_path: Path to save the synthesized audio file.
        """
        if generate_audio is None:
            raise RuntimeError("mlx-audio-plus is required for MLXCosyVoice3, install via 'pip install -U mlx-audio-plus'")
            
        # Suppress all warnings during tokenizer loading to avoid the mistral regex warning
        import warnings
        warnings.filterwarnings("ignore")
        
        # Set environment variables to force offline mode for Hugging Face libraries
        import os
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        self.s3_tokenizer_path = s3_tokenizer_path
        # Set s3_tokenizer_path in environment for the mlx-audio library
        if s3_tokenizer_path:
            os.environ["MLX_AUDIO_S3_TOKENIZER_PATH"] = s3_tokenizer_path
            # Patch S3TokenizerV3.from_pretrained to use local path instead of downloading
            self._patch_s3_tokenizer_from_pretrained(s3_tokenizer_path)
        
        self.model = model
        self.s3_tokenizer_path = s3_tokenizer_path
        self.mode = mode
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.instruct_text = instruct_text
        self._sample_rate = sample_rate
        self.format = format
        self.save_audio_to_file = save_audio_to_file
        self.audio_output_path = audio_output_path

        # Validate mode
        valid_modes = [
            "cross-lingual",
            "zero-shot",
            "instruct",
            "voice_conversion",
        ]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode {mode}, supported modes: {valid_modes}")

        # Validate required parameters for each mode
        if mode in ["cross-lingual", "zero-shot", "instruct", "voice_conversion"]:
            if not ref_audio:
                raise ValueError(f"{mode} mode requires ref_audio parameter")
        
        if mode == "zero-shot":
            if not ref_text:
                logging.warning("zero-shot mode without ref_text may result in lower quality")

    def clone(self):
        return MLXCosyVoice3(
            model=self.model,
            s3_tokenizer_path=self.s3_tokenizer_path,
            mode=self.mode,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            instruct_text=self.instruct_text,
            sample_rate=self.sample_rate,
            format=self.format,
            save_audio_to_file=self.save_audio_to_file,
            audio_output_path=self.audio_output_path,
        )

    def close(self):
        """Close any resources (mlx-audio-plus doesn't require explicit closing)."""
        pass

    @staticmethod
    def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """PCM int16 bytes -> float32 ndarray in range [-1, 1]."""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        # Ensure values are within [-1, 1] range
        return np.clip(audio_float, -1.0, 1.0)

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        """float32 ndarray [-1, 1] -> PCM int16 bytes."""
        # Double clip to ensure safety
        audio_float = np.clip(audio_float, -1.0, 1.0)
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _resample_audio(
        self, audio_float: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio with soxr."""
        if soxr is None:
            raise RuntimeError("soxr is required for resampling, install via 'pip install soxr'")

        # Resample the audio
        resampled_float = soxr.resample(audio_float, orig_sr, target_sr)
        return resampled_float

    def synthesize(
        self,
        text: str,
        mode: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        source_audio: Optional[str] = None,  # For voice conversion
    ) -> bytes:
        """
        Synthesize speech.

        Args:
            text: Text to synthesize.
            mode: Optional mode overriding the default.
            ref_audio: Optional reference audio path overriding the default.
            ref_text: Optional reference text overriding the default (used in zero-shot mode).
            instruct_text: Optional instruction text overriding the default (used in instruct mode).
            source_audio: Source audio path (used in voice_conversion mode).

        Returns:
            bytes: Synthesized audio payload.
        """
        # Apply overrides if provided
        selected_mode = mode or self.mode
        selected_ref_audio = ref_audio or self.ref_audio
        selected_ref_text = ref_text or self.ref_text
        selected_instruct_text = instruct_text or self.instruct_text

        try:
            # Build parameters for mlx-audio.generate_audio
            gen_params = {
                "model": self.model,
                "fix_mistral_regex": True  # Always set fix_mistral_regex=True to avoid tokenizer warning
            }

            if selected_mode == "cross-lingual":
                logging.info("Using cross-lingual synthesis mode")
                gen_params.update({
                    "text": text,
                    "ref_audio": selected_ref_audio,
                })
            elif selected_mode == "zero-shot":
                logging.info("Using zero-shot synthesis mode")
                gen_params.update({
                    "text": text,
                    "ref_audio": selected_ref_audio,
                    "ref_text": selected_ref_text,
                })
            elif selected_mode == "instruct":
                logging.info("Using instruct synthesis mode")
                gen_params.update({
                    "text": text,
                    "ref_audio": selected_ref_audio,
                    "instruct_text": selected_instruct_text,
                })
            elif selected_mode == "voice_conversion":
                logging.info("Using voice conversion mode")
                if not source_audio:
                    raise ValueError("voice_conversion mode requires source_audio parameter")
                gen_params.update({
                    "source_audio": source_audio,
                    "ref_audio": selected_ref_audio,
                })
            else:
                raise ValueError(f"Unsupported mode: {selected_mode}")

            # Generate audio using mlx-audio-plus
            logging.info(f"Generating audio with params: {gen_params}")
            
            # Create a temporary directory to store the generated audio
            import tempfile
            import shutil
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set file_prefix to generate audio in the temporary directory
                temp_file_prefix = os.path.join(temp_dir, "temp_audio")
                gen_params["file_prefix"] = temp_file_prefix
                gen_params["audio_format"] = "wav"  # Use wav format for compatibility
                gen_params["join_audio"] = True  # Join all audio segments into one file
                
                # Call generate_audio - it will save to file but return None
                generate_audio(**gen_params)
                
                # Read the generated audio file back
                temp_audio_path = f"{temp_file_prefix}.wav"
                if not os.path.exists(temp_audio_path):
                    # If the joined file doesn't exist, check for individual segments
                    temp_audio_path = f"{temp_file_prefix}_000.wav"
                
                if not os.path.exists(temp_audio_path):
                    raise RuntimeError(f"TTS synthesis failed: No audio file generated at {temp_audio_path}")
                
                # Read the audio file
                logging.info(f"Reading synthesized audio from {temp_audio_path}")
                audio_float, gen_sample_rate = sf.read(temp_audio_path, dtype=np.float32)
                
                # Ensure audio is 1D
                if len(audio_float.shape) > 1:
                    audio_float = np.mean(audio_float, axis=1)

                # Resample if needed
                if self._sample_rate != gen_sample_rate:
                    logging.info(f"Resampling from {gen_sample_rate}Hz to {self._sample_rate}Hz")
                    audio_float = self._resample_audio(
                        audio_float, 
                        orig_sr=gen_sample_rate, 
                        target_sr=self._sample_rate
                    )

                # Convert to desired format
                if self.format == "pcm":
                    # Convert to PCM 16-bit
                    audio_bytes = self._float32_to_pcm_bytes(audio_float)
                else:
                    # Keep as raw float32
                    audio_bytes = audio_float.tobytes()

            # Save audio to file if enabled
            if self.save_audio_to_file:
                logging.info(f"Saving synthesized audio to {self.audio_output_path}")
                self.save_audio(audio_bytes, self.audio_output_path)

            return audio_bytes
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize_stream(self, text: str, **kwargs) -> Generator[bytes, None, None]:
        """
        Streaming speech synthesis.

        Args:
            text: Text to synthesize.

        Yields:
            bytes: Audio chunks.
        """
        # Note: mlx-audio-plus doesn't support streaming synthesis yet,
        # so we implement this by synthesizing the entire audio and yielding it as a single chunk
        try:
            audio_bytes = self.synthesize(text, **kwargs)
            yield audio_bytes
        except Exception as e:
            logging.error(f"Streaming synthesis failed: {e}")
            raise RuntimeError(f"Streaming synthesis failed: {e}")

    def convert_audio_bytes_to_ndarray(
        self, raw_audio: bytes, format: str = None
    ) -> np.ndarray:
        """Convert raw bytes to numpy array."""
        if not format or format == "":
            return np.frombuffer(raw_audio, dtype=np.float32).reshape(1, -1)
        elif format == "pcm":
            return np.frombuffer(raw_audio, dtype=np.int16).reshape(1, -1)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_audio(self, audio_data: bytes, output_path: str):
        """Persist synthesized audio to a file."""
        try:
            if self.format == "" or self.format is None:
                # Raw float32 format
                audio_array = self.convert_audio_bytes_to_ndarray(audio_data)
                # Transpose to (samples, channels) for soundfile
                sf.write(output_path, audio_array.T, self.sample_rate)
            elif self.format == "pcm":
                # PCM int16 format - convert to float32 in range [-1, 1] before saving
                audio_array_int16 = self.convert_audio_bytes_to_ndarray(audio_data, "pcm")
                # Convert int16 to float32 in range [-1, 1]
                audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
                # Transpose to (samples, channels) for soundfile
                sf.write(output_path, audio_array_float32.T, self.sample_rate)
            else:
                # Write raw bytes directly
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            # Compute duration
            if self.format in ["", None, "pcm"]:
                audio_array = self.convert_audio_bytes_to_ndarray(
                    audio_data, self.format
                )
                duration = audio_array.shape[1] / self.sample_rate
                logging.info(f"Audio saved to {output_path} (duration: {duration:.2f}s)")
        except Exception as e:
            logging.error(f"Failed to save audio: {str(e)}")
            raise RuntimeError(f"Failed to save audio: {str(e)}")

    def set_mode(self, mode: str):
        """Set the default inference mode."""
        valid_modes = [
            "cross-lingual",
            "zero-shot",
            "instruct",
            "voice_conversion",
        ]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode {mode}, supported modes: {valid_modes}")
        self.mode = mode

    def set_ref_audio(self, ref_audio: str):
        """Set the default reference audio path."""
        self.ref_audio = ref_audio

    def set_ref_text(self, ref_text: str):
        """Set the default reference text."""
        self.ref_text = ref_text

    def set_instruct_text(self, instruct_text: str):
        """Set the default instruction text."""
        self.instruct_text = instruct_text

    def set_format(self, format: str):
        """Set the default audio format."""
        if format not in ["", "pcm"]:
            raise ValueError(f"Unsupported audio format {format}, allowed: '', 'pcm'")
        self.format = format
