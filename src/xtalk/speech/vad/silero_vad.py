"""
Silero VAD wrapper that mimics webrtcvad.Vad API (`is_speech`).
It keeps an internal VADIterator so it can be called chunk-by-chunk.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from ..interfaces import VAD


class SileroVAD(VAD):
    def __init__(self, threshold: float = 0.5) -> None:
        # load jit model from torch hub
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        self._model = model

        self.threshold = threshold
        self.window_samples = 512

    def is_speech(self, frame: bytes) -> bool:
        # int16 PCM ➜ float32 tensor
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        wav = torch.from_numpy(pcm).unsqueeze(0)  # [1, T]

        # Feed window_samples-sized chunks to VADIterator
        num_samples = self.window_samples
        prob: float = 0.0  # probability of the last processed chunk

        # Iterate over the waveform using fixed windows
        for start in range(0, wav.shape[1], num_samples):
            chunk = wav[:, start : start + num_samples]
            if chunk.shape[1] < num_samples:
                break
            # VADIterator returns speech probability for this chunk
            prob = float(self._model(chunk.squeeze(0), 16000).item())

        # Use the probability of the last full chunk as the speech decision
        return prob >= self.threshold
