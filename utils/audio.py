import librosa
import numpy as np
import torch

from utils.config import MAX_SAMPLES, SAMPLE_RATE


def pad(x, max_len=MAX_SAMPLES):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_audio(path, sr=SAMPLE_RATE):
    waveform, _ = librosa.load(path, sr=sr)
    return waveform


def preprocess(waveform, max_len=MAX_SAMPLES):
    padded = pad(waveform, max_len=max_len)
    return torch.tensor(padded, dtype=torch.float32)
