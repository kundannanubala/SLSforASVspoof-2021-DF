import os

from torch.utils.data import Dataset

from utils.audio import load_audio, preprocess
from utils.config import AUDIO_EXTENSIONS


def collect_audio_files(input_dir):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root, filename))
    return audio_files


class InputFolderDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.file_paths = collect_audio_files(input_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        waveform = load_audio(path)
        tensor = preprocess(waveform)
        filename = os.path.basename(path)
        return tensor, filename
