from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

def features(waveform, sample_rate):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    waveform = waveform[:, :SAMPLE_RATE]
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64)(waveform)
    return torch.log(mel + 1e-6)

class Commands(Dataset):
    def __init__(self, root, classes, subset):
        self.root = Path(root)
        self.classes = list(classes)
        split = self.root / ('validation_list.txt' if subset == 'validation' else 'testing_list.txt')
        holdouts = set()
        for name in ('validation_list.txt', 'testing_list.txt'):
            holdouts.update((self.root / name).read_text().splitlines())
        if subset == 'training':
            self.files = [f for c in classes for f in sorted((self.root / c).glob('*.wav')) if f.relative_to(self.root).as_posix() not in holdouts]
        else:
            selected = set(split.read_text().splitlines())
            self.files = [self.root / name for name in sorted(selected) if name.split('/')[0] in classes]
        if not self.files:
            raise RuntimeError(f'No audio files in {self.root}; check dataset extraction')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        audio, sr = torchaudio.load(str(file))
        return features(audio, sr), self.classes.index(file.parent.name)
