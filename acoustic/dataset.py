import os
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, root: Path, train: bool = True):
        self.mels_dir  = root / "mels"
        self.units_dir = root / "units-pitdyn"

        mel_fps = [Path(os.path.join(self.mels_dir, fn)) for fn in os.listdir(self.mels_dir)]
        cp = int(len(mel_fps) * 0.9)
        if train: mel_fps = mel_fps[:cp]
        else:     mel_fps = mel_fps[cp:]

        self.metadata = [path.relative_to(self.mels_dir) for path in mel_fps]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path   = self.mels_dir  / path
        units_path = self.units_dir / path

        mel  = np.load(mel_path  .with_suffix(".npy")).T
        unit = np.load(units_path.with_suffix(".npy"))

        length = 2 * unit.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        unit = torch.from_numpy(unit)

        return mel, unit


    def pad_collate(self, batch):
        mels, units = zip(*batch)
        mels, units = list(mels), list(units)

        mels_lengths  = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0)     for x in units])

        mels  = pad_sequence(mels,  batch_first=True)
        units = pad_sequence(units, batch_first=True, padding_value=0)

        return mels, mels_lengths, units, units_lengths
