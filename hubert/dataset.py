import random
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import audio as A


class AcousticUnitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        label_rate: int = 50,
        min_samples: int = 32000,
        max_samples: int = 250000,
        train=True,
        embds_dn = "pits",
    ):
        self.wavs_dir = root / "wavs"
        self.embds_dir = root / embds_dn

        # train-valid split
        self.metadata = [path for path in self.wavs_dir.rglob('*.wav')]
        cp = int(len(self.metadata) * 0.9)
        self.metadata = self.metadata[:cp] if train else self.metadata[cp:]

        self.sample_rate = A.hp.sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.embds_dn = embds_dn

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        wav_path = self.metadata[index]
        wav = A.load_wav(wav_path)      # compatible with preprocessor
        wav = A.trim_silence(wav)
        wav = A.align_wav(wav)
        wav = np.pad(wav, (40, 40))     # compatible with HuBERT-fx_net
        wav = torch.from_numpy(wav).float()

        # code is either raw pit/dyn before qt
        codes_path = self.embds_dir / wav_path.relative_to(self.wavs_dir)
        codes = np.load(codes_path.with_suffix(".npy"))

        # qt
        if self.embds_dn == 'pits':
            codes = A.qt_pit(codes)
            assert 0 <= codes.min() and codes.max() < A.hp.n_bin_pit
        elif self.embds_dn == 'dyns':
            codes = A.qt_dyn(codes)
            assert 0 <= codes.min() and codes.max() < A.hp.n_bin_dyn
        else: raise
        codes = torch.from_numpy(codes).long()

        return wav.unsqueeze(0), codes

    def collate(self, batch):
        wavs, codes = zip(*batch)
        wavs, codes = list(wavs), list(codes)

        wav_lengths  = [ wav.size(-1) for  wav in  wavs]
        code_lengths = [code.size(-1) for code in codes]

        wav_frames = min(self.max_samples, *wav_lengths)

        collated_wavs, wav_offsets = [], []
        for wav in wavs:
            wav_diff = wav.size(-1) - wav_frames
            wav_offset = random.randint(0, wav_diff)
            wav = wav[:, wav_offset : wav_offset + wav_frames]

            collated_wavs.append(wav)
            wav_offsets.append(wav_offset)

        rate = self.label_rate / self.sample_rate
        code_offsets = [round(wav_offset * rate) for wav_offset in wav_offsets]
        code_frames = round(wav_frames * rate)
        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset : code_offset + code_frames]
            collated_codes.append(code)

        wavs = torch.stack(collated_wavs, dim=0)
        codes = torch.stack(collated_codes, dim=0)

        return wavs, codes
